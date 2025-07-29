import os
import cv2
import base64
import json
import argparse
import time
from collections import deque
from tqdm import tqdm
from dotenv import load_dotenv

import dspy
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Inject MISTRAL_API_KEY into OPENAI_API_KEY if needed
if os.environ.get("MISTRAL_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["MISTRAL_API_KEY"]

mistral_api = "https://api.mistral.ai/v1"
mistral_model = "mistral-small"

# Configure DSPy to use Mistral without truncate (manual truncation is handled)
dspy.configure(
    lm=dspy.LM(
        provider="openai",
        model=mistral_model,
        api_base=mistral_api,
    )
)

class Pixtrale(dspy.Signature):
    image = dspy.InputField()
    prompt_variant = dspy.InputField()
    caption = dspy.OutputField(desc="Advice caption")

pixtrale = dspy.Predict(Pixtrale)

class MotoAdvisor:
    def __init__(self, resume=False):
        self.model = YOLO("yolov8n.pt")
        print("YOLOv8n model loaded")
        self.resume = resume

        self.pixtrale_prompts = [
            "Analyze the scene at {DESCRIPTION_SCÈNE} from a motorcyclist's point of view. Describe traffic, road layout, and key objects: {LISTE_OBJETS}. Keep your description short (3 sentences max).",
            #"Describe what a motorbike rider sees in {DESCRIPTION_SCÈNE}. Mention close vehicles, road risks, and objects: {LISTE_OBJETS}. Limit to 3 sentences or less.",
            "Evaluate riding conditions at {DESCRIPTION_SCÈNE}. Include road risks, moving objects, and relevant elements: {LISTE_OBJETS}. Be concise: 3 sentences max.",
            "Simulate the motorcyclist’s view at {DESCRIPTION_SCÈNE}. Highlight obstacles and nearby objects: {LISTE_OBJETS}. Keep it brief to avoid verbosity.",
            "From a rider’s perspective, describe {DESCRIPTION_SCÈNE}. Focus on space, visibility, and surroundings: {LISTE_OBJETS}. Use under 100 words."
]


        self.llama_prompts = [
            "Based on {DESCRIPTION_SCÈNE} and objects {LISTE_OBJETS}, write 2 riding tips with max 6 words. If there is no immediate threat, simply advise to stay alert.",
            "From the scene {DESCRIPTION_SCÈNE}, list two actions (in 6 words) a motorcyclist should take based on {LISTE_OBJETS}. If safe, recommend maintaining current course.",
            "If {LISTE_OBJETS} shows hazards in {DESCRIPTION_SCÈNE}, suggest 2 quick reactions in max 6 words. Otherwise, say that no action is needed besides staying aware.",
            #"Looking at {DESCRIPTION_SCÈNE} and {LISTE_OBJETS}, give up to three relevant tips. If nothing risky appears, return a calm recommendation.",
            "Generate riding advice in 6 words for {DESCRIPTION_SCÈNE} from {LISTE_OBJETS}. Mention danger if any. If clear, advise staying focused but relaxed."
        ]

        self.buffer = deque(maxlen=5)

    def truncate_prompt(self, prompt, max_words=300):
        words = prompt.split()
        return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")

    def extract_frames(self, video_path, sample_interval=30, max_frames=20, start_frame=0):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS={fps:.2f} → sampling every {sample_interval} frames (max {max_frames})")

        frames = []
        count = 0
        extracted = 0

        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if count >= start_frame and (count - start_frame) % sample_interval == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frames.append((frame, count, timestamp))
                extracted += 1

            count += 1


        cap.release()
        return frames

    def encode_image_to_base64(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

    # Resize si trop large
        max_width = 640
        if pil_image.width > max_width:
            ratio = max_width / pil_image.width
            new_size = (max_width, int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size)

    # Compression JPEG
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=60)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


    def synthesize_advices(self, frame_advices):
        advice_texts = [a["advice"] for a in frame_advices]
        joined = "\n".join(f"- {text}" for text in advice_texts)
        prompt = (
            "You are a helpful, calm motorcycle riding assistant. Based on the following 25 advice suggestions:\n"
            f"{joined}\n\n"
            "Summarize them into a single concise instruction, no longer than 6 words, that’s reassuring, actionable, and clear."
        )
        time.sleep(1.5)
        return pixtrale(image="", prompt_variant=prompt).caption

    def synthesize_global_advice(self):
        if len(self.buffer) < 3:
            return None
        prompt = (
            "Summarize the situation for the motorcyclist based on recent instructions:\n"
            + "\n".join(f"- {a}" for a in self.buffer) +
            "\n\nProvide one gentle, clear sentence with your best advice. no longer than 6 words."
        )
        time.sleep(1.5)
        return pixtrale(image="", prompt_variant=prompt).caption

    def is_dangerous(self, obj):
        if obj["class_name"] not in ["car", "truck", "bus", "person", "bicycle"]:
            return False
        if obj["confidence"] < 0.6:
            return False

        x1, y1, x2, y2 = obj["bbox"]
        bbox_area = (x2 - x1) * (y2 - y1)
        return bbox_area > 5000  # tune this threshold if needed

    def process_video(self, video_path, max_frames=20, start_frame=0):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_folder = os.path.join("results", video_id, "frames")
        os.makedirs(frame_folder, exist_ok=True)

        frames = self.extract_frames(video_path, max_frames=max_frames, start_frame=start_frame)

        all_results = []
        enhanced_results = []
        processed_frames = set()
        global_start = time.time()

        results_path = os.path.join("results", "results.jsonl")
        enhanced_path = os.path.join("results", "enhanced_results.jsonl")


        if self.resume and os.path.exists(results_path):
            with open(results_path) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_frames.add(data["frame_id"])
                        all_results.append(data)
                    except:
                        continue


        for image, frame_id, timestamp in tqdm(frames, desc="Processing frames"):
            frame_start = time.time()
            if frame_id in processed_frames:
                continue

            b64 = self.encode_image_to_base64(image)
            frame_data = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image_path": os.path.join(frame_folder, f"frame_{frame_id}.jpg"),
                "objects": [],
                "advice_variants": []
            }
            cv2.imwrite(frame_data["image_path"], image)

            detection = self.model(image)[0]
            for box in detection.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.2:
                    cls_name = self.model.names[cls_id]
                    bbox = box.xyxy[0].tolist()
                    frame_data["objects"].append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": bbox
                    })

            description_scene = f"Frame ID {frame_id} at {timestamp:.2f}s"
            max_obj = 10
            liste_objets = ", ".join([f"{o['class_name']} ({o['confidence']:.2f})" for o in frame_data["objects"][:max_obj]])

            for pix_prompt_template in self.pixtrale_prompts:
                # Ajoute consigne de concision au prompt
                constrained_prompt = pix_prompt_template + " Keep it under 3 short sentences."
                pix_prompt = constrained_prompt.replace("{DESCRIPTION_SCÈNE}", description_scene).replace("{LISTE_OBJETS}", liste_objets)
                pix_prompt = self.truncate_prompt(pix_prompt, max_words=300)

                try:
                    description = pixtrale(image=b64, prompt_variant=pix_prompt).caption
                    description = self.truncate_prompt(description, max_words=100)
                    if "too large" in description.lower():
                        raise ValueError("Pixtrale output too large")
                    time.sleep(1.5)
                except Exception as e:
                    description = f"Error: {str(e)}"

                for llama_prompt_template in self.llama_prompts:
                    llama_prompt = llama_prompt_template.replace("{DESCRIPTION_SCÈNE}", description).replace("{LISTE_OBJETS}", liste_objets)
                    llama_prompt = self.truncate_prompt(llama_prompt, max_words=300)

                    try:
                        if "too large" in llama_prompt.lower():
                            raise ValueError("Skipping LLaMA prompt: previous output too long")

                        advice = pixtrale(image=b64, prompt_variant=llama_prompt).caption
                        time.sleep(1.5)
                    except Exception as e:
                        advice = f"Error: {str(e)}"

                    frame_data["advice_variants"].append({
                        "pixtrale_prompt": pix_prompt,
                        "llama_prompt": llama_prompt,
                        "advice": advice
                    })


            all_results.append(frame_data)

            synthesized = self.synthesize_advices(frame_data["advice_variants"])
            self.buffer.append(synthesized)

            global_advice = self.synthesize_global_advice()

            danger = any(self.is_dangerous(o) for o in frame_data["objects"])
            if danger:
                final = frame_data["advice_variants"][0]["advice"]
                override = True
            else:
                final = global_advice or synthesized
                override = False
            
            frame_end = time.time()
            print(f"→ Frame {frame_id} processed in {frame_end - frame_start:.2f} seconds.")
            
            enhanced_results.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image_path": frame_data["image_path"],
                "synthesized_advice": synthesized,
                "global_advice": global_advice,
                "final_advice": final,
                "is_override": override
            })

        with open("results/results.jsonl", "w") as f:
            for entry in all_results:
                f.write(json.dumps(entry) + "\n")

        with open("results/enhanced_results.jsonl", "w") as f:
            for entry in enhanced_results:
                f.write(json.dumps(entry) + "\n")
                
        global_end = time.time()
        elapsed = global_end - global_start
        print(f"\nTotal pipeline duration: {elapsed:.2f} seconds.")
        print("Processing complete. Results saved to results/results.jsonl and results/enhanced_results.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum number of frames to process")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index")
    args = parser.parse_args()

    MotoAdvisor(resume=args.resume).process_video(args.video, max_frames=args.max_frames, start_frame=args.start_frame)


