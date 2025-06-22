import os
import cv2
import base64
import json
import argparse
import numpy as np
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
    """
    Generates a caption that gives driving advice to the rider given the frame.
    """
    image = dspy.InputField()
    prompt_variant = dspy.InputField()
    caption = dspy.OutputField(desc="Advice caption")

pixtrale = dspy.Predict(Pixtrale)

class MotoAdvisor:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        print("YOLOv8n model loaded")

        self.pixtrale_prompts = [
            "Analyze the scene at {DESCRIPTION_SCÈNE} from a motorcyclist's point of view. Describe traffic, road layout, and key objects: {LISTE_OBJETS}. If no threat is visible, say so.",
            "Describe what a motorbike rider sees in {DESCRIPTION_SCÈNE}. Mention close vehicles, road risks, and objects: {LISTE_OBJETS}. Only mention real risks.",
            "Evaluate riding conditions at {DESCRIPTION_SCÈNE}. Include road risks, moving objects, and relevant elements: {LISTE_OBJETS}. Say if the environment is clear.",
            "Simulate the motorcyclist’s view at {DESCRIPTION_SCÈNE}. Highlight obstacles and nearby objects: {LISTE_OBJETS}. If the road is clear, note it.",
            "From a rider’s perspective, describe {DESCRIPTION_SCÈNE}. Focus on space, visibility, and surroundings: {LISTE_OBJETS}. If no immediate danger, make it clear."
        ]

        self.llama_prompts = [
            "Based on {DESCRIPTION_SCÈNE} and objects {LISTE_OBJETS}, write 2 riding tips. If there is no immediate threat, simply advise to stay alert.",
            "From the scene {DESCRIPTION_SCÈNE}, list two actions a motorcyclist should take based on {LISTE_OBJETS}. If safe, recommend maintaining current course.",
            "If {LISTE_OBJETS} shows hazards in {DESCRIPTION_SCÈNE}, suggest 2 quick reactions. Otherwise, say that no action is needed besides staying aware.",
            "Looking at {DESCRIPTION_SCÈNE} and {LISTE_OBJETS}, give up to three relevant tips. If nothing risky appears, return a calm recommendation.",
            "Generate riding advice for {DESCRIPTION_SCÈNE} from {LISTE_OBJETS}. Mention danger if any. If clear, advise staying focused but relaxed."
        ]

    def truncate_prompt(self, prompt, max_words=300):
        words = prompt.split()
        return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")

    def extract_frames(self, video_path, sample_interval=30, max_frames=20):
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
            if count % sample_interval == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frames.append((frame, count, timestamp))
                extracted += 1
            count += 1

        cap.release()
        return frames

    def encode_image_to_base64(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def process_video(self, video_path):
        frames = self.extract_frames(video_path)
        results = []

        os.makedirs("results/frames", exist_ok=True)

        for image, frame_id, timestamp in tqdm(frames, desc="Processing frames"):
            b64 = self.encode_image_to_base64(image)
            frame_data = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image_path": f"results/frames/frame_{frame_id}.jpg",
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
                pix_prompt = pix_prompt_template.replace("{DESCRIPTION_SCÈNE}", description_scene).replace("{LISTE_OBJETS}", liste_objets)
                pix_prompt = self.truncate_prompt(pix_prompt, max_words=300)
                try:
                    description = pixtrale(image=b64, prompt_variant=pix_prompt).caption
                except Exception as e:
                    description = f"Error: {str(e)}"

                for llama_prompt_template in self.llama_prompts:
                    llama_prompt = llama_prompt_template.replace("{DESCRIPTION_SCÈNE}", description).replace("{LISTE_OBJETS}", liste_objets)
                    llama_prompt = self.truncate_prompt(llama_prompt, max_words=300)
                    try:
                        advice = pixtrale(image=b64, prompt_variant=llama_prompt).caption
                    except Exception as e:
                        advice = f"Error: {str(e)}"
                    frame_data["advice_variants"].append({
                        "pixtrale_prompt": pix_prompt,
                        "llama_prompt": llama_prompt,
                        "advice": advice
                    })

            results.append(frame_data)

        with open("results/results.jsonl", "w") as f:
            for entry in results:
                f.write(json.dumps(entry) + "\n")

        print("Processing complete. Results saved to results/results.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    args = parser.parse_args()

    MotoAdvisor().process_video(args.video)
