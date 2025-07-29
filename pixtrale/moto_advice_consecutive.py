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

load_dotenv()

if os.environ.get("MISTRAL_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["MISTRAL_API_KEY"]

mistral_api = "https://api.mistral.ai/v1"
mistral_model = "mistral-small"

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

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

class MotoAdvisor:
    def __init__(self, resume: bool = False, output_dir: str = "results_temporal"):
        self.model = YOLO("yolov8n.pt")
        print("YOLOv8n model loaded")
        self.resume = resume
        self.output_dir = output_dir
        self.buffer = deque(maxlen=5)
        self.previous_description = None
        self.previous_objects = []

    def truncate_prompt(self, prompt: str, max_words: int = 300) -> str:
        words = prompt.split()
        return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")

    def extract_frames(self, video_path: str, sample_interval: int = 30, max_frames: int = 20, start_frame: int = 0):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS={fps:.2f} → sampling every {sample_interval} frames (max {max_frames})")
        frames, count, extracted = [], 0, 0
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
        max_width = 640
        if pil_image.width > max_width:
            ratio = max_width / pil_image.width
            pil_image = pil_image.resize((max_width, int(pil_image.height * ratio)))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=60)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def is_dangerous(self, obj):
        if obj["class_name"] not in ["car", "truck", "bus", "person", "bicycle"]:
            return False
        if obj["confidence"] < 0.6:
            return False
        x1, y1, x2, y2 = obj["bbox"]
        return (x2 - x1) * (y2 - y1) > 5000

    def describe_changes(self, prev_objs, curr_objs):
        changes = []
        for curr in curr_objs:
            matched = False
            for prev in prev_objs:
                if curr["class_id"] == prev["class_id"]:
                    iou = compute_iou(curr["bbox"], prev["bbox"])
                    if iou > 0.3:
                        matched = True
                        break
            if not matched:
                changes.append(f"New {curr['class_name']}")
        return ", ".join(changes) if changes else "No significant change"

    def synthesize_advices(self, frame_advices):
        advice_texts = [a["advice"] for a in frame_advices]
        joined = "\n".join(f"- {text}" for text in advice_texts)
        prompt = (
            "You are a helpful, calm motorcycle riding assistant.\n"
            f"{joined}\n\n"
            "Summarize them into a single clear instruction, no longer than 6 words."
        )
        time.sleep(1.5)
        return pixtrale(image="", prompt_variant=prompt).caption

    def synthesize_global_advice(self):
        if len(self.buffer) < 3:
            return None
        advice_lines = []
        for item in self.buffer:
            advice_lines.append(f"- Advice: {item['advice']} | Scene: {item['desc']} | Changes: {item['changes']}")
        prompt = (
            "You're an expert motorcycle safety assistant. Based on recent scene evolution and advice:\n"
            + "\n".join(advice_lines) +
            "\n\nGive ONE context-specific tip, max 6 words."
        )
        time.sleep(1.5)
        return pixtrale(image="", prompt_variant=self.truncate_prompt(prompt)).caption

    def process_video(self, video_path: str, max_frames: int = 20, start_frame: int = 0, output_dir: str = None):
        if output_dir:
            self.output_dir = output_dir

        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(self.output_dir, video_id)
        frame_folder = os.path.join(output_folder, "frames")
        os.makedirs(frame_folder, exist_ok=True)

        results_path = os.path.join(output_folder, "results.jsonl")
        enhanced_path = os.path.join(output_folder, "enhanced_results.jsonl")

        frames = self.extract_frames(video_path, max_frames=max_frames, start_frame=start_frame)
        all_results, enhanced_results = [], []
        global_start = time.time()

        for image, frame_id, timestamp in tqdm(frames, desc="Processing frames"):
            img_path = os.path.join(frame_folder, f"frame_{frame_id}.jpg")
            cv2.imwrite(img_path, image)
            b64 = self.encode_image_to_base64(image)

            detection = self.model(image)[0]
            objects = []
            for box in detection.boxes:
                if float(box.conf[0]) > 0.2:
                    cls_id = int(box.cls[0])
                    cls = self.model.names[cls_id]
                    bbox = box.xyxy[0].tolist()
                    objects.append({
                        "class_id": cls_id,
                        "class_name": cls,
                        "confidence": float(box.conf[0]),
                        "bbox": bbox
                    })

            changes_summary = self.describe_changes(self.previous_objects, objects)
            object_summary = ", ".join(f"{o['class_name']} ({o['confidence']:.2f})" for o in objects[:10])

            description_prompt = (
                f"Previously: {self.previous_description or 'N/A'}.\n"
                f"Now: {object_summary}.\n"
                f"Changes: {changes_summary}.\n"
                "Summarize current scene for motorcyclist."
            )
            try:
                description = pixtrale(image=b64, prompt_variant=self.truncate_prompt(description_prompt)).caption
                time.sleep(1.5)
            except Exception as e:
                description = f"Error: {e}"

            advice_prompt = f"Scene: {description}. Objects: {object_summary}. Suggest two actions (6 words max)."
            try:
                advice = pixtrale(image=b64, prompt_variant=self.truncate_prompt(advice_prompt)).caption
                time.sleep(1.5)
            except Exception as e:
                advice = f"Error: {e}"

            synthesized = self.synthesize_advices([{ "advice": advice }])
            self.buffer.append({"advice": synthesized, "desc": description, "changes": changes_summary})
            global_advice = self.synthesize_global_advice()

            danger = any(self.is_dangerous(o) for o in objects)
            final_advice = advice if danger else (global_advice or synthesized)

            enhanced_results.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image_path": img_path,
                "synthesized_advice": synthesized,
                "global_advice": global_advice,
                "final_advice": final_advice,
                "is_override": danger
            })

            all_results.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image_path": img_path,
                "objects": objects,
                "description": description,
                "advice": advice,
                "changes": changes_summary
            })

            self.previous_description = description
            self.previous_objects = objects

        with open(results_path, "w") as f:
            for entry in all_results:
                f.write(json.dumps(entry) + "\n")
        with open(enhanced_path, "w") as f:
            for entry in enhanced_results:
                f.write(json.dumps(entry) + "\n")

        print(f"\n✅ Completed in {time.time() - global_start:.2f} seconds. Results saved to: {enhanced_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--output-dir", default="results_temporal")
    args = parser.parse_args()

    MotoAdvisor(resume=args.resume, output_dir=args.output_dir).process_video(
        video_path=args.video,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        output_dir=args.output_dir
    )