from my_utils.scene_understanding import generate_rich_description
from my_utils.blip_utils import generate_captions
from my_utils.llama_utils import ask_llama
import os
import pandas as pd

# === Config
input_dir = "data/frames_custom"
output_csv = "data/outputs/analysis.csv"

# === Generate BLIP captions
captions = generate_captions(input_dir)

results = []

for fname, caption in captions:
    img_path = os.path.join(input_dir, fname)
    scene_description = generate_rich_description(img_path)

    # English prompt for LLaMA
    prompt = f"""
You are a motorcycle safety expert. Below is information extracted from an onboard scene image:

- Visual description (automatically generated):
{caption}

- Scene context (detected objects, weather, speed, etc.):
{scene_description}

Your tasks:
- Identify visible or likely dangers in this scene.
- Provide specific and useful advice to the rider.
- If no risk is apparent, reply: \"Nothing to report for this scene.\"

Respond clearly and concisely in 1–2 sentences.
"""

    print(f"\U0001F9E0 Analyzing {fname}...")
    analyse = ask_llama(prompt)

    results.append({
        "image": fname,
        "caption_blip": caption,
        "scene_description": scene_description,
        "analyse_llama": analyse
    })

# === Save output
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ Full analysis saved to: {output_csv}")
