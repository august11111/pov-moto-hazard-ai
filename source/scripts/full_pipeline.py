import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from source.utils.youtube_utils import download_youtube_video
from source.scripts.extract_frames import extract_frames
from source.scripts.generate_captions import generate_captions_for_folder
from source.utils.yolo_utils import detect_objects_yolo
from source.utils.llama_utils import ask_llama

def full_analysis_pipeline(yturl=None, interval=1):
    # === Dossiers et noms déjà définis pour les frames existantes ===
    video_name = "ytb_video"  # pas "ytb_video_pipeline"
    frames_dir = os.path.join("test", "data", "frames", video_name)
    captions_csv = os.path.join("test", "data", "descriptions", f"{video_name}.csv")
    analysis_csv = os.path.join("test", "data", "analyses", f"{video_name}_analyses.csv")

    # === Étapes qu'on saute car déjà faites ===
    # if yturl:
    #     video_filename = f"{video_name}.mp4"
    #     video_path = download_youtube_video(yturl, video_filename)
    #     extract_frames(video_path, frames_dir, interval)
    generate_captions_for_folder(frames_dir, captions_csv)

    # === YOLO + LLaMA ===
    df = pd.read_csv(captions_csv)
    results = []

    for _, row in df.iterrows():
        frame_path = os.path.join(frames_dir, row["image"])
        caption = row["description"]

        # YOLO
        objects = detect_objects_yolo(frame_path)
        object_text = ", ".join(objects) if objects else "aucun objet détecté"

        # Prompt LLaMA
        prompt = f"""
Tu es un assistant expert en sécurité moto.
Analyse la situation suivante et donne des conseils pour éviter un accident.
Voici la description automatique : {caption}
Éléments détectés : {object_text}
Réponds en 1 ou 2 phrases.
        """.strip()

        print(f"[{row['image']}] → Prompt envoyé à LLaMA...")
        answer = ask_llama(prompt)

        results.append({
            "image": row["image"],
            "description": caption,
            "objects": object_text,
            "analyse_llama": answer
        })

    # === Sauvegarde de l'analyse ===
    output_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(analysis_csv), exist_ok=True)
    output_df.to_csv(analysis_csv, index=False)
    print(f"\n✅ Analyse complète sauvegardée dans {analysis_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline complet : YouTube → Analyse LLaMA")
    parser.add_argument("--yturl", help="Lien de la vidéo YouTube (optionnel si frames déjà prêtes)")
    parser.add_argument("--interval", type=int, default=1, help="Intervalle entre les frames (en secondes)")
    args = parser.parse_args()

    full_analysis_pipeline(args.yturl, args.interval)
