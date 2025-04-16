import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import argparse
from source.utils.youtube_utils import download_youtube_video

def extract_frames(video_path, output_dir, interval_sec=2):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo")
        return 

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_sec)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if frame_count % interval_frames == 0:
            filename = f"frame_{saved_count:04}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"→ Image sauvegardée : {filename}")
            saved_count += 1
        
        frame_count += 1
        
    cap.release()
    print(f"\n✅ Extraction terminée : {saved_count} images sauvegardées dans '{output_dir}'")

# === Point d’entrée du script ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction de frames à partir d'une vidéo")
    parser.add_argument("--video", help="Chemin de la vidéo .mp4 (optionnel si --yturl)")
    parser.add_argument("--yturl", help="Lien YouTube de la vidéo à télécharger")
    parser.add_argument("--output", help="Dossier de sortie des frames (optionnel si --yturl)")
    parser.add_argument("--interval", type=int, default=1, help="Intervalle en secondes (défaut: 1)")

    args = parser.parse_args()

    # === Gestion vidéo locale OU YouTube ===
    if args.yturl:
        video_name = "ytb_video.mp4"
        video_path = download_youtube_video(args.yturl, video_name)

        folder_name = video_name.replace(".mp4", "")
        output_dir = os.path.join("test", "data", "frames", folder_name)
    else:
        if not args.video or not args.output:
            parser.error("Avec --video, tu dois aussi fournir --output.")
        video_path = args.video
        output_dir = args.output

    # === Lancer l'extraction ===
    extract_frames(video_path, output_dir, args.interval)
