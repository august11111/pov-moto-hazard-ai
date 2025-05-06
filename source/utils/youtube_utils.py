import yt_dlp
import os
import subprocess

def download_youtube_video(url, video_filename):
    """
    Télécharge la vidéo puis la convertit en H.264 si nécessaire.
    """
    output_dir = os.path.join("test", "data", "videos")
    output_path = os.path.join(output_dir, video_filename)
    tmp_path = os.path.join(output_dir, "temp_download.mp4")

    ydl_opts = {
        'format': 'bv*[ext=mp4]/bestvideo[ext=mp4]/best',
        'outtmpl': tmp_path,
        'quiet': False
    }

    os.makedirs(output_dir, exist_ok=True)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Re-encode en H.264 (avc1)
    print("🔁 Conversion en H.264 avec ffmpeg...")
    convert_cmd = [
        '/home/centralesupelec/ffmpeg-7.0.2-amd64-static/ffmpeg',  # ← Mets ici le chemin complet vers le ffmpeg téléchargé
        '-y', '-i', tmp_path,
        '-c:v', 'libx264', '-preset', 'fast',
        '-an',
        output_path
    ]

    subprocess.run(convert_cmd, check=True)

    # Nettoyage
    os.remove(tmp_path)

    print(f"\n✅ Vidéo convertie et sauvegardée : {output_path}")
    return output_path
