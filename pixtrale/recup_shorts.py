import yt_dlp
import os

def download_youtube_video(url, video_filename):
    """
    Télécharge une vidéo YouTube dans 'pixtrale/shorts_clips/{video_filename}' avec qualité réduite (≤360p)
    """
    output_path = os.path.join("pixtrale", "shorts_clips", video_filename)

    ydl_opts = {
        'format': 'bv*[height<=360][ext=mp4]+ba[ext=m4a]/b[height<=360]',
        'outtmpl': output_path,
        'quiet': False,
        'merge_output_format': 'mp4',
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return output_path

# Liste des 6 URLs
urls = [
    "https://www.youtube.com/shorts/PpyeOpLMFt8",
    "https://www.youtube.com/shorts/EH1nd0PgW1c",
    "https://www.youtube.com/shorts/9CRjTEyhUeo",
    "https://www.youtube.com/shorts/isnj92Wi7wE",
    "https://www.youtube.com/shorts/ZGrYskt8TMo"
]

# ancien URLs
#    "https://www.youtube.com/shorts/H_a8X395c60", short_1
#   "https://www.youtube.com/shorts/OumTu5YT67U", short_2
#   "https://www.youtube.com/shorts/DPYJLvycb-A", short_3
#   "https://www.youtube.com/shorts/oswukP-GYYk", short_4
#   "https://www.youtube.com/shorts/lridoUOakHk", short_5
#   "https://www.youtube.com/shorts/xh1HRSLpjr0", short_6

# Téléchargement en boucle
for i, url in enumerate(urls, start=7):
    try:
        print(f"🔽 Téléchargement vidéo {i} (qualité ≤360p)...")
        filename = f"short_{i}.mp4"
        path = download_youtube_video(url, filename)
        print(f"✅ Vidéo {i} enregistrée dans : {path}")
    except Exception as e:
        print(f"❌ Erreur pour la vidéo {i} : {e}")
