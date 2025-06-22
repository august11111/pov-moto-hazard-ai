import yt_dlp
import os

def download_youtube_video(url, video_filename):
    """
    Télécharge une vidéo YouTube dans 'test/data/videos/{video_filename}'
    """
    output_path = os.path.join("test", "data", "videos", video_filename)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]',
        'outtmpl': output_path,
        'quiet': False
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return output_path  # On retourne le chemin pour pouvoir l'utiliser
