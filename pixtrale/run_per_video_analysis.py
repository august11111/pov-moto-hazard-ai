import os
import shutil
import subprocess
import sys

VIDEOS_DIR = "pixtrale/shorts_clips"
BASE_RESULTS_DIR = "results"
BASE_REPORT_DIR = "report"

# Ne traiter que short_8 à short_11
allowed_videos = {"short_8.mp4", "short_9.mp4", "short_10.mp4", "short_11.mp4"}
video_files = [f for f in os.listdir(VIDEOS_DIR) if f in allowed_videos]
video_files.sort()


for video_file in video_files:
    video_name = os.path.splitext(video_file)[0]  # ex: "short_2"
    video_path = os.path.join(VIDEOS_DIR, video_file)

    results_dir = os.path.join(BASE_RESULTS_DIR, video_name)
    frames_dir = os.path.join(results_dir, "frames")
    report_path = os.path.join(BASE_REPORT_DIR, f"{video_name}.pdf")

    # Nettoyer les anciens résultats de cette vidéo
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(BASE_REPORT_DIR, exist_ok=True)

    print(f"\n🎥 === TRAITEMENT DE {video_name} ===")
    print("▶️ Étape 1 : Analyse vidéo via moto_advice_pipeline.py\n")

    # Appel direct au pipeline avec sortie en direct (stdout/stderr)
    process1 = subprocess.Popen(
        [sys.executable, "moto_advice_pipeline.py", "--video", video_path, "--max-frames", "5", "--start-frame", "300"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    process1.wait()

    # Déplacer les fichiers de résultats dans le dossier dédié
    for fname in ["results.jsonl", "enhanced_results.jsonl"]:
        src = os.path.join(BASE_RESULTS_DIR, fname)
        dst = os.path.join(results_dir, fname)
        if os.path.exists(src):
            shutil.move(src, dst)

    print("\n📄 Étape 2 : Génération du PDF via generate_report_pdf.py\n")

    env_vars = os.environ.copy()
    env_vars["RESULTS_PATH"] = os.path.join(results_dir, "enhanced_results.jsonl")
    env_vars["FRAMES_FOLDER"] = frames_dir
    env_vars["OUTPUT_PATH"] = report_path

    # Appel direct à generate_report_pdf avec logs visibles
    process2 = subprocess.Popen(
        [sys.executable, "generate_report_pdf.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env_vars
    )
    process2.wait()

    print(f"✅ Rapport généré pour {video_name} → {report_path}\n")
    print("========================================\n")
