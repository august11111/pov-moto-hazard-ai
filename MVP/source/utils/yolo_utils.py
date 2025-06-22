import torch
from pathlib import Path

# Load YOLOv5 model (first time it va le télécharger)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

def detect_objects_yolo(img_path):
    """
    Prend le chemin d'une image et retourne une liste d'objets détectés.
    """
    results = model(img_path)
    detected = results.pandas().xyxy[0]  # DataFrame avec toutes les prédictions

    # On garde les noms d’objets détectés
    labels = detected['name'].tolist()
    return labels
