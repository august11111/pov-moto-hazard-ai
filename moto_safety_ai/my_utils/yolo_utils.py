import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

def detect_objects_yolo(img_path):
    results = model(img_path)
    detected = results.pandas().xyxy[0]
    return detected['name'].tolist()
