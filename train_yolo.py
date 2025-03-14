from ultralytics import YOLO
import torch

yaml_path = "C:/Users/lukas/PycharmProjects/pythonProject/master_degree/coco_dataset.yaml"

model = YOLO("yolov8n.pt")  # Można zmienić na yolov8s.pt, yolov8m.pt, yolov8l.pt

model.train(
    data=yaml_path,  # Używamy nowego zbioru danych labels_data/
    epochs=10,  # Minimum 50 epok, aby model nauczył się poprawnie
    batch=16,  # Dostosowane do ilości VRAM (możesz zwiększyć, jeśli masz mocne GPU)
    imgsz=960,  # Optymalny rozmiar dla obrazów 1280x720
    name="yolo_car_detection_labels_data",  # Nowa nazwa eksperymentu
    workers=16,  # Przyspiesza ładowanie danych
    device="cuda" if torch.cuda.is_available() else "cpu"  # Wykorzystuje GPU jeśli dostępne
)

print("✅ Trening zakończony z nowymi danymi labels_data!")
