import torch
import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO

# 🔹 Załaduj model YOLOv8
model = YOLO("runs/detect/yolo_car_detection_labels_data5/weights/best.pt")  # Własny model YOLO (zmień na "yolov8n.pt" dla domyślnego)

# 🔹 Ustawienia przechwytywania ekranu
monitor = {"top": 100, "left": 100, "width": 1920, "height": 1080}  # Obszar do przechwycenia

# 🔹 Uruchom przechwytywanie ekranu w pętli
with mss.mss() as sct:
    while True:
        start_time = time.time()

        # Przechwyć ekran
        screen = sct.grab(monitor)
        frame = np.array(screen)

        # Konwersja na format RGB (YOLO wymaga)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Wykonaj detekcję YOLOv8
        results = model(frame)

        # Rysuj bounding boxy na ekranie
        result_img = results[0].plot()

        # Wyświetl obraz
        cv2.imshow("YOLO Screen Detection", result_img)

        # Wyjście po wciśnięciu "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # FPS info
        print(f"FPS: {1 / (time.time() - start_time):.2f}")

# Zamknij okno
cv2.destroyAllWindows()