from ultralytics import YOLO
import cv2

# Wczytanie wytrenowanego modelu
model_path = "runs/detect/yolo_car_detection_labels_data5/weights/best.pt"
model = YOLO(model_path)

# Ścieżka do testowego obrazu
image_path = "png/tilton/images/tilton_frame_0640.png"

# Wykrywanie obiektów
results = model(image_path)

# Pobranie obrazu z zaznaczonymi obiektami
annotated_frame = results[0].plot()

# Wyświetlenie obrazu
cv2.imshow("Wynik detekcji", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opcjonalnie: Zapisz wynik detekcji
cv2.imwrite("C:/Users/lukas/PycharmProjects/pythonProject/master_degree/test_output.png", annotated_frame)

print("✅ Test detekcji zakończony! Obraz zapisany jako test_output.png")
