import cv2
import os

output_folder = "test5_pokoj"
os.makedirs(output_folder, exist_ok=True)

video_path = "C:/studia/magisterka/Praca magisterska/test/4.MOV"
cap = cv2.VideoCapture(video_path)

count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % 2 == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    count += 1

cap.release()

