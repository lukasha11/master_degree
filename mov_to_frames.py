import cv2
import os


output_folder = "test2"
os.makedirs(output_folder, exist_ok=True)

video_path = "C:/studia/magisterka/Praca magisterska/test/1.MOV"
cap = cv2.VideoCapture(video_path)
#test git
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_filename = os.path.join(output_folder, f"frame_{count:04d}.png")
    cv2.imwrite(frame_filename, frame)
    count += 1

cap.release()