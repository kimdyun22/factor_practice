import cv2
import os

video_path = r"D:/FaceForensics/data/manipulated_sequences/Deepfakes/c40/videos/000_003.mp4"
output_dir = r"D:/FaceForensics/input_images/fake"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_count >= 16:
        break
    out_path = os.path.join(output_dir, f"frame_{frame_count:03d}.jpg")
    cv2.imwrite(out_path, frame)
    frame_count += 1

cap.release()
print(f"✅ {frame_count} fake frames saved to {output_dir}")
