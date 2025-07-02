import cv2
import os

def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # 원래 FPS
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 매 'frame_rate // fps' 프레임마다 저장
        if count % max(1, frame_rate // fps) == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved:03d}.jpg"), frame)
            saved += 1
        count += 1

    cap.release()
    print(f"✅ {saved} frames saved to {output_dir}")

# 사용 예시
extract_frames(
    video_path="D:/FaceForensics/data/original_sequences/youtube/c40/videos/000.mp4",
    output_dir="D:/FaceForensics/input_images/real",
    fps=1  # 초당 1프레임
)
