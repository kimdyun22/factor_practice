import cv2
import numpy as np

class FaceRecImageCropper:
    def __init__(self, image_size=(112, 112)):
        self.image_size = image_size
        self.dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def align(self, img, landmarks):
        """
        Aligns the face image based on 5 landmarks.

        Parameters:
        - img: Input BGR image
        - landmarks: 5x2 or 68x2 facial landmark coordinates

        Returns:
        - aligned face image (112x112)
        """
        if landmarks.shape[0] == 68:
            # Use standard 5 landmarks
            idx = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth
            landmarks = landmarks[idx]

        src = np.array(landmarks).astype(np.float32)
        tform = cv2.estimateAffinePartial2D(src, self.dst, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(img, tform, self.image_size, borderValue=0.0)
        return aligned

    def crop_image_by_mat(self, img, landmarks):
        """
        얼굴을 landmarks(5점) 기준으로 정렬된 얼굴 이미지를 생성하여 반환합니다.
        :param img: BGR 이미지
        :param landmarks: [x1, y1, x2, y2, ..., x5, y5] 형태의 1D array (총 10개)
        """
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks).reshape((5, 2))
        elif landmarks.shape[0] == 10:
            landmarks = landmarks.reshape((5, 2))
        else:
            raise ValueError("landmarks must be a 10-element 1D array or a (5, 2) array")

        src = np.array(landmarks).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, self.dst, method=cv2.LMEDS)
        aligned = cv2.warpAffine(img, M, self.image_size, borderValue=0.0)
        return aligned
