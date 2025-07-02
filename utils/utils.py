import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    paths.append(path)
    return images, paths

def save_features(features, paths, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for feat, path in zip(features, paths):
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        save_path = os.path.join(output_path, name + '.npy')
        np.save(save_path, feat.detach().cpu().numpy())
