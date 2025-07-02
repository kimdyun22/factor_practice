# make_labels.py (수정 버전)
import os
import numpy as np

real_dir = 'aligned_output/real'
fake_dir = 'aligned_output/fake'

real_files = [f for f in os.listdir(real_dir) if f.lower().endswith('.jpg')]
fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith('.jpg')]

labels = np.array([0]*len(real_files) + [1]*len(fake_files))
print(f"real: {len(real_files)}, fake: {len(fake_files)}, total: {len(labels)}")

np.save('extracted_features/labels.npy', labels)
