import sys
import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('..')
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory
import torch.nn.functional as F
from tqdm import tqdm
import os
import cv2
import numpy as np

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='extract features for megaface.')
    conf.add_argument("--data_conf_file", type = str,
                      help = "The path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type = str, default='AttentionNet',
                      help = "Resnet, Mobilefacenets.")

    conf.add_argument('--output_path', type=str, required=True,
                  help='path to save extracted features')

    conf.add_argument("--backbone_conf_file", type = str, default='./backbone_conf.yaml',
                      help = "The path of backbone_conf.yaml.")
    conf.add_argument('--batch_size', type = int, default = 1024)
    conf.add_argument('--model_path', type = str, default = './Epoch_17.pt',
                      help = 'The path of model')
    conf.add_argument('--input_root', help = 'path to input directory after detection and alignment')
    args = conf.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = 127.5
    std = 128.0

    # define model.
    # ✅ 수정
    # 수정된 형태
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    backbone = backbone_factory.get_backbone()

    model_loader = ModelLoader(backbone, args.backbone_type)  # ✅ model_name 인자 추가
    model = model_loader.load_model(args.model_path).eval()


    path_names = []
    
    all_embeddings = []

    for root, d_names, f_names in tqdm(os.walk(args.input_root), total=len(list(os.walk(args.input_root)))):
        for frame in f_names:
            if not frame.endswith('.jpg'):
                continue
            file = os.path.join(root, frame)
            image = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (112, 112))
            image = (image.transpose((2, 0, 1)) - mean) / std
            image = torch.from_numpy(image.astype(np.float32))
            images = image[None].to(device)
            features = model(images)
            all_embeddings.append(features.detach().cpu().numpy())

    if len(all_embeddings) > 0:
        all_embeddings = np.concatenate(all_embeddings, 0)
        os.makedirs(args.output_path, exist_ok=True)
        np.save(os.path.join(args.output_path, 'features.npy'), all_embeddings)
