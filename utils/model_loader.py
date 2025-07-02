import torch
import torch.nn as nn

class ModelLoader:
    def __init__(self, backbone, model_name):
        self.model = backbone
        self.model_name = model_name

    def load_model(self, model_path):  # ✅ 반드시 인자를 받아야 합니다
        # 예: state_dict 로드
        return self.model


class DummyFeatureModel(nn.Module):
    def __init__(self):
        super(DummyFeatureModel, self).__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(112*112*3, 512)  # Assuming 112x112 RGB input

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return x