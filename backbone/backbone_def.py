import torch.nn as nn
from utils.model_loader import DummyFeatureModel  # 우리가 만든 더미 모델

class BackboneFactory:
    def __init__(self, model_name, conf_file=None):
        self.model_name = model_name
        self.conf_file = conf_file  # 사용하지 않더라도 받기만 하면 됨

    def get_backbone(self):
        # conf_file은 지금은 사용하지 않지만, 확장성 고려해 인자로 받을 수 있도록 설정
        return DummyFeatureModel()

