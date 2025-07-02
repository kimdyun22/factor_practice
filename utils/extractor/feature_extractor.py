import torch

class CommonExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def extract(self, image_tensor):
        """
        image_tensor: torch.Tensor of shape (1, 3, 112, 112)
        Returns: feature vector of shape (1, 512)
        """
        with torch.no_grad():
            features = self.model(image_tensor)
        return features
