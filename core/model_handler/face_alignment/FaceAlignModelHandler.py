class FaceAlignModelHandler:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.model.to(device)
        self.model.eval()

    def inference_on_image(self, image, det):
        # Dummy landmark inference logic
        # Normally, the detection box would be used to crop & align face, and the model would predict landmarks
        # Here we return fixed ArcFace-style 5 landmarks for demonstration
        return [[38.2, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.3], [70.7, 92.2]]