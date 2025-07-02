class FaceDetModelHandler:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.model.to(device)
        self.model.eval()

    def inference_on_image(self, image):
        # Dummy face detection
        # Normally, you'd use a detection model like RetinaFace, MTCNN, etc.
        # Here, we simulate detection with a hard-coded box: [x1, y1, x2, y2]
        h, w, _ = image.shape
        return [[w*0.3, h*0.3, w*0.7, h*0.7]]