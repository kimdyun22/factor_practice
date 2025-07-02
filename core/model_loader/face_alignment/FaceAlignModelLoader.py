class FaceAlignModelLoader:
    def __init__(self, model_path, model_category, model_name):
        self.model_path = model_path
        self.model_category = model_category
        self.model_name = model_name

    def load_model(self):
        # Normally load model from file. Here return dummy model & config
        class DummyAlignModel:
            def to(self, device):
                return self
            def eval(self):
                return self
        return DummyAlignModel(), {}