from models.depth_estimator import BTS
from models.base_classes.base_destimator import BaseDepthEstimator


class BTSEstimator(BaseDepthEstimator):
    def __init__(self, path):
        super().__init__()
        self.model = BTS.BtsController()
        self.model.load_model(path)
        self.model.eval()

    def estimate(self, img):
        prediction = self.model.predict(img, is_channels_first=False, normalize=True)
        return prediction
