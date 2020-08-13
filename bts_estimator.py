from depth_estimator import BTS
import time

from base_destimator import BaseDepthEstimator


class BTSEstimator(BaseDepthEstimator):
    def __init__(self, path):
        super().__init__()
        self.model = BTS.BtsController()
        self.model.load_model(path)
        self.model.eval()

    def estimate(self, img):
        start = time.time()
        prediction = self.model.predict(img, is_channels_first=False, normalize=True)
        return prediction, time.time() - start
