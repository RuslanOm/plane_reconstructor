import cv2
from base_segmentator import BaseSegmentator
from base_destimator import BaseDepthEstimator


class PlaneDetector:

    def __init__(self, depth_model: BaseDepthEstimator, segm_model: BaseSegmentator):
        self.depth_model = depth_model
        self.segm_model = segm_model

    def get_segm_map(self, path="", img=None):
        assert path or not None, "Path or img must be identified"
        if path:
            img = cv2.imread(path)
            preds = self.segm_model.segment("", img, (640, 480))
        else:
            preds = self.segm_model.segment(img=img, size=(640, 480))
        return preds

    def get_depth_map(self, path="", img=None):
        assert path or not None, "Path or img must be identified"
        if path:
            img = cv2.imread(path)
        preds = self.depth_model.estimate(img)
        return preds