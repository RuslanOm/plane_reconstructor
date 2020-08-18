import time
import cv2
import numpy as np
import torch
from depth_estimator.BTS import BtsController

from utils import load_segm_model, load_depth_model, preprocess_image


class PlaneDetector:
    def __init__(self, path_depth, path_segm):
        self.depth_model = load_depth_model(path_depth)
        self.segm_model = load_segm_model(path_segm)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_segm_map(self, path="", img=None):
        images, resized_img = preprocess_image(self.device, path, img)

        start = time.time()
        outputs = self.segm_model(images)
        procc_time = time.time() - start
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        return pred, procc_time

    def get_depth_map(self, path="", img=None):
        if path:
            img = cv2.imread(path)
        if img.shape != (640, 480, 3):
            img = cv2.resize(img, (640, 480))
        preds = self.depth_model.predict(img, is_channels_first=False, normalize=True)
        return preds

    def get_segmented_depth(self, img, interested_classes):
        seg_map, _ = self.get_segm_map("", img)
        depth_map = self.get_depth_map("", img)
        vect_func = np.vectorize(lambda x: x in set(interested_classes))
        map_arr = vect_func(seg_map)
        return BtsController.depth_map_to_rgbimg(depth_map * map_arr)
