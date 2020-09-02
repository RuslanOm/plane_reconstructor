import numpy as np
import torch
from models.base_classes.base_destimator import BaseDepthEstimator
from models.base_classes.base_segmentator import BaseSegmentator


class PlaneDetector:
    def __init__(self, depth_model: BaseDepthEstimator, segm_model: BaseSegmentator):
        self.depth_model = depth_model
        self.segm_model = segm_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_segm_map(self, images):
        """
        Segments input images
        :param images: list of images or one image
        :return: segment maps
        """
        return self.segm_model.segment(images)

    def get_depth_map(self, img):
        """
        Estimates depth for input image
        :param img: input image
        :return: result depth map
        """
        return self.depth_model.estimate(img)

    def get_segmented_depth(self, img, interested_classes):
        """
        Estimates depth only in interested regions of image
        :param img: input image
        :param interested_classes: list of sets, classes for which depth will be estimated
        :return: Tuple of depth map, segm map, map arrays for depth
        """
        seg_map = self.get_segm_map(img)
        depth_map = self.get_depth_map(img)
        answer = []
        for item in interested_classes:
            vect_func = np.vectorize(lambda x: x in set(item))
            map_arr = vect_func(seg_map)
            answer.append(map_arr)
        return depth_map, seg_map, answer
