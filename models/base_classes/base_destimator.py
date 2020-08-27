from abc import ABC, abstractmethod


class BaseDepthEstimator(ABC):
    def __init__(self, *args, **kwargs):
        """Init method for class"""
        pass

    @abstractmethod
    def estimate(self, *args, **kwargs):
        """Method for depth estimation from monocular image/video"""
        pass
