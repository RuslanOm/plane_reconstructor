from abc import ABC, abstractmethod


class BaseSegmentator(ABC):
    def __init__(self, *args, **kwargs):
        """Init method for segmentator"""
        pass

    @abstractmethod
    def segment(self, *args, **kwargs):
        """Method for depth segmentation from monocular image/video"""
        pass

    @abstractmethod
    def get_parted(self, *args, **kwargs):
        """Method for selecting only interested parts of image/video"""
        pass
