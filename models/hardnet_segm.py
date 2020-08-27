import numpy as np
import torch
from utils import load_segm_model, preprocess_image

from models.base_classes.base_segmentator import BaseSegmentator


class HardNetSegm(BaseSegmentator):
    def __init__(self, path, num_classes=19):
        super().__init__()
        model = load_segm_model(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.n_classes = num_classes

    def segment(self, images):
        images, resized_img = preprocess_image(self.device, "", images)
        outputs = self.model(images)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        # decoded = decode_segmap(pred, range(self.n_classes))
        return pred

