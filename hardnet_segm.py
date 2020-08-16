import numpy as np
import torch
import time
from utils import load_segm_model, decode_segmap

from base_segmentator import BaseSegmentator


class HardNetSegm(BaseSegmentator):
    def __init__(self, path, num_classes=19):
        super().__init__()
        model = load_segm_model(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.n_classes = num_classes

    def segment(self, images):
        start = time.time()
        outputs = self.model(images)
        procc_time = time.time() - start
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = decode_segmap(pred, range(self.n_classes))
        return decoded, pred, procc_time

