import cv2
import numpy as np
import torch
import time
from segmnetator.test import init_model

from base_segmentator import BaseSegmentator

n_classes = 19
colors = [#[0, 0, 0],
          [128, 64, 128],
          [244, 35, 232],
          [70, 70, 70],
          [102, 102, 156],
          [190, 153, 153],
          [153, 153, 153],
          [250, 170, 30],
          [220, 220, 0],
          [107, 142, 35],
          [152, 251, 152],
          [0, 130, 180],
          [220, 20, 60],
          [255, 0, 0],
          [0, 0, 142],
          [0, 0, 70],
          [0, 60, 100],
          [0, 80, 100],
          [0, 0, 230],
          [119, 11, 32],
          ]

class_names = [
    # "unlabelled",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
label_colours = dict(zip(range(19), colors))


class HardNetSegm(BaseSegmentator):
    def __init__(self, path, class_labels, n_classes=19):
        super().__init__()
        device, model = init_model({"model_path": path})
        self.device = device
        self.model = model
        self.class_labels = class_labels
        self.n_classes = n_classes

    def decode_segmap(self, temp, ls):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for item in ls:
            r[temp == item] = label_colours[item][0]
            g[temp == item] = label_colours[item][1]
            b[temp == item] = label_colours[item][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def segment(self, img_path="", img=False, size=(640, 480), device=torch.device("cpu")):
        assert img_path or img, "Path or img must be identified"
        if img_path:
            img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
        img = img_resized.astype(np.float16)

        # norm
        value_scale = 255
        mean = [0.406, 0.456, 0.485]
        mean = [item * value_scale for item in mean]
        std = [0.225, 0.224, 0.229]
        std = [item * value_scale for item in std]
        img = (img - mean) / std

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()

        images = img.to(device)
        start = time.time()
        outputs = self.model(images)
        procc_time = time.time() - start
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = self.decode_segmap(pred, range(self.n_classes))

        return img_resized, decoded, pred, procc_time

