import cv2
import numpy as np
import torch

from segmnetator.utils import init_model, label_colours
from depth_estimator import BTS


def load_depth_model(path):
    model = BTS.BtsController()
    model.load_model(path)
    model.eval()
    return model


def load_segm_model(path):
    _, model = init_model({"model_path": path})
    return model


def decode_segmap(temp, ls):
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


def preprocess_image(device, img_path="", img=None, size=(640, 480)):
    if img_path:
        img = cv2.imread(img_path)
    if img.shape != (size[0], size[1], 3):
        img_resized = cv2.resize(img, (size[0], size[1]))  # uint8 with RGB mode
    else:
        img_resized = img
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

    return img.to(device), img_resized
