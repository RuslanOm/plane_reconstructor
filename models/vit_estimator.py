import torch
from models.base_classes.base_destimator import BaseDepthEstimator
import cv2
import numpy as np

import models
from models.vit.util.io import write_pfm

from torchvision.transforms import Compose

from models.vit.dpt.models import DPTDepthModel
from models.vit.dpt.midas_net import MidasNet_large
from models.vit.dpt.transforms import Resize, NormalizeImage, PrepareForNet


class VITEstimator(BaseDepthEstimator):
    def __init__(self, model_type="dpt_large", model_path=''):
        super().__init__()
        self.model_type = model_type
        self.model_path = model_path
        self.transform = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self._init_model(self.model_type, self.device, self.model_path)

    def _init_model(self, model_type, device, model_path):
        if model_type == "dpt_large":
            net_w = net_h = 384
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid":  # vit-Hybrid
            net_w = net_h = 384
            model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid_kitti":
            net_w = 1216
            net_h = 352

            model = DPTDepthModel(
                path=model_path,
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid_nyu":
            net_w = 640
            net_h = 480

            model = DPTDepthModel(
                path=model_path,
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "midas_v21":  # Convolutional model
            net_w = net_h = 384

            model = MidasNet_large(model_path, non_negative=True)
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()
        model.to(device)

        return model

    def estimate(self, img, use_file='', remove_trade_mark=True):
        img_input = self.transform({"image": img})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            if remove_trade_mark:
                prediction[460:, 590:] = 0.0
                prediction[460:, :70] = 0.0
            # if model_type == "dpt_hybrid_kitti":
            #     prediction *= 256
            #
            # if model_type == "dpt_hybrid_nyu":
            #     prediction *= 1000.0
        if use_file:
            write_pfm(use_file, prediction.astype(np.float32))

        return prediction
