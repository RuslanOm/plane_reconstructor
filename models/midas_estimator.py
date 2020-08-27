import torch
from models.base_classes.base_destimator import BaseDepthEstimator


class MIDASEstimator(BaseDepthEstimator):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.default_transform

    def estimate(self, img):
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        return output
