import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import base64
import io
from torchvision import transforms

from backend.app.models.cnn_model import DigitCNN
from backend.app.explainability.gradcam import GradCAM


class InferenceService:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DigitCNN().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        self.gradcam = GradCAM(self.model)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess(self, image: Image.Image):
        return self.transform(image).unsqueeze(0).to(self.device)

    def _encode_heatmap(self, heatmap):
        heatmap = np.uint8(255 * heatmap)
        img = Image.fromarray(heatmap)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def predict(self, image: Image.Image):
        tensor = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        confidence, prediction = torch.max(probs, dim=1)
        pred_class = int(prediction.item())

        # Generate Grad-CAM
        heatmap = self.gradcam.generate(tensor, pred_class)
        heatmap_b64 = self._encode_heatmap(heatmap)

        return {
            "predicted_digit": pred_class,
            "confidence": float(confidence.item()),
            "gradcam_heatmap": heatmap_b64
        }
