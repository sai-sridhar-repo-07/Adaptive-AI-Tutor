import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None

        # Hook the gradients of the last conv layer
        self.model.conv2.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        """
        Returns Grad-CAM heatmap as numpy array
        """

        self.model.zero_grad()
        output = self.model(input_tensor)

        class_score = output[:, target_class]
        class_score.backward()

        gradients = self.gradients
        activations = self.model.feature_maps

        # Global average pooling
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (28, 28))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam
