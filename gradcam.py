import torch
import numpy as np
import torch.nn as nn

def generate_gradcam(model, image_tensor, target_layer=None):
    """
    Generates a Grad-CAM heatmap for the top predicted class.
    Returns a 2D numpy array (H, W) normalized to [0, 1].

    Notes:
    - model should be on the same device as image_tensor.
    - image_tensor should have shape (1, C, H, W).
    """

    model.eval()

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        # grad_out is a tuple
        gradients.append(grad_out[0].detach())

    # Determine target layer: use provided, otherwise try common locations, otherwise find last Conv2d
    if target_layer is None:
        try:
            # common for many torchvision models
            target_layer = model.features[-1][0]
        except Exception:
            # fallback: find last Conv2d in model
            last_conv = None
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is None:
                raise RuntimeError("No Conv2d layer found in model to compute Grad-CAM.")
            target_layer = last_conv

    forward_handle = target_layer.register_forward_hook(forward_hook)
    # prefer register_full_backward_hook when available
    try:
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
    except AttributeError:
        backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass (we need gradients for Grad-CAM so don't use no_grad here)
    output = model(image_tensor)
    pred_class = torch.argmax(output, dim=1).item()

    model.zero_grad()
    score = output[0, pred_class]
    score.backward(retain_graph=True)

    if not gradients or not activations:
        forward_handle.remove()
        backward_handle.remove()
        raise RuntimeError("Failed to collect gradients/activations for Grad-CAM. Hooks may not have been called.")

    grads = gradients[0]      # shape: (B, C, H, W)
    acts = activations[0]     # shape: (B, C, H, W)

    # Global-average-pool the gradients
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # shape (B, C, 1, 1)

    cam = torch.sum(weights * acts, dim=1, keepdim=False)  # shape (B, H, W)
    cam = torch.relu(cam)

    cam = cam.squeeze().cpu().numpy()

    # Normalize to [0,1], guarding against zero division
    cam = cam - np.min(cam)
    max_val = np.max(cam)
    if max_val > 0:
        cam = cam / max_val

    forward_handle.remove()
    backward_handle.remove()

    return cam
