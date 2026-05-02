import torch
import numpy as np

def generate_gradcam(model, image_tensor):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # ConvNeXt last convolution block
    target_layer = model.features[-1][0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)

    pred_class = torch.argmax(output)

    model.zero_grad()

    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2,3), keepdim=True)

    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = torch.relu(cam)

    cam = cam / torch.max(cam)

    cam = cam.detach().numpy()

    forward_handle.remove()
    backward_handle.remove()

    return cam