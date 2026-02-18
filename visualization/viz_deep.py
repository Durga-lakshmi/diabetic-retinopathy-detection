import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

import hydra
#from datasets import get_dataset
#from models import get_model
#from evaluator import Evaluator


def run_deep_viz(cfg, model, device, samples_test):

    if cfg.test.check_path is not None:
        model_version = os.path.basename(cfg.test.check_path).replace('.pth','')
    else:
        model_version = f"{save.date_prefix}_{cfg.model.name}"
    os.makedirs(cfg.deep_viz.save_viz_dir, exist_ok=True)
    

    #***
    target_layer = find_last_conv_layer(model)
    print("[INFO] Using target_layer:", target_layer)


    # Main Visualization Loop
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    #***
    for i, (input_tensor, label, prediction) in enumerate(samples_test):
        img_tensor = input_tensor.unsqueeze(0).to(device)

        img_np = input_tensor.cpu().numpy().transpose(1,2,0)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        
        target_class = int(label)

        save_viz_path=os.path.join(cfg.deep_viz.save_viz_dir, f"{model_version}_sample_{i+1}.png")

        #for Resnet
        model.apply(disable_inplace_relu)

        with torch.enable_grad():
            visualize_all(model,target_layer,img_tensor,img_np,target_class,save_viz_path)
            print(f"[INFO] Deep Visualization saved to: {cfg.deep_viz.save_viz_dir}")


def find_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model. GradCAM cannot run.")
    return last_conv


# for Resnet
def disable_inplace_relu(m):
    if isinstance(m, nn.ReLU):
        m.inplace = False



# ============================================================
# GradCAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()          # [B,C,H',W']
        #print("[DEBUG] FWD hook hit", output.shape)



    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()    # [B,C,H',W']
        #print("[DEBUG] BWD hook hit", grad_output[0].shape if grad_output and grad_output[0] is not None else None)



    def __call__(self, x, target_class=None):
        """
        target_class:
        if multi-classification, specify the class index for which to compute CAM
        if binary-classification, can be None
        """
        self.model.zero_grad()
        # other Model
        out = self.model(x)
        # below code only for Dense121
        #out = self.model(x.clone())


        # binary classes ->1
        if out.shape[1] == 1:
            logit = out.squeeze()
        else:
            assert target_class is not None
            logit = out[0, target_class]

        logit.backward(retain_graph=True)

        A = self.activations[0]      # [C,H',W']
        G = self.gradients[0]        # [C,H',W']
        #if self.gradients is None:
        #    raise RuntimeError("GradCAM: self.gradients is None, backward hook not used.")
        #G = self.gradients           

        weights = G.mean(dim=(1,2), keepdim=True)   # GAP over gradients
        cam = (A * weights).sum(dim=0)              # [H',W']

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()     # 2D CAM


# ============================================================
# Guided Backprop
# ============================================================
class GuidedBackprop:
    def __init__(self, model):
        self.model = model.eval()
        self._register_hooks()

    def _register_hooks(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.register_full_backward_hook(self._guided_backward_hook)

    def _guided_backward_hook(self, module, grad_input, grad_output):
        if grad_input[0] is None:
            return grad_input
        return (torch.clamp(grad_input[0], min=0.0),)

    def generate(self, x, target_class=None):
        x = x.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        out = self.model(x)

        if out.shape[1] == 1:
            logit = out.squeeze()
        else:
            assert target_class is not None
            logit = out[0, target_class]

        logit.backward()

        grad = x.grad[0].detach().cpu().numpy()  # [3,H,W]
        grad -= grad.min()
        grad /= (grad.max() + 1e-8)

        return grad


# ============================================================
# Utility: overlay heatmap
# ============================================================
def apply_colormap_on_image(img, cam, alpha=0.45):

    if cam.ndim == 3:
        cam = cam.squeeze()

    H, W, _ = img.shape
    cam = cv2.resize(cam, (W, H))

    cam = np.nan_to_num(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    cam_uint8 = np.uint8(cam * 255)
    #print(type(cam), cam.dtype, cam.shape)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    out = img * (1 - alpha) + heatmap * alpha
    return np.clip(out, 0, 1)


def overlay_on_image(img, mask, alpha=0.4):
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-8)

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    if mask.ndim == 2:
        mask = np.stack([mask]*3, axis=-1)

    out = img * (1 - alpha) + mask * alpha
    return np.clip(out, 0, 1)


# ============================================================
# Final visualization: Original / GradCAM / GBP / GGC
# ============================================================
def visualize_all(model, target_layer, img_tensor, img_np, target_class, save_path):

    gradcam = GradCAM(model, target_layer)
    guided_bp = GuidedBackprop(model)

    # ------------------------
    # GradCAM
    # ------------------------
    cam = gradcam(img_tensor, target_class)
    cam_on_img = apply_colormap_on_image(img_np, cam)

    # ------------------------
    # Guided Backprop
    # ------------------------
    gb = guided_bp.generate(img_tensor, target_class)   # [3,H,W]
    gb = gb.transpose(1,2,0)                            # (H,W,3)
    gb_overlay = overlay_on_image(img_np, gb)

    # ------------------------
    # Guided GradCAM
    # ------------------------
    H, W, _ = img_np.shape
    cam_resized = cv2.resize(cam, (W,H))
    guided_gradcam = gb * cam_resized[...,None]
    ggcam_overlay = overlay_on_image(img_np, guided_gradcam)

    def norm(x):
        x -= x.min()
        x /= (x.max() + 1e-8)
        return x

    guided_gradcam = norm(guided_gradcam)

    # ------------------------
    # Save figure
    # ------------------------
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))

    ax[0].imshow(img_np); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(cam_on_img); ax[1].set_title("GradCAM"); ax[1].axis("off")
    ax[2].imshow(gb); ax[2].set_title("Guided Backprop"); ax[2].axis("off")
    ax[3].imshow(ggcam_overlay); ax[3].set_title("Guided GradCAM"); ax[3].axis("off")

    #fig.suptitle(f"Model: {model.name} ", fontsize=18)
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()





if __name__ == '__main__':
    main()