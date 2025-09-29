import torch
import torch.nn.functional as F
import cv2
import numpy as np

# -------------------- 改进版 GradCAM --------------------
class GradCAM:
    def __init__(self, model, target_layers, use_plus=False, topk=None):
        """
        use_plus: 是否使用 Grad-CAM++
        topk: 使用前 k 个通道 (None=全部通道)
        """
        self.model = model
        self.model.eval()
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
        self.activations = [None] * len(self.target_layers)
        self.gradients = [None] * len(self.target_layers)
        self.use_plus = use_plus
        self.topk = topk
        self._register_hooks()

    def _register_hooks(self):
        for idx, layer in enumerate(self.target_layers):
            layer.register_forward_hook(lambda m, i, o, idx=idx: self._forward_hook(idx, o))
            layer.register_backward_hook(lambda m, gi, go, idx=idx: self._backward_hook(idx, go[0]))

    def _forward_hook(self, idx, output):
        self.activations[idx] = output

    def _backward_hook(self, idx, grad_output):
        self.gradients[idx] = grad_output

    def _compute_weights(self, grad, act):
        if not self.use_plus:  # 标准 Grad-CAM
            return grad.mean(dim=(2, 3), keepdim=True)
        else:  # Grad-CAM++
            grad2 = grad ** 2
            grad3 = grad2 * grad
            act = act.detach()
            alpha = grad2 / (2 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True) + 1e-5)
            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            return weights

    def generate_cam(self, target_score, mode="fusion", weights=None, output_size=(320, 320), method="max"):
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        cams = []
        for act, grad in zip(self.activations, self.gradients):
            if act is None or grad is None:
                raise ValueError("Forward/Backward hook not triggered.")

            weights_ch = self._compute_weights(grad, act)
            cam = (weights_ch * act).sum(dim=1, keepdim=True)

            if self.topk is not None:  # Top-k 通道
                abs_vals = torch.abs(weights_ch.squeeze())
                topk_idx = torch.topk(abs_vals, k=min(self.topk, abs_vals.numel()))[1]
                cam = (weights_ch[:, topk_idx, :, :] * act[:, topk_idx, :, :]).sum(dim=1, keepdim=True)

            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-5)
            cam_resized = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
            cams.append(cam_resized.detach().squeeze().cpu().numpy())  # [H,W]

        # --- 融合方式 ---
        if mode == "side_by_side":
            cams_resized = []
            for c in cams:
                c = c - c.min()
                c = c / (c.max() + 1e-5)
                c_color = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)
                c_color = cv2.cvtColor(c_color, cv2.COLOR_BGR2RGB)
                cams_resized.append(c_color)
            concat = np.concatenate(cams_resized, axis=1)
            return concat
        else:  # fusion
            if method == "avg":
                if weights is None:
                    weights = [1.0 / len(cams)] * len(cams)
                fused = np.zeros_like(cams[0], dtype=np.float32)
                for c, w in zip(cams, weights):
                    fused += w * c
            elif method == "max":
                fused = np.maximum.reduce(cams)
            elif method == "auto":
                scores = [c.mean() for c in cams]
                norm_scores = [s / sum(scores) for s in scores]
                fused = np.zeros_like(cams[0], dtype=np.float32)
                for c, w in zip(cams, norm_scores):
                    fused += w * c
            else:
                raise ValueError("method must be 'avg', 'max', or 'auto'")

            fused = np.clip(fused, 0, 1)
            fused_color = cv2.applyColorMap((fused * 255).astype(np.uint8), cv2.COLORMAP_JET)
            fused_color = cv2.cvtColor(fused_color, cv2.COLOR_BGR2RGB)
            return fused_color

    def overlay_gradcam_on_image(self, cam_img, img_resized, alpha=0.5, mode="fusion"):
        if mode == "fusion":
            overlay = (alpha * img_resized + (1 - alpha) * cam_img).astype(np.uint8)
        else:
            H, W, _ = cam_img.shape
            n = len(self.activations)
            img_single = cv2.resize(img_resized, (W // n, H))
            img_tile = np.concatenate([img_single] * n, axis=1)
            overlay = (alpha * img_tile + (1 - alpha) * cam_img).astype(np.uint8)
        return overlay

