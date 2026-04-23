"""Grad-CAM for the fine-tuned MobileNetV3-Small classifier.

Two consumers:

* `notebooks/01_train_eval.ipynb` — renders overlays for the evaluation report.
* `service/app.py` (Phase 5) — compresses the heatmap into a short natural
  language `rationale` string ("lesion activity concentrated upper-left") so
  the JSON response carries model-derived evidence, not a lookup table.

We target `model.features[-1]` (the final conv stage, pre-classifier). This is
the standard Grad-CAM choice for MobileNetV3: it's the last layer where spatial
structure survives before the global pool collapses everything to a vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GradCAMResult:
    """Output of a single Grad-CAM forward/backward pass."""
    heatmap: np.ndarray       # (H, W) float32 in [0, 1], upsampled to input res
    pred_class: int
    pred_confidence: float


class GradCAM:
    """Grad-CAM for a classifier with a `.features` block (MobileNet / EffNet style).

    Not thread-safe — hooks mutate instance state. Build one per request or per
    notebook cell and drop it afterward.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.model.eval()
        # Default target: the last sub-module of `features` (true for MobileNetV3).
        # Passing a specific layer is useful if we ever swap backbones.
        if target_layer is None:
            target_layer = model.features[-1]  # type: ignore[index]
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        # `full_backward_hook` fires after the backward pass on this submodule.
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out: tuple[torch.Tensor, ...]) -> None:
        self._gradients = grad_out[0].detach()

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def compute(self, x: torch.Tensor, class_idx: int | None = None) -> GradCAMResult:
        """Run forward + backward and return the heatmap at input resolution.

        `x` is (1, 3, H, W) on the same device as the model. If `class_idx` is
        None we use the top-1 prediction (the defensible default for post-hoc
        rationale on an API response — explain what you just returned).
        """
        x = x.requires_grad_(True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        if class_idx is None:
            class_idx = int(torch.argmax(probs).item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        assert self._activations is not None and self._gradients is not None, (
            "hooks did not fire — check target layer"
        )
        # Grad-CAM weights: global-average the gradient over spatial dims.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)                 # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)             # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return GradCAMResult(
            heatmap=cam.astype(np.float32),
            pred_class=class_idx,
            pred_confidence=float(probs[class_idx].item()),
        )


def heatmap_summary(heatmap: np.ndarray) -> dict:
    """Reduce a 224x224 Grad-CAM map to cues the API / SMS copy can use.

    We emit three numbers:
      * `peak_fraction` — how concentrated the attention is (1.0 = one pixel,
        0.0 = uniform). High peak → confident point-defect rationale.
      * `quadrant`      — "upper-left" | "upper-right" | "lower-left" |
                          "lower-right" | "centre". Useful as a human hint in
                          the rationale string ("lesion activity concentrated
                          upper-left").
      * `area_gt50`     — fraction of pixels above 0.5 intensity. High value
                          → diffuse / symptomatic-everywhere rationale.
    """
    h, w = heatmap.shape
    total = heatmap.sum()
    peak = float(heatmap.max())
    area_gt50 = float((heatmap > 0.5).mean())

    # Centre of mass (weighted). Only meaningful if the heatmap actually fires.
    if total < 1e-6:
        cy, cx = h / 2, w / 2
    else:
        ys, xs = np.mgrid[0:h, 0:w]
        cy = float((ys * heatmap).sum() / total)
        cx = float((xs * heatmap).sum() / total)

    # Quadrant vs a central band (middle 1/3 of each axis = "centre").
    third = 1 / 3
    vert = "upper" if cy < h * third else ("lower" if cy > h * (1 - third) else "middle")
    horiz = "left" if cx < w * third else ("right" if cx > w * (1 - third) else "centre")
    if vert == "middle" and horiz == "centre":
        quadrant = "centre"
    elif vert == "middle":
        quadrant = horiz
    elif horiz == "centre":
        quadrant = vert
    else:
        quadrant = f"{vert}-{horiz}"

    return {
        "peak_fraction": round(peak, 3),
        "area_gt50": round(area_gt50, 3),
        "quadrant": quadrant,
        "centre_x": round(cx / w, 3),
        "centre_y": round(cy / h, 3),
    }
