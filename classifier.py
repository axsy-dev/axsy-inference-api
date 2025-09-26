import os
import json
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor


class Classifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 35,
        inp_size: int = 32,
        n_layers: int = 5,
        base_filters: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.sizes = (inp_size / torch.pow(2, torch.arange(6))).to(int)
        self.n_layers = n_layers
        self.base_filters = base_filters
        self.device = device

        self.register_buffer("T", torch.tensor(1))
        self.register_buffer("delta", torch.tensor(0.25))

        nfilters = [3] + [base_filters * n for n in [4, 4, 4, 6, 8]]
        self.nfilters = nfilters
        self.neurons = torch.square(self.sizes[self.n_layers]).item() * sum([nfilters[i] for i in range(n_layers + 1)])
        k_size = [3, 3, 3] + (n_layers - 3) * [3]
        b_norm = [0 for _ in range(n_layers - 2)] + [1, 1]

        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        sum(nfilters[: i + 1]),
                        int(nfilters[i + 1] / 2),
                        kernel_size=k_size[i],
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(int(nfilters[i + 1] / 2)) if b_norm[i] else nn.Identity(),
                    nn.GELU(approximate="none"),
                    nn.Conv2d(
                        int(nfilters[i + 1] / 2),
                        nfilters[i + 1],
                        kernel_size=k_size[i],
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(nfilters[i + 1]) if b_norm[i] else nn.Identity(),
                    nn.GELU(approximate="none"),
                )
                for i in range(n_layers)
            ]
        )

        self.downsample = nn.MaxPool2d(1, stride=2)

        self.dense = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.neurons, base_filters * 2),
                    nn.Sigmoid(),
                ),
                nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(base_filters * 2, n_classes),
                ),
            ]
        )

        # Explicit dim to avoid deprecation warning and ensure class-probability over classes
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        inp: torch.Tensor,
        extract_layer: int = 0,
        T: float = None,
        delta: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inp.size(dim=0)
        T = self.T.item() if T is None else T
        delta = self.delta.item() if delta is None else delta

        for i, layer in enumerate(self.conv):
            out = layer(inp)
            inp = self.downsample(torch.cat((out, inp), dim=1))

            if i == extract_layer:
                intermediate = torch.clone(out)

        out = inp.reshape(batch_size, self.neurons)
        i += 1

        for layer in self.dense:
            if i == extract_layer:
                intermediate = torch.clone(out)
            i += 1
            out = layer(out)
        probs = self.softmax(out / T)
        return probs, intermediate

    __call__ = forward


def _read_classifier_config() -> Optional[Dict[str, Any]]:
    """Read axsy-classifier.json if present and return dict, else None."""
    try:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "axsy-classifier.json"))
        if not os.path.isfile(config_path):
            return None
        with open(config_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def _extract_inp_size_from_config(cfg: Dict[str, Any]) -> Optional[int]:
    """Extract integer input size from config dict.

    Supports keys 'imp_size' (note the 'm'), 'inp_size' at the top level,
    or inside 'additional_metadata'. Values may be strings or integers.
    """
    def _to_int(v: Any) -> Optional[int]:
        try:
            iv = int(str(v).strip())
            return iv if iv > 0 else None
        except Exception:
            return None

    for key in ("imp_size", "inp_size"):
        if key in cfg:
            val = _to_int(cfg.get(key))
            if val is not None:
                return val

    meta = cfg.get("additional_metadata")
    if isinstance(meta, dict):
        for key in ("imp_size", "inp_size"):
            if key in meta:
                val = _to_int(meta.get(key))
                if val is not None:
                    return val
    return None


def _normalise_batch(patches: Tensor) -> Tensor:
    # Compute per-channel mean/std over batch and spatial dims
    mean = patches.mean(dim=(0, 2, 3), keepdim=True)
    std = patches.std(dim=(0, 2, 3), keepdim=True)
    std = torch.where(std < 1e-6, torch.full_like(std, 1e-6), std)
    return (patches - mean) / std


def load_classifier(
    weights_path: str,
    n_classes: Optional[int] = None,
    inp_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Classifier:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Prefer explicit configuration from JSON when available and caller didn't specify
    if inp_size is None:
        cfg = _read_classifier_config()
        if cfg is not None:
            cfg_inp = _extract_inp_size_from_config(cfg)
            if cfg_inp is not None:
                inp_size = cfg_inp

    # Infer architecture parameters when possible from checkpoint shapes
    inferred_classes = None
    inferred_neurons = None
    for k, v in state_dict.items():
        if k == "dense.1.1.weight" and hasattr(v, "shape") and len(v.shape) == 2:
            inferred_classes = int(v.shape[0])
        if k == "dense.0.1.weight" and hasattr(v, "shape") and len(v.shape) == 2:
            inferred_neurons = int(v.shape[1])
    # Defaults if not inferred
    base_filters = 64
    n_layers = 5
    sum_filters = 3 + base_filters * (4 + 4 + 4 + 6 + 8)  # 1667 for base_filters=64
    if inp_size is None and inferred_neurons is not None:
        # neurons = (inp_size/32)^2 * sum_filters
        from math import isclose, sqrt
        ratio = inferred_neurons / float(sum_filters)
        root = sqrt(max(1.0, ratio))
        # sizes[n_layers] = inp_size/32 should be a small integer (1,2,3,...)
        sizes_value = int(round(root))
        sizes_value = max(1, sizes_value)
        inp_size = 32 * sizes_value
    inp_size = inp_size or 32
    if n_classes is None and inferred_classes is not None:
        n_classes = inferred_classes
    n_classes = n_classes or 35

    model = Classifier(n_classes=n_classes, inp_size=inp_size, device=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


@torch.inference_mode()
def classify_image_tensor(
    model: Classifier,
    image_tensor_bchw: torch.Tensor,
) -> Dict[str, Any]:
    x = _normalise_batch(image_tensor_bchw)
    probs, _ = model(x)
    probs_np = probs.detach().cpu().float().numpy()
    top_idx = int(probs[0].argmax().item())
    top_prob = float(probs[0, top_idx].item())
    return {
        "top_index": top_idx,
        "top_prob": top_prob,
        "probs": probs_np[0].tolist(),
    }


