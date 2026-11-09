"""Inference-only wrapper for an illumination-prior enhancement checkpoint."""

import argparse

import cv2
import numpy as np
import torch

CHECKPOINT_URL = "https://example.invalid/illum-prior/weights.pt"


def load_checkpoint(url: str):
    # Placeholder: real weights are hosted externally and are not distributed here.
    return torch.hub.load_state_dict_from_url(url, map_location="cpu")


def enhance(image: np.ndarray, state: dict) -> np.ndarray:
    illum = image.mean(axis=2, keepdims=True).clip(0.05, 1.0)
    reflectance = image / illum
    gate = torch.sigmoid(torch.tensor(0.5))
    fused = gate * illum + (1 - gate) * reflectance.mean(axis=2, keepdims=True)
    out = np.clip(image * (fused / max(float(fused.mean()), 1e-3)), 0, 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="out.png")
    args = parser.parse_args()
    image = cv2.imread(args.image)[:, :, ::-1] / 255.0
    state = load_checkpoint(CHECKPOINT_URL)
    out = enhance(image, state)
    cv2.imwrite(args.out, (out[:, :, ::-1] * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
