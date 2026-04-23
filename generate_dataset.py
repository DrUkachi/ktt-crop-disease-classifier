"""Synthetic-recipe dataset generator for T2.1.

Recipe (from the candidate brief, page 1):

  * Subset PlantVillage + Cassava Leaf Disease (public mirrors; ~300 imgs/class).
  * 5 classes: healthy, maize_rust, maize_blight, cassava_mosaic, bean_spot.
  * 80 / 10 / 10 train / val / test split.
  * Field-robustness variant: random Gaussian blur (sigma in [0, 1.5]),
    JPEG re-compression (q in [50, 85]), and brightness jitter.

Sources (three HF dataset mirrors, all public, all license-permissive as of
2026-04). These were chosen after the scaffold's originally-picked mirrors
went dead; the swap is documented in process_log.md.

  PlantVillage -> "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
                  (maize classes at ClassLabel indices 8, 9, 10)
  Cassava      -> "dpdl-benchmark/cassava"
                  (CMD = label 3; labels are raw int64, no ClassLabel names)
  Beans        -> "AI-Lab-Makerere/beans"
                  (angular_leaf_spot = ClassLabel index 0)

Usage
-----

    # Default: auto-download from HF and build everything:
    python generate_dataset.py --out data/

    # Use a local HF cache (same as default on re-runs):
    python generate_dataset.py --out data/

    # Only rebuild the field-noisy test set:
    python generate_dataset.py --out data/ --field-only

Output layout
-------------

    data/
      train/<class>/*.jpg
      val/<class>/*.jpg
      test/<class>/*.jpg
      test_field/<class>/*.jpg     # noisy variant of test/
      manifest.json                 # counts + per-image source provenance
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Class -> source mapping. Each target class is drawn from one HF dataset
# mirror, one split, and one integer label index.
# ---------------------------------------------------------------------------

TARGET_CLASSES = ["bean_spot", "cassava_mosaic", "healthy", "maize_blight", "maize_rust"]
PER_CLASS = 300
IMG_SIZE = 224
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 1337

SOURCE_MAP = {
    # Maize classes: PlantVillage mirror. 38-class ClassLabel.
    "healthy": dict(
        hf_id="BrandonFors/Plant-Diseases-PlantVillage-Dataset",
        split="train",
        label_col="label",
        label_idx=10,                      # Corn_(maize)___healthy
        label_name="Corn_(maize)___healthy",
    ),
    "maize_rust": dict(
        hf_id="BrandonFors/Plant-Diseases-PlantVillage-Dataset",
        split="train",
        label_col="label",
        label_idx=8,                       # Corn_(maize)___Common_rust_
        label_name="Corn_(maize)___Common_rust_",
    ),
    "maize_blight": dict(
        hf_id="BrandonFors/Plant-Diseases-PlantVillage-Dataset",
        split="train",
        label_col="label",
        label_idx=9,                       # Corn_(maize)___Northern_Leaf_Blight
        label_name="Corn_(maize)___Northern_Leaf_Blight",
    ),
    # Cassava mosaic disease: dpdl-benchmark mirror. Raw int labels, not ClassLabel.
    # Label 3 corresponds to "Cassava Mosaic Disease" (confirmed by filename
    # pattern train-cmd-*.jpg on the mirror).
    "cassava_mosaic": dict(
        hf_id="dpdl-benchmark/cassava",
        split="train",
        label_col="label",
        label_idx=3,
        label_name="Cassava Mosaic Disease (CMD)",
    ),
    # Bean angular leaf spot: Makerere iBeans mirror. ClassLabel with 3 names.
    "bean_spot": dict(
        hf_id="AI-Lab-Makerere/beans",
        split="train",
        label_col="labels",
        label_idx=0,
        label_name="angular_leaf_spot",
    ),
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_square(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Centre-crop to square then resize. Keeps aspect ratio sane for leaves."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.Resampling.BILINEAR)


def _apply_field_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    """Brief recipe: blur sigma in [0, 1.5], JPEG q in [50, 85], brightness jitter."""
    sigma = rng.uniform(0.0, 1.5)
    if sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    brightness = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    q = rng.randint(50, 85)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Source loader (HF)
# ---------------------------------------------------------------------------

@dataclass
class SourceItem:
    image: Image.Image
    src_id: str


def _load_hf(target_class: str, max_items: int) -> list[SourceItem]:
    """Stream from a HuggingFace dataset mirror and keep rows matching label_idx.

    We use streaming so fresh Colab runs don't have to materialise the entire
    PlantVillage archive (~850 MB) before we start selecting. On re-runs the HF
    cache short-circuits the stream.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "datasets library is required.\n    pip install datasets"
        ) from e

    cfg = SOURCE_MAP[target_class]
    ds = load_dataset(cfg["hf_id"], split=cfg["split"], streaming=True)

    out: list[SourceItem] = []
    label_col = cfg["label_col"]
    want_idx = cfg["label_idx"]

    for i, row in enumerate(ds):
        raw = row.get(label_col)
        if raw is None:
            continue
        # `raw` can be int (ClassLabel / raw int64) or str (if a mirror stores
        # label as a string). Compare both ways.
        if isinstance(raw, str):
            if raw != cfg["label_name"]:
                continue
        else:
            if int(raw) != want_idx:
                continue

        img_field = row.get("image") or row.get("img")
        if img_field is None:
            continue
        if isinstance(img_field, dict):
            if "bytes" not in img_field:
                continue
            img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif isinstance(img_field, Image.Image):
            img = img_field.convert("RGB")
        else:
            continue

        out.append(SourceItem(image=img, src_id=f"{cfg['hf_id']}:{cfg['split']}:{i}"))
        if len(out) >= max_items:
            return out

    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _split_indices(n: int, rng: random.Random) -> dict[str, list[int]]:
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(round(n * SPLITS["train"]))
    n_val = int(round(n * SPLITS["val"]))
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def build_dataset(out_dir: Path) -> dict:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    manifest: dict = {"per_class": {}, "splits": {s: 0 for s in SPLITS}, "source_map": {}}
    for cls, cfg in SOURCE_MAP.items():
        manifest["source_map"][cls] = {
            "hf_id": cfg["hf_id"],
            "label_name": cfg["label_name"],
            "label_idx": cfg["label_idx"],
        }

    for cls in TARGET_CLASSES:
        print(f"[{cls}] streaming from HF: {SOURCE_MAP[cls]['hf_id']} (label={SOURCE_MAP[cls]['label_name']})")
        items = _load_hf(cls, max_items=PER_CLASS)

        if not items:
            raise SystemExit(f"[{cls}] no images found — check HF mirror availability")

        rng.shuffle(items)
        items = items[:PER_CLASS]

        splits = _split_indices(len(items), rng)
        manifest["per_class"][cls] = {
            "n_total": len(items),
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "sources": [items[i].src_id for i in range(len(items))],
        }

        for split, indices in splits.items():
            split_dir = out_dir / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for j, idx in enumerate(indices):
                img = _resize_square(items[idx].image)
                img.save(split_dir / f"{cls}_{split}_{j:04d}.jpg", quality=92)
            manifest["splits"][split] += len(indices)

        print(
            f"[{cls}] -> train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
        )

    return manifest


def build_field_set(out_dir: Path) -> int:
    """Recreate test/ as test_field/ with the brief's noise recipe."""
    rng = random.Random(SEED + 1)
    test_dir = out_dir / "test"
    field_dir = out_dir / "test_field"
    if not test_dir.exists():
        raise SystemExit("test/ does not exist yet — build the clean set first.")
    if field_dir.exists():
        shutil.rmtree(field_dir)

    n = 0
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        out_cls = field_dir / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)
        for fp in sorted(cls_dir.glob("*.jpg")):
            img = Image.open(fp).convert("RGB")
            noisy = _apply_field_noise(img, rng)
            noisy.save(out_cls / fp.name, quality=88)
            n += 1
    print(f"test_field/ built from test/: {n} images")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data"), help="output dataset root")
    p.add_argument("--field-only", action="store_true", help="only rebuild test_field/ from existing test/")
    args = p.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.field_only:
        build_field_set(out_dir)
        return

    manifest = build_dataset(out_dir)
    n_field = build_field_set(out_dir)
    manifest["test_field"] = n_field

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest written -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
