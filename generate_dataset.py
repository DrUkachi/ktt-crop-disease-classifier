"""Synthetic-recipe dataset generator for T2.1.

Recipe (from the candidate brief, page 1):

  * Subset PlantVillage + Cassava Leaf Disease (public mirrors; ~300 imgs/class).
  * 5 classes: healthy, maize_rust, maize_blight, cassava_mosaic, bean_spot.
  * 80 / 10 / 10 train / val / test split.
  * Field-robustness variant: random Gaussian blur (sigma in [0, 1.5]),
    JPEG re-compression (q in [50, 85]), and brightness jitter.

Sources (three HF dataset mirrors, all public, all license-permissive):

  PlantVillage     -> "yusufberkay/plantvillage-dataset"  (maize classes)
  Cassava          -> "Dauka-CA/Cassava-Leaf-Disease-Detection-Train"
  Beans            -> "nateraw/beans"                     (angular leaf spot)

If you already have any of these unpacked locally, point the script at the
parent dir with `--source-dir <path>` and the script will skip the download.

Usage
-----

    # Auto-download from HF and build everything (default):
    python generate_dataset.py --out data/

    # Use local copies (expects ./raw/{plantvillage,cassava,beans}):
    python generate_dataset.py --source-dir ./raw --out data/

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
# Class -> source mapping. Each target class draws from one (source, subdir)
# pair. Subdirs match the canonical folder names in the upstream mirrors.
# ---------------------------------------------------------------------------

TARGET_CLASSES = ["bean_spot", "cassava_mosaic", "healthy", "maize_blight", "maize_rust"]
PER_CLASS = 300                       # ~300 per class per the brief
IMG_SIZE = 224
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 1337

# (hf_dataset_id, image_column, label_column, label_value, local_subdir_hint)
# `local_subdir_hint` is what we expect under --source-dir if downloads are
# skipped. The HF column names below match the named mirrors as of 2026-04.
SOURCE_MAP = {
    "healthy": dict(
        hf_id="yusufberkay/plantvillage-dataset",
        label_value="Corn_(maize)___healthy",
        local_hint=("plantvillage", "Corn_(maize)___healthy"),
    ),
    "maize_rust": dict(
        hf_id="yusufberkay/plantvillage-dataset",
        label_value="Corn_(maize)___Common_rust_",
        local_hint=("plantvillage", "Corn_(maize)___Common_rust_"),
    ),
    "maize_blight": dict(
        hf_id="yusufberkay/plantvillage-dataset",
        label_value="Corn_(maize)___Northern_Leaf_Blight",
        local_hint=("plantvillage", "Corn_(maize)___Northern_Leaf_Blight"),
    ),
    "cassava_mosaic": dict(
        hf_id="Dauka-CA/Cassava-Leaf-Disease-Detection-Train",
        label_value="Cassava Mosaic Disease (CMD)",
        local_hint=("cassava", "Cassava Mosaic Disease (CMD)"),
    ),
    "bean_spot": dict(
        hf_id="nateraw/beans",
        label_value="angular_leaf_spot",
        local_hint=("beans", "angular_leaf_spot"),
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
    return img.resize((size, size), Image.BILINEAR)


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
# Source loaders
# ---------------------------------------------------------------------------

@dataclass
class SourceItem:
    image: Image.Image
    src_id: str          # e.g., "plantvillage:Corn_(maize)___Common_rust_:0007.jpg"


def _load_local(source_dir: Path, target_class: str) -> Iterable[SourceItem]:
    cfg = SOURCE_MAP[target_class]
    parent, subdir = cfg["local_hint"]
    folder = source_dir / parent / subdir
    if not folder.exists():
        return []
    files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")) + sorted(folder.glob("*.JPG"))
    for fp in files:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            continue
        yield SourceItem(image=img, src_id=f"{parent}:{subdir}:{fp.name}")


def _load_hf(target_class: str, max_items: int) -> Iterable[SourceItem]:
    """Stream from a HuggingFace dataset mirror.

    Streaming avoids downloading the full archive (PlantVillage is >1 GB) when
    we only want ~300 images per class.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "datasets library is required for --hf-download mode.\n"
            "    pip install datasets"
        ) from e

    cfg = SOURCE_MAP[target_class]
    ds = load_dataset(cfg["hf_id"], split="train", streaming=True)

    label_value = cfg["label_value"]
    found = 0
    for i, row in enumerate(ds):
        # Different mirrors expose the label under different column names.
        label = (
            row.get("label")
            or row.get("labels")
            or row.get("class")
            or row.get("category")
        )
        # If the dataset's label is an int, it indexes into ds.features['label'].names
        # which we can't introspect in streaming mode without a peek. Mirrors above
        # expose string labels directly; if you swap a mirror, update this branch.
        if isinstance(label, str) and label != label_value:
            continue

        img_field = row.get("image") or row.get("img")
        if img_field is None:
            continue
        if isinstance(img_field, dict) and "bytes" in img_field:
            img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        else:
            img = img_field.convert("RGB")  # PIL.Image already

        yield SourceItem(image=img, src_id=f"{cfg['hf_id']}:{i}")
        found += 1
        if found >= max_items:
            return


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


def build_dataset(out_dir: Path, source_dir: Path | None, hf_download: bool) -> dict:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    manifest: dict = {"per_class": {}, "splits": {s: 0 for s in SPLITS}}

    for cls in TARGET_CLASSES:
        items: list[SourceItem] = []

        if source_dir is not None:
            items = list(_load_local(source_dir, cls))
            if items:
                print(f"[{cls}] loaded {len(items)} local images")

        if not items:
            if not hf_download:
                raise SystemExit(
                    f"No local images for class '{cls}' under {source_dir}.\n"
                    f"  Re-run with --hf-download, or place images in "
                    f"{source_dir}/{SOURCE_MAP[cls]['local_hint'][0]}/"
                    f"{SOURCE_MAP[cls]['local_hint'][1]}/"
                )
            print(f"[{cls}] streaming from HF: {SOURCE_MAP[cls]['hf_id']}")
            items = list(_load_hf(cls, max_items=PER_CLASS))

        if len(items) > PER_CLASS:
            rng.shuffle(items)
            items = items[:PER_CLASS]

        if not items:
            raise SystemExit(f"[{cls}] no images found from any source")

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
    p.add_argument("--source-dir", type=Path, default=None, help="local raw-data parent dir")
    p.add_argument("--hf-download", action="store_true", help="stream from HuggingFace mirrors")
    p.add_argument("--field-only", action="store_true", help="only rebuild test_field/ from existing test/")
    args = p.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.field_only:
        build_field_set(out_dir)
        return

    if args.source_dir is None and not args.hf_download:
        # Default: try HF download if no local source is provided.
        args.hf_download = True

    manifest = build_dataset(out_dir, args.source_dir, args.hf_download)
    n_field = build_field_set(out_dir)
    manifest["test_field"] = n_field

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest written -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
