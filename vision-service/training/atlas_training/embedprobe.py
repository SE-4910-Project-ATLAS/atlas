"""Linear probe on frozen foundation-model features of the classifier crops (experiment).

    python -m atlas_training.embedprobe datasets/crops_v5g --model vit_small_patch14_dinov2.lvd142m \
        --out runs/probe/dinov2_s_v5g [--device cpu]

Fine-grained work reports that frozen DINOv2 features with a linear classifier are strong
with little data (Oquab et al., DINOv2, arXiv 2304.07193; frozen-feature benchmarks), and
ambulance vs white van is a fine-grained question. This extracts one embedding per crop
(train and val of a crop set, the same crops the YOLO classifier is trained and checked
on), fits a logistic regression on train and reports val accuracy, per-class recall and
the val errors, so it can be compared with the YOLO classifier on the same crops. The
features are cached (<out>/features.npz) and the probe is saved (<out>/probe.joblib) for
a later two-model test on the development videos. Test sets are not touched here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ("ambulance", "other")


def files_of(root: Path, split: str) -> tuple[list[Path], np.ndarray]:
    files, labels = [], []
    for k, c in enumerate(CLASSES):
        fs = sorted((root / split / c).glob("*.jpg"))
        files += fs
        labels += [k] * len(fs)
    return files, np.array(labels)


def embed(files: list[Path], model, transform, device: str, batch: int = 64) -> np.ndarray:
    import torch
    from PIL import Image

    out = []
    with torch.inference_mode():
        for i in range(0, len(files), batch):
            x = torch.stack([transform(Image.open(f).convert("RGB")) for f in files[i:i + batch]]).to(device)
            out.append(model(x).float().cpu().numpy())
            if i // batch % 20 == 0:
                print(f"  {i + len(x)}/{len(files)}", flush=True)
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("crops", type=Path, help="crop set with train/ and val/ (ambulance, other)")
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m", help="timm model name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--C", type=float, default=1.0, help="inverse regularisation of the logistic regression")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import joblib
    import timm
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "features.npz"
    train_f, train_y = files_of(args.crops, "train")
    val_f, val_y = files_of(args.crops, "val")
    if cache.exists():
        d = np.load(cache)
        xtr, xva = d["train"], d["val"]
    else:
        torch.set_num_threads(max(1, torch.get_num_threads()))
        model = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(args.device)
        cfg = timm.data.resolve_data_config({}, model=model)
        cfg["input_size"] = (3, 224, 224)
        transform = timm.data.create_transform(**cfg)
        print(f"train {len(train_f)} crops", flush=True)
        xtr = embed(train_f, model, transform, args.device)
        print(f"val {len(val_f)} crops", flush=True)
        xva = embed(val_f, model, transform, args.device)
        np.savez_compressed(cache, train=xtr, val=xva)

    scaler = StandardScaler().fit(xtr)
    probe = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced")
    probe.fit(scaler.transform(xtr), train_y)
    p_amb = probe.predict_proba(scaler.transform(xva))[:, 0]
    pred = np.where(p_amb >= 0.5, 0, 1)
    acc = float((pred == val_y).mean())
    rec = {c: float((pred[val_y == k] == k).mean()) for k, c in enumerate(CLASSES)}
    errors = [{"file": str(f), "label": CLASSES[y], "p_ambulance": round(float(p), 4)}
              for f, y, p, q in zip(val_f, val_y, p_amb, pred, strict=True) if q != y]
    result = {"model": args.model, "C": args.C, "train": len(train_f), "val": len(val_f), "val_accuracy": acc,
              "val_recall": rec, "errors": errors}
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    joblib.dump({"scaler": scaler, "probe": probe, "model": args.model}, args.out / "probe.joblib")
    print(f"val accuracy {acc:.4f}, recall {rec}, errors {len(errors)}")


if __name__ == "__main__":
    main()
