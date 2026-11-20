"""Merge the downloaded datasets into one YOLO dataset with our own classes.

    python -m atlas_training.prepare configs/ambulance_v1.yaml
    python -m atlas_training.prepare configs/ambulance_v1.yaml --inspect 20

Each source label is mapped with the config's class_map; a label that is not listed
stops the run. Splits: a source's own test split stays the test set (otherwise a share
of its images, chosen with the seed); the same for validation. Roboflow's augmented
copies of one image are always kept in the same split, and move_to_test can move a
share of a source's images with a target box into test. Images are hard-linked
into datasets/<name>/ where possible, so preparing does not double the disk use.
The output folder gets data.yaml (for Ultralytics) and manifest.json (counts, dropped
images and ignored boxes per source).

--inspect N draws the boxes of N random images per source into runs/inspect/ so that
the labels can be checked by eye before training (S4PT-52, step 2).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config as cfg

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ORIENTATION_MODEL = "yolo11n.pt"   # COCO-pretrained; only used to find upright copies
SOURCE_SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


@dataclass
class Sample:
    source: str
    image: Path
    labels: list[tuple[int, float, float, float, float]]   # target class, cx, cy, w, h
    origin_split: str
    group: str = ""   # Roboflow source image; augmented copies share it
    rotate: int = 0   # quarter turns counter-clockwise applied when writing (upright test copies)
    clip: str = ""    # frames that look like the same camera clip (see assign_clips)


def copy_key(s: Sample) -> str:
    """Copies of one image (Roboflow augmentation)."""
    return s.group or str(s.image)


def split_key(s: Sample) -> str:
    """The unit that is never divided between train, val and test."""
    return s.clip or copy_key(s)


def frame_features(path: Path, size: int = 32):
    """Grey thumbnail in its four 90-degree orientations, each normalised for correlation."""
    import cv2
    import numpy as np

    image = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (size, size), interpolation=cv2.INTER_AREA)
    views = []
    for k in range(4):
        v = np.rot90(image.astype(np.float32), k).ravel()
        v -= v.mean()
        views.append(v / (np.linalg.norm(v) + 1e-6))
    return views


def assign_clips(samples: list[Sample], threshold: float) -> int:
    """Put frames of the same video clip into one split unit.

    Many Roboflow images are consecutive frames of one video. If some went to train and
    others to test, the model would be tested on near copies of what it learned. Frames
    whose grey thumbnails correlate at least `threshold` in some 90-degree orientation
    (Roboflow also rotated copies) are linked, and linked frames form one clip. Returns
    the number of clips.
    """
    import numpy as np

    if not samples:
        return 0
    features = [frame_features(s.image) for s in samples]
    upright = np.stack([f[0] for f in features])
    similarity = np.max([upright @ np.stack([f[k] for f in features]).T for k in range(4)], axis=0)

    parent = list(range(len(samples)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(similarity, 1) >= threshold), strict=True):
        parent[find(int(i))] = find(int(j))
    for i, s in enumerate(samples):
        s.clip = f"{s.source}-clip{find(i)}"
    return len({s.clip for s in samples})


def group_key(stem: str, original_names: dict[str, str]) -> str:
    """Roboflow names copies '<original>.rf.<hash>'; copies of one image share <original>."""
    original = original_names.get(stem, stem)
    return original.split(".rf.")[0]


def read_original_names(source_dir: Path) -> dict[str, str]:
    """Short stem -> original stem, from the renamed_files.csv written by download."""
    path = source_dir / "renamed_files.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {Path(row["short_name"]).stem: Path(row["original_name"]).stem for row in csv.DictReader(f)}


@dataclass
class SourceReport:
    images: Counter = field(default_factory=Counter)        # per output split
    boxes: Counter = field(default_factory=Counter)         # per target class
    dropped_images: Counter = field(default_factory=Counter)  # per label that caused it
    ignored_boxes: Counter = field(default_factory=Counter)   # per source label
    upright: dict = field(default_factory=dict)               # see upright_test_copies
    clips: int | None = None                                  # see assign_clips
    near_copies_dropped: dict = field(default_factory=dict)   # see drop_test_near_copies
    excluded: int = 0                                         # see DatasetSource.exclude_images
    background_dropped: int = 0                               # see cap_background


def read_names(source_dir: Path) -> list[str]:
    data = yaml.safe_load((source_dir / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def parse_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    """YOLO box (cls cx cy w h) or polygon (cls x1 y1 x2 y2 ...), as a box."""
    parts = line.split()
    if not parts:
        return None
    cls, values = int(parts[0]), [float(v) for v in parts[1:]]
    if len(values) == 4:
        return (cls, *values)
    if len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return (cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    raise ValueError(f"unexpected label line: {line!r}")


def map_labels(
    raw: list[tuple[int, float, float, float, float]],
    names: list[str],
    class_map: dict[str, str],
    targets: list[str],
    report: SourceReport,
) -> list[tuple[int, float, float, float, float]] | None:
    """Apply class_map to one image. None means the image is dropped."""
    for cls, *_ in raw:
        if class_map[names[cls]] == cfg.DROP_IMAGE:
            report.dropped_images[names[cls]] += 1
            return None
    kept = []
    for cls, *box in raw:
        label = names[cls]
        action = class_map[label]
        if action == cfg.IGNORE:
            report.ignored_boxes[label] += 1
            continue
        kept.append((targets.index(action), *box))
    return kept


def collect(source: cfg.DatasetSource, targets: list[str], report: SourceReport) -> list[Sample]:
    root = source.download_dir
    if not (root / "data.yaml").exists():
        sys.exit(f"{source.id}: not downloaded yet ({root}); run atlas_training.download first.")
    names = read_names(root)
    unknown = sorted(set(names) - set(source.class_map))
    if unknown:
        sys.exit(f"{source.id}: labels {unknown} are not in class_map; check them and add them to the config.")

    # Copies are recognised by their shared original name, which is only meaningful for
    # an export with Roboflow augmentation; elsewhere different uploads can share a
    # generic name like "0053.jpg".
    original_names = read_original_names(root)
    samples = []
    seen = set()
    for split_dir, split in SOURCE_SPLITS.items():
        image_dir = root / split_dir / "images"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if image.stem in source.exclude_images:
                seen.add(image.stem)
                report.excluded += 1
                continue
            label_file = root / split_dir / "labels" / f"{image.stem}.txt"
            lines = label_file.read_text(encoding="utf-8").splitlines() if label_file.exists() else []
            raw = [b for b in (parse_label_line(line) for line in lines) if b is not None]
            labels = map_labels(raw, names, source.class_map, targets, report)
            if labels is not None:
                group = group_key(image.stem, original_names) if source.augmented_copies else ""
                samples.append(Sample(source.id, image, labels, split, group))
    missing = sorted(set(source.exclude_images) - seen)
    if missing:
        sys.exit(f"{source.id}: exclude_images {missing} not found; check the file stems in the config.")
    return samples


def by_key(samples: list[Sample], key=split_key) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(key(s), []).append(s)
    return groups


def assign_splits(
    samples: list[Sample], test_fraction: float, val_fraction: float, seed: int, move_to_test: float = 0.0,
) -> dict[str, list[Sample]]:
    """Split one source without dividing a unit (a clip, or the copies of one image).

    A unit goes to test if any of its frames is in the source's own test split, else to
    val if any frame is in its own val split, else it stays in the pool. A missing test
    or val split is filled with a seeded share of whole units from the pool. Then
    move_to_test moves that share of the remaining units that contain a target box into
    test (for sources whose test split has none).
    """
    has = {s.origin_split for s in samples}
    rng = random.Random(seed)
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    pool = []
    for key, items in sorted(by_key(samples).items()):
        origins = {s.origin_split for s in items}
        target = next((split for split in ("test", "val") if split in has and split in origins), None)
        if target:
            out[target].extend(items)
        else:
            pool.append((key, items))
    for split in ("test", "val"):
        if split in has:
            continue
        rng.shuffle(pool)
        wanted = round(len(samples) * (test_fraction if split == "test" else val_fraction))
        taken = 0
        while pool and taken < wanted:
            _, items = pool.pop()
            out[split].extend(items)
            taken += len(items)
    out["train"] = [s for _, items in pool for s in items]

    if move_to_test:
        candidates = sorted(
            (key, items) for key, items in by_key(out["train"] + out["val"]).items()
            if any(s.labels for s in items)
        )
        rng.shuffle(candidates)
        moved = {key for key, _ in candidates[:round(len(candidates) * move_to_test)]}
        for split in ("train", "val"):
            out["test"] += [s for s in out[split] if split_key(s) in moved]
            out[split] = [s for s in out[split] if split_key(s) not in moved]
    return out


def rotate_label(box: tuple[int, float, float, float, float], quarter_turns: int) -> tuple[int, float, float, float, float]:
    """The YOLO box after numpy.rot90(image, quarter_turns) (counter-clockwise)."""
    cls, cx, cy, w, h = box
    for _ in range(quarter_turns % 4):
        cx, cy, w, h = cy, 1 - cx, h, w
    return (cls, cx, cy, w, h)


# COCO classes that are vehicles: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]


def upright_test_copies(
    test: list[Sample], weights: Path, overrides: dict[str, int] | None = None,
) -> tuple[list[Sample], dict]:
    """Keep one upright copy of every augmented image that was moved into test.

    Roboflow rotated train copies by random multiples of 90 degrees and does not record
    which. For each copy and each of the four orientations, a COCO-pretrained detector
    (it knows cars and trucks, not ambulances, so it does not favour the model under
    test) sums the confidence of the vehicles it finds; the copy and orientation with
    the highest sum is kept and written rotated back. Where the detector is wrong (dark
    night scenes), `overrides` names the copy (file stem) and its quarter turns, checked
    by eye. Images from the source's own test split are not touched.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    overrides = overrides or {}
    moved = by_key([s for s in test if s.origin_split != "test"], copy_key)
    kept = [s for s in test if s.origin_split == "test"]
    if not moved:
        return test, {}
    model = YOLO(str(weights))
    stats = {"images": len(moved), "copies_removed": 0, "rotated_back": 0, "set_by_hand": 0,
             "aligned_to_clip": 0, "chosen": {}}
    decisions: dict[str, tuple[Sample, int]] = {}
    for key, copies_ in moved.items():
        fixed = [s for s in copies_ if s.image.stem in overrides]
        if fixed:
            decisions[key] = (fixed[0], overrides[fixed[0].image.stem])
            stats["set_by_hand"] += 1

    # Frames of one clip share their orientation. In each clip the frame the detector
    # is most sure about (or one set by hand) is the reference; the other frames take
    # the copy and orientation that look most like it.
    clips = by_key([c[0] for c in moved.values()], lambda s: s.clip or copy_key(s))
    for clip_frames in clips.values():
        keys = [copy_key(s) for s in clip_frames]
        reference = None
        for key in keys:
            if key in decisions:
                reference = decisions[key]
                break
        if reference is None:
            scored = [(best_orientation(moved[key], model, cv2, np, with_score=True), key) for key in keys]
            (_score, s, k), key = max(scored, key=lambda item: item[0][0])
            reference = decisions[key] = (s, k)
        undecided = [key for key in keys if key not in decisions]
        if not undecided:
            continue
        ref_sample, ref_k = reference
        ref_vec = frame_features(ref_sample.image)[ref_k]
        for key in undecided:
            decisions[key] = max(
                ((s, k) for s in moved[key] for k in range(4)),
                key=lambda sk: float(frame_features(sk[0].image)[sk[1]] @ ref_vec),
            )
            stats["aligned_to_clip"] += 1

    for key, copies_ in sorted(moved.items()):
        chosen, k = decisions[key]
        chosen.rotate = k
        chosen.labels = [rotate_label(b, k) for b in chosen.labels]
        kept.append(chosen)
        stats["copies_removed"] += len(copies_) - 1
        stats["rotated_back"] += k != 0
        stats["chosen"][chosen.image.stem] = k
    return kept, stats


def best_orientation(copies_: list[Sample], model, cv2, np, with_score: bool = False):
    """The copy and quarter turns in which the detector sees the most vehicles."""
    best = (-1.0, copies_[0], 0)
    for s in copies_:
        image = cv2.imread(str(s.image))
        views = [np.ascontiguousarray(np.rot90(image, k)) for k in range(4)]
        results = model.predict(views, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)
        for k, r in enumerate(results):
            score = float(r.boxes.conf.sum()) if r.boxes is not None else 0.0
            if score > best[0]:
                best = (score, s, k)
    return best if with_score else (best[1], best[2])


def drop_test_near_copies(
    splits_by_source: dict[str, dict[str, list[Sample]]], threshold: float, reports: dict[str, SourceReport],
) -> None:
    """Remove test images that are (near) copies of a train or val image of any source.

    Some sources contain the same photo, or consecutive frames of one video, in more
    than one of their own splits; a model would then be tested on images it has
    already learned. A test image whose grey thumbnail correlates at least `threshold`
    with a train or val image in any 90-degree orientation is removed from test.
    Train and val are not changed.
    """
    import numpy as np

    learned = [s for splits in splits_by_source.values() for split in ("train", "val") for s in splits[split]]
    if not learned:
        return
    learned_views = [frame_features(s.image) for s in learned]
    by_orientation = [np.stack([v[k] for v in learned_views]) for k in range(4)]
    for source_id, splits in splits_by_source.items():
        kept = []
        dropped = {"images": 0, "with_target": 0}
        for s in splits["test"]:
            view = frame_features(s.image)[s.rotate % 4]
            if max(float((m @ view).max()) for m in by_orientation) >= threshold:
                dropped["images"] += 1
                dropped["with_target"] += bool(s.labels)
            else:
                kept.append(s)
        splits["test"] = kept
        reports[source_id].near_copies_dropped = dropped


def cap_background(samples: list[Sample], limit: int, seed: int) -> tuple[list[Sample], int]:
    """Keep every sample with a target box and at most `limit` without one (chosen with the seed).

    Large sets of images without an ambulance take most of the training time but teach
    little more than a sample of them; the order of the input is kept.
    """
    background = [i for i, s in enumerate(samples) if not s.labels]
    if len(background) <= limit:
        return list(samples), 0
    keep = set(random.Random(seed).sample(background, limit))
    kept = [s for i, s in enumerate(samples) if s.labels or i in keep]
    return kept, len(samples) - len(kept)


def write_image(s: Sample, dst: Path) -> None:
    if not s.rotate:
        link_or_copy(s.image, dst)
        return
    import cv2
    import numpy as np

    cv2.imwrite(str(dst), np.rot90(cv2.imread(str(s.image)), s.rotate))


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_dataset(exp: cfg.Experiment, splits_by_source: dict[str, dict[str, list[Sample]]]) -> None:
    repeats = {s.id: s.train_repeat for s in exp.datasets}
    out = exp.dataset_dir
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
    for source_id, splits in splits_by_source.items():
        for split, samples in splits.items():
            # Oversampling: train images of a small source are written several times, so
            # that each epoch shows them as often (hardlinks, no extra disk space).
            copies = repeats.get(source_id, 1) if split == "train" else 1
            for s in samples:
                lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in s.labels]
                for r in range(copies):
                    name = f"{s.source}__{s.image.stem}" + (f"__r{r}" if r else "")
                    write_image(s, out / split / "images" / f"{name}{s.image.suffix.lower()}")
                    (out / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    data = {
        "path": str(out),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": dict(enumerate(exp.target_classes)),
    }
    exp.data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))


def config_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def prepare(config_path: str) -> dict:
    exp = cfg.load(config_path)
    splits_by_source, reports = {}, {}
    for source in exp.datasets:
        report = SourceReport()
        samples = collect(source, exp.target_classes, report)
        if source.clip_similarity:
            report.clips = assign_clips(samples, source.clip_similarity)
        splits = assign_splits(samples, exp.test_fraction, exp.val_fraction, exp.seed, source.move_to_test)
        if source.upright_test_copies:
            splits["test"], report.upright = upright_test_copies(
                splits["test"], cfg.WEIGHTS_DIR / ORIENTATION_MODEL, source.orientation_overrides)
        splits_by_source[source.id], reports[source.id] = splits, report

    if exp.test_near_copy_similarity:
        drop_test_near_copies(splits_by_source, exp.test_near_copy_similarity, reports)
    for source in exp.datasets:
        if source.max_train_background is not None:
            splits = splits_by_source[source.id]
            splits["train"], reports[source.id].background_dropped = cap_background(
                splits["train"], source.max_train_background, exp.seed)
    for source in exp.datasets:
        report = reports[source.id]
        for split, items in splits_by_source[source.id].items():
            report.images[split] = len(items)
            for s in items:
                report.boxes.update(exp.target_classes[c] for c, *_ in s.labels)
    write_dataset(exp, splits_by_source)

    manifest = {
        "experiment": exp.name,
        "config_sha256": config_hash(config_path),
        "seed": exp.seed,
        "sources": {
            s.id: {
                "version": s.version,
                "move_to_test": s.move_to_test,
                "licence": s.licence,
                "url": s.url,
                "images": dict(reports[s.id].images),
                "boxes": dict(reports[s.id].boxes),
                "dropped_images": dict(reports[s.id].dropped_images),
                "ignored_boxes": dict(reports[s.id].ignored_boxes),
                "clip_similarity": s.clip_similarity,
                "clips": reports[s.id].clips,
                "test_clips": sorted({x.clip for x in splits_by_source[s.id]["test"] if x.clip}),
                "upright_test_copies": reports[s.id].upright,
                "test_near_copies_dropped": reports[s.id].near_copies_dropped,
                "excluded_images": reports[s.id].excluded,
                "train_background_dropped": reports[s.id].background_dropped,
                "train_repeat": s.train_repeat,
            }
            for s in exp.datasets
        },
        "test_near_copy_similarity": exp.test_near_copy_similarity,
    }
    (exp.dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def inspect(config_path: str, per_source: int) -> Path:
    """Draw the original and mapped labels of random images for a manual check."""
    import cv2

    exp = cfg.load(config_path)
    rng = random.Random(exp.seed)
    out = cfg.RUNS_DIR / "inspect" / exp.name
    out.mkdir(parents=True, exist_ok=True)
    for source in exp.datasets:
        names = read_names(source.download_dir)
        images = sorted(p for p in source.download_dir.glob("*/images/*") if p.suffix.lower() in IMAGE_SUFFIXES)
        for image in rng.sample(images, min(per_source, len(images))):
            label_file = image.parent.parent / "labels" / f"{image.stem}.txt"
            frame = cv2.imread(str(image))
            h, w = frame.shape[:2]
            for line in label_file.read_text().splitlines() if label_file.exists() else []:
                box = parse_label_line(line)
                if box is None:
                    continue
                cls, cx, cy, bw, bh = box
                label = names[cls]
                action = source.class_map.get(label, "?")
                colour = {"ambulance": (0, 200, 0), cfg.IGNORE: (160, 160, 160)}.get(action, (0, 0, 255))
                p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
                p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
                cv2.rectangle(frame, p1, p2, colour, 2)
                cv2.putText(frame, f"{label} -> {action}", (p1[0], max(p1[1] - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            cv2.imwrite(str(out / f"{source.id}__{image.name}"), frame)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--inspect", type=int, metavar="N", help="only draw N labelled images per source")
    args = parser.parse_args()
    if args.inspect:
        print(f"Images written to {inspect(args.config, args.inspect)}")
        return
    print(json.dumps(prepare(args.config), indent=2))


if __name__ == "__main__":
    main()
