"""Load frames from atlas_training.frames into a local Label Studio project.

    python -m atlas_training.labelstudio import D:/code/atlas-data/labeling/cctv_v1 \
        --project "ATLAS ambulance - cctv_v1" --token-file D:/code/atlas-data/labeling/.ls_token

Label Studio runs on this computer (see README.md > Labelling) and serves the images from
its local-files root, so frames never leave the machine. The model's boxes are imported as
predictions: they are shown as suggestions and become labels only when a person accepts
or corrects them. Export the finished project in YOLO format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="ambulance" background="#e53935"/>
    <Label value="police" background="#1e88e5"/>
    <Label value="fire_truck" background="#fb8c00"/>
  </RectangleLabels>
</View>"""

INSTRUCTIONS = (
    "Draw a box around every ambulance, police car and fire truck, also small, far, partly "
    "hidden or night-time ones. Label vehicles only: the word AMBULANS in a news caption or "
    "on a sign is not an ambulance. The box covers all of the vehicle that can be seen, not "
    "only the lights or the lettering; label a partly hidden or cut-off ambulance when the "
    "visible part shows it is one. "
    "Fix or delete the suggested boxes. Submit frames without an ambulance empty. "
    "Skip frames that are not camera footage (news studio, graphics, text)."
)


def prediction(lines: list[str], model_version: str) -> dict | None:
    """Label Studio prediction from YOLO lines "class cx cy w h [confidence]"."""
    result, scores = [], []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        score = float(parts[5]) if len(parts) > 5 else 1.0
        scores.append(score)
        result.append({
            "from_name": "label", "to_name": "image", "type": "rectanglelabels", "score": score,
            "value": {"x": (cx - w / 2) * 100, "y": (cy - h / 2) * 100, "width": w * 100, "height": h * 100,
                      "rotation": 0, "rectanglelabels": ["ambulance"]},
        })
    if not result:
        return None
    return {"model_version": model_version, "score": min(scores), "result": result}


def task(image: Path, root: Path, label_file: Path, model_version: str) -> dict:
    """One task whose image is served from Label Studio's local-files root."""
    relative = image.relative_to(root).as_posix()
    item = {"data": {"image": f"/data/local-files/?d={relative}"}}
    lines = label_file.read_text(encoding="utf-8").splitlines() if label_file.exists() else []
    pred = prediction(lines, model_version)
    if pred:
        item["predictions"] = [pred]
    return item


def session(base_url: str, token_file: Path):
    """Requests session authorised with a personal access token (never printed)."""
    import requests

    refresh = token_file.read_text(encoding="utf-8").strip()
    r = requests.post(f"{base_url}/api/token/refresh/", json={"refresh": refresh}, timeout=30)
    r.raise_for_status()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {r.json()['access']}"
    return s


CLASSES = ["ambulance", "police", "fire_truck"]


def clip_of(stem: str) -> str:
    """Clip id of a frame named '<clip>_f<frame>' by atlas_training.frames."""
    return stem.rsplit("_f", 1)[0]


def yolo_lines(annotation: dict) -> list[str]:
    """YOLO lines "class cx cy w h" from one Label Studio annotation."""
    lines = []
    for r in annotation.get("result", []):
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        w, h = v["width"] / 100, v["height"] / 100
        cx, cy = v["x"] / 100 + w / 2, v["y"] / 100 + h / 2
        cls = CLASSES.index(v["rectanglelabels"][0])
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def all_tasks(s, base_url: str, project: int) -> list[dict]:
    tasks, page = [], 1
    while True:
        r = s.get(f"{base_url}/api/tasks/", params={"project": project, "page": page, "page_size": 100,
                                                    "fields": "all"}, timeout=60)
        r.raise_for_status()
        items = r.json().get("tasks", [])
        tasks += items
        page += 1
        if len(items) < 100:
            return tasks


def export(s, base_url: str, projects: list[int], root: Path, out: Path, test_clips: set[str]) -> dict:
    """Write the submitted frames as a dataset source folder for prepare.py.

    Frames of the clips in test_clips go to test/, the rest to train/; prepare.py
    samples val from train. Skipped frames and frames nobody submitted are left out.
    renamed_files.csv gives every frame the original name '<clip>.rf.<frame>', so that
    with augmented_copies the frames of one clip stay in one split.
    """
    import csv
    import os
    import shutil

    import yaml

    if out.exists():
        shutil.rmtree(out)
    counts = {"train": 0, "test": 0, "boxes": {c: 0 for c in CLASSES}, "clips": {"train": set(), "test": set()}}
    rows = []
    for project in projects:
        for t in all_tasks(s, base_url, project):
            done = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
            if not done:
                continue
            image = root / t["data"]["image"].split("?d=", 1)[1]
            clip = clip_of(image.stem)
            split = "test" if clip in test_clips else "train"
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            target = out / split / "images" / image.name
            try:
                os.link(image, target)
            except OSError:
                shutil.copy2(image, target)
            lines = yolo_lines(done[-1])
            (out / split / "labels" / f"{image.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            for line in lines:
                counts["boxes"][CLASSES[int(line.split()[0])]] += 1
            counts[split] += 1
            counts["clips"][split].add(clip)
            rows.append((split, f"{clip}.rf.{image.stem}", image.name))
    (out / "data.yaml").write_text(yaml.safe_dump({"names": CLASSES}), encoding="utf-8")
    with (out / "renamed_files.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "original_name", "short_name"])
        w.writerows(rows)
    (out / ".download_complete").write_text("exported from Label Studio\n")
    counts["clips"] = {k: sorted(v) for k, v in counts["clips"].items()}
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="write submitted frames as a dataset source for prepare.py")
    exp.add_argument("out", type=Path, help="e.g. datasets/raw/own_cctv-v1")
    exp.add_argument("--projects", type=int, nargs="+", required=True)
    exp.add_argument("--test-clip", action="append", default=[], dest="test_clips", help="clip id whose frames go to test (repeat)")
    exp.add_argument("--root", type=Path, required=True, help="Label Studio local-files root")
    exp.add_argument("--token-file", type=Path, required=True)
    exp.add_argument("--url", default="http://localhost:8080")
    imp = sub.add_parser("import", help="create a project and import a frames folder")
    imp.add_argument("folder", type=Path, help="output folder of atlas_training.frames")
    imp.add_argument("--project", required=True)
    imp.add_argument("--token-file", type=Path, required=True)
    imp.add_argument("--root", type=Path, help="Label Studio local-files root (default: parent of folder)")
    imp.add_argument("--url", default="http://localhost:8080")
    imp.add_argument("--model-version", default="ambulance_v2")
    args = parser.parse_args()

    if args.command == "export":
        import json

        s = session(args.url, args.token_file)
        counts = export(s, args.url, args.projects, args.root.resolve(), args.out, set(args.test_clips))
        print(json.dumps(counts, indent=2))
        return

    root = (args.root or args.folder.parent).resolve()
    images_dir = (args.folder / "images").resolve()
    images = sorted(images_dir.glob("*.jpg"))
    tasks = [task(p, root, args.folder / "labels" / f"{p.stem}.txt", args.model_version) for p in images]

    s = session(args.url, args.token_file)
    project = s.post(f"{args.url}/api/projects/", json={
        "title": args.project, "label_config": LABEL_CONFIG, "expert_instruction": INSTRUCTIONS,
        "show_instruction": True, "show_skip_button": True,
    }, timeout=30)
    project.raise_for_status()
    pid = project.json()["id"]
    # Label Studio only serves local files below a storage path registered for the project.
    storage = s.post(f"{args.url}/api/storages/localfiles/", json={
        "project": pid, "title": args.folder.name, "path": str(images_dir), "use_blob_urls": True,
    }, timeout=30)
    storage.raise_for_status()
    imported = s.post(f"{args.url}/api/projects/{pid}/import", json=tasks, timeout=300)
    imported.raise_for_status()
    with_pred = sum("predictions" in t for t in tasks)
    print(f"project {pid}: {len(tasks)} frames imported, {with_pred} with suggested boxes")
    print(f"open {args.url}/projects/{pid}/data")


if __name__ == "__main__":
    main()
