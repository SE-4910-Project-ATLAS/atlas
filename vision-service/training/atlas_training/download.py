"""Download the datasets and pre-trained weights named in an experiment file.

    python -m atlas_training.download configs/ambulance_v1.yaml --list-versions
    python -m atlas_training.download configs/ambulance_v1.yaml

Roboflow needs a free account's API key in the ROBOFLOW_API_KEY environment variable
or in the repository's .env file (never committed). Files go to datasets/raw/ and
weights/, which Git ignores. The SHA-256 of every weight file is written to
weights/manifest.json so that a result can be traced back to the exact file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from . import config as cfg

ENV_FILE = cfg.TRAINING_DIR.parent.parent / ".env"
COMPLETE = ".download_complete"   # written last, so a broken download is redone


def api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ROBOFLOW_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set (environment variable or the repository's .env file).")
    return key


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_versions(exp: cfg.Experiment) -> None:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key())
    for source in exp.datasets:
        project = rf.workspace(source.workspace).project(source.project)
        print(f"{source.id}:")
        for v in project.versions():
            print(f"  version {v.version}: {v.images} images, splits {v.splits}")


def extract_short(zip_path: Path, target: Path) -> list[tuple[str, str, str]]:
    """Unpack a Roboflow export, shortening long file names (see config.short_stem).

    Roboflow keeps the original upload names, some over 200 characters, which Windows
    cannot open under its default 260-character path limit. An image and its label
    file share a stem, so both get the same short name.
    """
    for split in ("train", "valid", "test"):
        shutil.rmtree(target / split, ignore_errors=True)   # leftovers of a failed unpack
    renamed = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename)
            name = cfg.short_stem(parts.stem) + parts.suffix
            if name != parts.name:
                renamed.append((str(parts.parent), parts.name, name))
            out = target / parts.parent / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return renamed


def shorten_in_place(target: Path) -> list[tuple[str, str, str]]:
    """Shorten long names in an export that Roboflow could unpack itself."""
    renamed = []
    for path in sorted(target.glob("*/*/*")):
        short = cfg.short_stem(path.stem) + path.suffix
        if short != path.name:
            path.rename(path.with_name(short))
            renamed.append((path.parent.relative_to(target).as_posix(), path.name, short))
    return renamed


def download_datasets(exp: cfg.Experiment) -> None:
    from roboflow import Roboflow

    missing = [s.id for s in exp.datasets if s.version is None]
    if missing:
        sys.exit(f"Set 'version' in the config for: {', '.join(missing)} (see --list-versions).")
    rf = Roboflow(api_key=api_key())
    for source in exp.datasets:
        target = source.download_dir
        if (target / COMPLETE).exists():
            print(f"{source.id}: already downloaded in {target}")
            continue
        zip_path = target / "roboflow.zip"
        if zip_path.exists():
            print(f"{source.id}: unpacking the earlier download {zip_path}")
            renamed = extract_short(zip_path, target)
        else:
            print(f"{source.id}: downloading version {source.version} to {target}")
            project = rf.workspace(source.workspace).project(source.project)
            try:
                project.version(source.version).download("yolov11", location=str(target), overwrite=True)
                renamed = shorten_in_place(target)
            except OSError as error:
                if not zip_path.exists():
                    raise
                print(f"{source.id}: Roboflow could not unpack ({error.__class__.__name__}); unpacking with short names")
                renamed = extract_short(zip_path, target)
        if renamed:
            with (target / "renamed_files.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["folder", "original_name", "short_name"])
                writer.writerows(renamed)
            print(f"{source.id}: {len(renamed)} long file names shortened, see renamed_files.csv")
        zip_path.unlink(missing_ok=True)
        (target / COMPLETE).write_text(f"version {source.version}\n")


def download_weights(exp: cfg.Experiment) -> None:
    cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.WEIGHTS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for model in exp.pretrained:
        if not model.path.exists():
            print(f"{model.id}: downloading {model.url}")
            tmp = model.path.with_suffix(".part")
            urllib.request.urlretrieve(model.url, tmp)
            tmp.replace(model.path)
        digest = sha256(model.path)
        if model.id in manifest and manifest[model.id]["sha256"] != digest:
            print(f"WARNING {model.id}: file differs from the one recorded earlier")
        manifest[model.id] = {"url": model.url, "licence": model.licence, "sha256": digest}
        print(f"{model.id}: {model.path.name} sha256 {digest}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--list-versions", action="store_true", help="show the Roboflow versions and exit")
    parser.add_argument("--weights-only", action="store_true", help="download only the pre-trained weights")
    args = parser.parse_args()
    exp = cfg.load(args.config)
    if args.list_versions:
        list_versions(exp)
        return
    download_weights(exp)
    if not args.weights_only:
        download_datasets(exp)


if __name__ == "__main__":
    main()
