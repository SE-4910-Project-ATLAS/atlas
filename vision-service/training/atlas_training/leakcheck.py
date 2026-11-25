"""Find new videos that share footage with the held-out test videos (leakage check).

    python -m atlas_training.leakcheck D:/code/atlas-data/clips/tr4/*.mp4 \
        --against D:/code/atlas-data/clips/test2/*.mp4 D:/code/atlas-data/clips/test3/*.mp4

News channels re-upload the same camera footage under other titles, often scaled down
inside their own frame with a caption bar and a logo, so neither a title check nor a
perceptual hash finds it (a difference hash missed a known pair of test3 uploads).
Self-supervised ViT features are robust to such copies (DINO, Caron et al., ICCV 2021,
evaluate copy detection), so one frame every --every seconds is embedded with frozen
DINOv2 ViT-S/14 (Apache 2.0) and compared by cosine similarity.

A pair is reported when at least --min-frames frames of the new video have a frame in the
other video with similarity >= --similarity. Calibration on test3 (24.11.2026): two known
re-uploads, one scaled into a news frame, gave 10 of 11 and 49 of 49 frames >= 0.85;
unrelated ambulance-crash videos peaked at 0.55–0.75. Flagged pairs are looked at by eye
before a video is dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MODEL = "vit_small_patch14_dinov2.lvd142m"


class Embedder:
    def __init__(self, device: str):
        import timm
        import torch

        self.torch = torch
        self.device = device
        self.model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        cfg["input_size"] = (3, 224, 224)
        self.transform = timm.data.create_transform(**cfg)

    def video(self, path: Path, every: float) -> np.ndarray:
        """Unit-length embeddings of one frame every `every` seconds (flat frames skipped)."""
        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        images = []
        for frame in range(0, n, max(1, round(fps * every))):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                break
            if image.std() < 8:          # black or flat frames match everything
                continue
            images.append(self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
        capture.release()
        if not images:
            return np.zeros((0, 384), np.float32)
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), 64):
                e = self.model(self.torch.stack(images[i:i + 64]).to(self.device)).float()
                out.append(self.torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        return np.concatenate(out)


def boilerplate(videos: dict[str, np.ndarray], similarity: float, min_videos: int = 3) -> dict[str, np.ndarray]:
    """Per video, a mask of frames that are channel graphics rather than footage.

    End cards ("Abone olun"), channel intros and outros recur nearly unchanged in many
    uploads of one channel, so they match across unrelated videos (seen on dev2: a TRT
    outro, an İHA subscribe card). A frame with a near-identical frame in at least
    `min_videos` different videos (its own included) is treated as such a graphic."""
    names = list(videos)
    allv = np.concatenate([videos[n] for n in names]) if names else np.zeros((0, 1))
    owner = np.concatenate([np.full(len(videos[n]), i) for i, n in enumerate(names)]) if names else np.zeros(0)
    out = {}
    for i, n in enumerate(names):
        e = videos[n]
        if not len(e):
            out[n] = np.zeros(0, bool)
            continue
        close = (e @ allv.T) >= similarity
        out[n] = np.array([len(set(owner[row].tolist())) >= min_videos for row in close])
    return out


def matching_frames(a: np.ndarray, b: np.ndarray, similarity: float) -> int:
    """How many frames of a have a frame in b at least this similar (cosine)."""
    if not len(a) or not len(b):
        return 0
    return int(((a @ b.T).max(axis=1) >= similarity).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--against", nargs="+", type=Path, required=True)
    parser.add_argument("--every", type=float, default=2.0)
    parser.add_argument("--similarity", type=float, default=0.85)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    embed = Embedder(args.device)
    ref = {v.stem: embed.video(v, args.every) for v in args.against if v.stem not in {x.stem for x in args.videos}}
    new = {v.stem: embed.video(v, args.every) for v in args.videos}
    graphics = boilerplate({**ref, **new}, args.similarity)
    clean_ref = {n: e[~graphics[n]] for n, e in ref.items()}
    dropped = sum(int(m.sum()) for m in graphics.values())
    print(f"{dropped} channel-graphic frames left out (near-identical in >= 3 videos)")
    flagged = 0
    for video in args.videos:
        mine = new[video.stem][~graphics[video.stem]]
        hits = [(name, matching_frames(mine, e, args.similarity)) for name, e in clean_ref.items()]
        hits = sorted([x for x in hits if x[1] >= args.min_frames], key=lambda x: -x[1])
        # The same event re-uploaded by three or more channels also looks like a graphic;
        # such pairs are listed apart, to be looked at, never silently dropped.
        weak = [(name, matching_frames(new[video.stem], e, args.similarity)) for name, e in ref.items()]
        weak = [x for x in weak if x[1] >= args.min_frames and x[0] not in {h[0] for h in hits}]
        if hits:
            flagged += 1
            print(f"{video.stem}: {', '.join(f'{n} ({k}/{len(mine)} frames)' for n, k in hits)}", flush=True)
        if weak:
            print(f"{video.stem} (graphics only, check by eye): {', '.join(f'{n} ({k})' for n, k in weak)}", flush=True)
    print(f"{flagged} of {len(args.videos)} videos share footage with the reference videos")


if __name__ == "__main__":
    main()
