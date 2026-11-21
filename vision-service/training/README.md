# Vision model training and evaluation

Tools to download data, fine-tune detection models with transfer learning and compare
them on a held-out test set. The first experiment is ambulance detection (S4PT-52;
FR-06, UC-2.1; stage S4 of the Vision Pipeline Definition): the pre-trained
SmartTrafficAI weights against a COCO-pretrained YOLO11s fine-tuned on public
emergency-vehicle datasets.

The running vision service does not use this folder; it only loads the chosen weights.
Datasets, weights and run outputs are never committed (see `.gitignore`).

| File | Purpose |
| --- | --- |
| `configs/ambulance_v*.yaml` | One file per experiment: dataset versions, class mapping, seed, training and evaluation settings (v1 public sets, v2 adds night images, v3 adds our own labels) |
| `atlas_training/download.py` | Downloads the Roboflow datasets and the pre-trained weights (SHA-256 recorded) |
| `atlas_training/prepare.py` | Maps source labels to our classes, builds train / val / test, draws labels for checking |
| `atlas_training/train.py` | Fine-tunes the base model on train / val |
| `atlas_training/evaluate.py` | Scores every model on the same test set against NFR-04 |
| `atlas_training/clips.py` | Runs a model on traffic camera clips and logs what it detects |
| `atlas_training/frames.py` | Takes distinct frames from video clips, with a model's boxes as pre-labels |
| `atlas_training/labelstudio.py` | Loads frames into a local Label Studio project and exports the checked labels as a dataset source |
| `atlas_training/metrics.py` | Precision, recall and AP@0.5 for one class, independent of the model's class list |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), an NVIDIA GPU with a current driver
(training on the CPU works but is very slow), about 5 GB of free disk space for the
environment and 2 GB for the data.

```bash
cd vision-service/training
uv python install 3.14.8
uv venv --python 3.14.8 .venv
uv pip sync --python .venv/Scripts/python.exe --index-strategy unsafe-best-match requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/activate                                  # Linux/macOS: source .venv/bin/activate
yolo settings sync=False                                # turn off Ultralytics usage analytics
python -m pytest -q
```

**Python 3.14, not 3.15.** The service is pinned to Python 3.15.2 (SR-16), but on
Windows two packages needed by Ultralytics, PyYAML and pillow-avif-plugin, have no 3.15
wheels yet and would have to be compiled with a C++ compiler. The training tools are
separate from the service, so they use the newest 3.14 release until the wheels exist.

**Roboflow key.** Downloading datasets needs a free Roboflow account. Put its API key in
the repository's `.env` (`ROBOFLOW_API_KEY=...`, see `.env.example`) or in the
environment. Never commit it.

## Steps

All commands run in `vision-service/training` with the environment active.

```bash
# 1. Pre-trained weights and data. Set every dataset "version" in the config first.
python -m atlas_training.download configs/ambulance_v1.yaml --list-versions
python -m atlas_training.download configs/ambulance_v1.yaml

# 2. Check the labels by eye, then build the dataset.
python -m atlas_training.prepare configs/ambulance_v1.yaml --inspect 20   # images in runs/inspect/
python -m atlas_training.prepare configs/ambulance_v1.yaml                 # prints datasets/ambulance_v1/manifest.json

# 3. Pre-trained model on traffic camera clips (notes on what it detects and misses).
python -m atlas_training.clips weights/smarttrafficai.pt path/to/clip.mp4

# 4. Fine-tune (about 100 epochs, early stop after 20 without improvement).
python -m atlas_training.train configs/ambulance_v1.yaml
python -m atlas_training.train configs/ambulance_v1.yaml --epochs 1 --name smoke   # quick check

# 5. Compare on the held-out test set.
python -m atlas_training.evaluate configs/ambulance_v1.yaml \
  --model finetuned=runs/train/ambulance_v1/weights/best.pt
```

`evaluate` writes `runs/eval/<experiment>/results.md`: precision, recall, F1 and
AP@0.5 for the ambulance class of every model, the gap to the NFR-04 target (precision
and recall ≥ 90%), and the threshold with the best F1. The table goes to the Confluence
results note.

## Own labels

Public datasets have almost no traffic-camera, night-time or Turkish ambulances, so we
label frames of such videos ourselves. The videos and frames stay outside the repository
(`D:/code/atlas-data` on the training computer, with a `SOURCES.csv` per folder); videos
with a standard YouTube licence are used for local training and testing only, and models
trained on them are not published.

```bash
# 1. Distinct frames, every 0.5 s, with the current model's boxes as suggestions
python -m atlas_training.frames path/to/clips/*.mp4 --out path/to/labeling/tr_v1   --weights runs/train/ambulance_v2/weights/best.pt
# 2. Label Studio runs on this computer (pip install label-studio in its own environment;
#    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true, LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=path/to/labeling)
python -m atlas_training.labelstudio import path/to/labeling/tr_v1 --project "ATLAS ambulance - tr_v1"   --token-file path/to/.ls_token
# 3. Label vehicles only (ambulance, police, fire_truck), not the word AMBULANS in a caption;
#    the box covers the visible part, also of partly hidden or cut-off ambulances.
# 4. Export the submitted frames; whole videos go to test with --test-clip
python -m atlas_training.labelstudio export datasets/raw/own_cctv-v1 --projects 1 2   --root path/to/labeling --token-file path/to/.ls_token --test-clip <video id>
```

Frames of one video always stay in one split (train, val or test), so the test videos
are never seen in training. The access token stays in a file outside the repository.

### Hard examples from the model's flicker

Instead of frames every 0.5 s, `atlas_training.mine` picks the frames where the model is
probably wrong, from how its detections change between neighbouring frames (Jin et al.,
ECCV 2018): an ambulance detected before and after a frame but not in it (a likely miss,
with the box interpolated) and a confident detection with nothing near it in the frames
around it (a likely false alarm). They are imported and checked like any other frames.

```bash
python -m atlas_training.mine path/to/clips/*.mp4 --out path/to/labeling/mined_v1   --weights runs/train/ambulance_v4/weights/best.pt
```

### Per passage: events found, delay and false alarms

The system raises an alarm per ambulance passage, not per frame, so
`atlas_training.events` smooths the per-frame detections (an alarm needs several
detections in the last second, after Fan et al., 2022) and reports how many labelled
ambulance passages got an alarm, how many seconds after they started, and alarm time with
no ambulance (per hour on footage without ambulances, for FR-11).

```bash
python -m atlas_training.events path/to/test/clips/*.mp4 --labels datasets/raw/own_cctv-v1/test/labels   --weights runs/train/ambulance_v4/weights/best.pt --out runs/eval/events_v4.json
```

## How results stay comparable and reproducible

- **One test set, one scorer.** The pre-trained model has four classes, our model one.
  Both are reduced to their ambulance boxes and scored by `metrics.py` on the same test
  images (IoU ≥ 0.5, matched by confidence). Images without an ambulance stay in the
  test set, so false alarms count.
- **The test set is never seen in training.** Training uses train and val; the best
  epoch is chosen on val. A source's own test split stays the test set.
- **Fixed inputs.** The config pins dataset versions, the class mapping and the seed;
  `requirements.txt` pins every package. `prepare` writes `manifest.json` (images, boxes,
  dropped images and ignored boxes per source, config hash), `train` copies the config
  and the manifest into the run folder, and `weights/manifest.json` records the SHA-256
  of each downloaded weight file.
- **Unchecked labels stop the run.** Every source label must be listed in `class_map`.
  A label that might hide an ambulance (for example the generic "emergency") removes the
  whole image instead of only the box, so that the image is not learned as background.
- **No near copies across splits.** Roboflow sets are often frames of a few videos, and
  augmented exports hold several rotated copies of each image. `prepare` keeps every
  unit in one split: the copies of an image (`augmented_copies`) and the frames of one
  clip (`clip_similarity`: frames whose grey thumbnails correlate at least that much in
  some 90-degree orientation). Test images that still correlate at least
  `split.drop_test_near_copies` with a train or val image of any source are removed from
  test. The counts are in `manifest.json`.
- **Upright test images.** Copies moved into test keep one copy, turned upright: per
  clip, a COCO-pretrained detector picks the orientation of the frame it is surest
  about, and the other frames are aligned to it. `orientation_overrides` sets a copy by
  hand if this is wrong; check `runs/inspect/` after preparing.

## Data and licences

| Source | Used for | Licence | Attribution |
| --- | --- | --- | --- |
| [Emergency_Vehicles_Detection](https://universe.roboflow.com/smarttrafficsignal/emergency_vehicles_detection) | training, test | CC BY 4.0 | smarttrafficsignal, Roboflow Universe |
| [special_vehicles_small_ds](https://universe.roboflow.com/licenseplate-e4lyg/special_vehicles_small_ds) | training, test | CC BY 4.0 | licenseplate-e4lyg, Roboflow Universe |
| [SmartTrafficAI](https://github.com/AmanAgrawalAI/SmartTrafficAI) weights | comparison | Repository MIT; weights trained with Ultralytics, metadata says AGPL-3.0 | AmanAgrawalAI |
| [Ultralytics YOLO11](https://docs.ultralytics.com/) | base model and training library | AGPL-3.0 (or Ultralytics Enterprise licence) | Ultralytics |

CC BY 4.0 allows use and changes with attribution; the table above and the Confluence
note give it. Ultralytics is AGPL-3.0: software that uses it must be offered under a
compatible licence when it is distributed or offered over a network. The repository is
public but has no licence file yet; the team decides on the licence (or on an
Apache-2.0 alternative such as RF-DETR) before trained weights are used in the service.
The licences were taken from the Model & Dataset Research page (12.11.2026).

## Known limits

- None of the datasets says that its vehicles are Turkish, and none is from our pilot
  cameras. Test results on these sets do not show how the model works on İzmir or
  İstanbul cameras; our own labelled footage (S4PT-51) is added when it is ready.
- The test set is small and uneven: in ambulance_v1, 53 traffic-camera ambulance boxes
  from 19 clips (most from one night-time clip) and 102 street-level boxes. Treat the
  scores as a first comparison, not as the NFR-04 verification.
- The near-copy check compares 32×32 grey thumbnails. It finds the same photo and
  consecutive frames, not the same vehicle filmed from another place.

## Updating the lock file

Change `requirements.in`, then:

```bash
uv pip compile requirements.in --python-version 3.14 --python-platform x86_64-pc-windows-msvc \
  --index-strategy unsafe-best-match --only-binary :all: --emit-index-url --no-header -o requirements.txt
```

The lock file is for Windows with CUDA 13.2. On Linux, use
`--python-platform x86_64-unknown-linux-gnu` and keep the result next to it.
