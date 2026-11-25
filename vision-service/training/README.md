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
| `atlas_training/mine.py` | Finds frames worth labelling from where the model flickers in video |
| `atlas_training/events.py` | Smoothed per-frame alarm, scored per ambulance passage |
| `atlas_training/crops.py` | Vehicle crop set (ambulance / other) for the classifier of the two-stage system |
| `atlas_training/lowres.py` | Adds low-resolution copies of the train crops |
| `atlas_training/twostage.py` | COCO vehicle detector + crop classifier, scored per frame; fusion with a one-stage model |
| `atlas_training/tracks.py` | Tracks vehicles, decides per track (mean classifier score) and scores per clip (NFR-04) |
| `atlas_training/rules.py` | Compares track decision rules on saved runs (mean, time, SPRT, once per place, light) |
| `atlas_training/lights.py` | Flashing red / blue light evidence per track |
| `atlas_training/diagnose.py` | Per labelled ambulance: boxed by the detector? classifier score? By size and day / night |
| `atlas_training/trackmine.py` | Mines classifier crops along vehicle tracks of new videos |
| `atlas_training/sheets.py` | Contact sheets of mined tracks that still need a verdict |
| `atlas_training/buildmined.py` | Crop set from checked tracks: preview sheets, exclude list, low-resolution copies |
| `atlas_training/timesheet.py` | One frame per n seconds of a video, for time-interval test labels |
| `atlas_training/report.py` | Applies the chosen track rule once to held-out runs; Wilson intervals, false alarms per hour, `--intervals` (audited labels), `--reviewed` (eye-checked camera alarms) |
| `atlas_training/demo.py` | Demo video of the final system on a clip (boxes, track scores, AMBULANS banner); output stays local |
| `atlas_training/bench.py` | Compares classifiers on the eye-checked hard-track set (dev + Sarıyer), with paired bootstrap intervals; use it before any dev-video run |
| `atlas_training/leakcheck.py` | Leakage check before new videos are used: DINOv2 frame similarity against the test videos, channel graphics left out |
| `atlas_training/addneg.py` | Adds checked negative crops (e.g. a public van set, ambulances excluded) to a copy of a crop set |
| `atlas_training/embedprobe.py` | Experiment: linear probe on frozen DINOv2 / timm features of the crops (tried, not used) |
| `atlas_training/night.py` | Synthetic night copies of crops (tried, not used: v5h) |
| `atlas_training/calibrate.py` | Temperature scaling of the classifier (checked: v5e already calibrated) |
| `RESEARCH.md` | Findings from the literature and our measurements, and the decision taken for each problem |

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

### Two-stage recognition and clip-level scoring

A generic COCO detector finds almost all ambulances as vehicles; the one-stage ambulance
detector misses most Turkish ones (see `RESEARCH.md`). So vehicles are cropped and a
classifier decides "ambulance or not"; in video the decision is the mean over the
vehicle's track, and NFR-04 is scored per clip.

```bash
python -m atlas_training.crops configs/ambulance_v5.yaml --out datasets/crops_v5b --neg-per-pos 8
python -m atlas_training.lowres datasets/crops_v5b --out datasets/crops_v5e
yolo classify train data=datasets/crops_v5e model=weights/yolo11s-cls.pt imgsz=224 epochs=30 project=runs/cls name=crops_v5e
python -m atlas_training.twostage configs/ambulance_v5.yaml --classifier runs/cls/crops_v5e/weights/best.pt   --compare v3=runs/train/ambulance_v3/weights/best.pt@640 --out runs/eval/ambulance_v5/twostage.json
python -m atlas_training.tracks path/to/test/clips/*.mp4 --labels datasets/raw/own_cctv-v2/test/labels   --classifier runs/cls/crops_v5e/weights/best.pt --one-stage runs/train/ambulance_v3/weights/best.pt@640   --out runs/eval/tracks/clips.json
```

Thresholds are chosen on val, not on test.

**Final system (23.11.2026):** COCO YOLO11s vehicles (960 px, conf 0.1) → ByteTrack
(`configs/tracker_ambulance.yaml`) → classifier `runs/cls/crops_v5g` on 224 px square
crops → a track is an ambulance when the mean p over at least 5 frames is >= 0.7 →
alarm once per place (`alarm_once_per_place`, radius 0.5 box widths, 30 min). Frames are
read at 5/s and at 12.5/s for 3 s while a track scores >= 0.3 (NFR-02). Rule and
threshold were chosen on the development set and the Sariyer recording only; test2,
test3 and İzmir were scored once.

### More Turkish crops from new videos (classifier v5g)

```bash
python -m atlas_training.diagnose datasets/raw/own_cctv-v2/test --classifier runs/cls/crops_v5e/weights/best.pt --out runs/eval/diagnose_dev.csv
python -m atlas_training.trackmine D:/code/atlas-data/clips/tr3/*.mp4 --classifier runs/cls/crops_v5e/weights/best.pt --candidate-p 0.2 --out D:/code/atlas-data/labeling/trackmine_v1
python -m atlas_training.sheets D:/code/atlas-data/labeling/trackmine_v1     # mark each track in verdicts.csv
python -m atlas_training.buildmined datasets/crops_v5e D:/code/atlas-data/labeling/trackmine_v1 --preview D:/code/atlas-data/labeling/trackmine_v1/check
python -m atlas_training.buildmined datasets/crops_v5e D:/code/atlas-data/labeling/trackmine_v1 D:/code/atlas-data/labeling/trackmine_v2     --exclude D:/code/atlas-data/labeling/trackmine_v1/check/exclude.txt D:/code/atlas-data/labeling/trackmine_v2/check/exclude.txt --max-pos 8 --out datasets/crops_v5g
yolo classify train data=datasets/crops_v5g model=weights/yolo11s-cls.pt imgsz=224 epochs=30 patience=10 batch=64 seed=42 project=runs/cls name=crops_v5g
python -m atlas_training.rules runs/eval/tracks/g5g_dev.json runs/eval/tracks/g5g_sariyer.json --cuts runs/eval/tracks/cuts_dev.json --out runs/eval/tracks/rules_v5g.json
```

Every mined track is looked at and marked ambulance / other / skip, with the time range
where it is the right vehicle; then every low-scoring positive crop and every negative
crop is checked tile by tile. Labels propagated along tracks without this check taught
van false alarms (classifier v5f, rejected).

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

- Small ambulances (40-64 px wide) are still rarely recognised (6 of 24 development
  frames with v5g); a third of them are not boxed by the vehicle detector at all.
- Clip-level tests are Turkish news, security-camera and phone videos, not the pilot
  cameras; the false-alarm recordings are one İzmir square camera and one Sariyer camera.

## Updating the lock file

Change `requirements.in`, then:

```bash
uv pip compile requirements.in --python-version 3.14 --python-platform x86_64-pc-windows-msvc \
  --index-strategy unsafe-best-match --only-binary :all: --emit-index-url --no-header -o requirements.txt
```

The lock file is for Windows with CUDA 13.2. On Linux, use
`--python-platform x86_64-unknown-linux-gnu` and keep the result next to it.
