# Ambulance detection: research notes and decisions (S4PT-52)

What the literature and our own measurements say, and what we do because of it. Numbers
from papers are the papers' own; numbers marked *ours* were measured on our test set
(`configs/ambulance_v5.yaml`, Turkish test videos: 9 clips, 159 ambulance boxes).

## The finding that sets the direction

*Ours (21.11.2026):* a COCO detector (yolo11s, car/bus/truck) boxes 129 of 148 Turkish
test ambulances at IoU 0.5; the one-stage ambulance detector v3 finds 52. On the
Adiyaman clip it is 28/28 against 0/28, with ambulances about 103 px wide. The problem
is recognising the vehicle as an ambulance, not finding it, and not resolution (v3 at
960 or 1280 px does not help).

## Decisions

| Problem | What the sources say | Decision | Status |
| --- | --- | --- | --- |
| Recognition | Detector + classifier on the vehicle crop (Akdag et al., VEHITS 2022: ViT on crops, 99.4% accuracy on near views; Fan et al. 2022, Aurora) | Two-stage: COCO vehicles, then a crop classifier (`crops.py`, `twostage.py`), fused with v3 (`twostage.fuse`) | *Ours:* Turkish P 0.75 / R 0.54 against v3 0.78 / 0.33 |
| Single-frame doubt, occlusion | Classification improves when per-frame scores along a track are accumulated (Bebis et al.); ByteTrack keeps occluded objects with low-score boxes, 30-frame buffer (Zhang et al., ECCV 2022); smoothing over a 25-frame buffer with a 30-50% threshold raised precision ~10% for -0.5% recall (Fan et al.) | Track vehicles, decide per track; alarm per passage (`events.py`) | Next |
| NFR-04 is clip level | Acceptance criteria page: precision and recall >= 90% at clip level (O2) | Report clip level as the NFR-04 number, frame level separately | Next |
| Frontal view | Viewpoint information cut fine-grained error by 26% (BoxCars, CVPR 2016) | More frontal ambulance crops; rely on track-level decision | Planned |
| Very small or low-resolution vehicles | ~93% vehicle classification on 100x100 px crops (Tas et al., Sensors 2022); no reliable results found below 32x32; training on high- and low-resolution images together beat super-resolution (59.3% vs 50.8%, arXiv 1605.06695) | Train the classifier also on downscaled crops; report recall by box width and a minimum size as a camera requirement | Planned |
| Synthetic far ambulances | Copy-paste helps small objects (AP-small 0.167 -> 0.179, Kisantal 2019) if pastes do not overlap and originals are kept; no-blend pasting worst (Dwibedi 2017) | Low priority: our misses are not size misses | Deferred |
| Beacon state (FR-07) | ABLDataset: 3,049 images, active blue light, RT-DETR variant P 0.89 / R 0.86; small light found 98% at 30 m by day, 63% at 40 m, 93% at 50 m by night. FlashLightNet: crop sequence (27 frames) + LSTM, F1 0.96 | Beacon decided on the ambulance track (2-3 lit frames in 1-2 s), not per frame; expect weak daytime range | Planned; ABLDataset licence to check |
| False alarms per hour (NFR-05) | Rare events: even a small false-positive rate gives many false alerts; measure false alarms per hour (Bosch, IPVM) | Measure after tracking on >= 10 h of footage without ambulances (we have about 1 h) | Planned |
| New cameras (pilot) | Fixed-camera adaptation with the detector's own labels plus tracking (CVPR 2023); plain self-training can collapse on noisy labels | Measure pilot cameras separately first; adapt only with track-confirmed labels | SE 4920 |
| Night | Labelled night data is the main lever; classical enhancement distorts colours | More labelled night frames, no enhancement step | Ongoing |

## Round 3 (21.11.2026, evening): how to build the next steps

| Topic | What the sources say | What we do |
| --- | --- | --- |
| Linking detections over time | Seq-NMS: link boxes in neighbouring frames at IoU > 0.5, rescore the linked sequence with the **average** score (better than max: 51.5 vs 50.1 mAP); +5 to +7 mAP on ImageNet VID, largest gains for occluded, blurred and extreme-scale objects; fails when similar objects are close (identity switch) and can add false positives (Han et al. 2016). Tubelet rescoring pools classifier scores along a track (Kang et al. 2018) | Track-level decision = **mean** of p(ambulance) over the track's frames, not the max; needs a minimum track length so one confident frame cannot raise an alarm |
| Crop size | Zero-shot CLIP-type accuracy falls from ~70% at 224 px to ~63-65% at 64 px, ~47-54% at 32 px, ~23-28% at 16 px; the drop starts near 64 px and is steep below 32 px; ViTs more robust than CNNs; fine-tuning lowers robustness (LR0.FM, ICLR 2025). Upscaling adds pixels, not detail | Report classifier results by crop width (< 40, 40-64, 64-100, > 100 px); train with crops downscaled to 30-100 px and upscaled back; below ~40 px a single-frame decision is not trusted |
| Mixed-resolution training | Half low-resolution images per batch improved recognition at all resolutions (Knoche and Hormann); training on full and downsampled images together kept accuracy at 64x64 (SpliceMix); train/test resolution mismatch matters (Touvron et al. 2019) | Classifier training set gets a low-resolution copy of every crop (about 50%) |
| Combining scores | Scores of different models are not comparable; calibrate each on held-out data (temperature scaling, a shift on logits may be needed) before fusing; temperature scaling keeps ranking (ProbEn, Chen et al. 2022) | Fit a temperature for the classifier on val crops before the fusion threshold is chosen; choose thresholds on val, report on test |
| Confusable classes | Attention models use few parts; a loss focused on the confusable classes helps (arXiv 1912.06842); errors concentrate on structurally similar classes (vans) | Keep the hardest negatives (white vans, minibuses) at a fixed share of each training batch |
| Clip/event-level scoring | Frame metrics hide event-level failure: models with > 61% frame AUC reached < 10% event precision (arXiv 2604.09327); event matching by temporal overlap, reward early detection (LaAP) | NFR-04 reported per clip with a fixed rule (an alarm inside the labelled passage, +/- 1 s), plus time to first alarm |

## Turkish regulation that affects the labels

- Lights: ambulances use blue-red or blue beacons (Gecis Ustunlugune Sahip Araclar
  Hakkinda Yonetmelik); fire trucks red. ABLDataset covers blue only. Observed on Turkish
  ambulances (team, 21.11.2026): red and blue flash fast one after the other, sometimes
  together, so nearly every frame shows a lit red or blue light; "lit" counts either
  colour, and a window of about 0.5-1 s is enough.
- Scope: beacon state (FR-07) and the long footage for false alarms per hour (NFR-05) are
  not part of S4PT-52 and have no task yet; noted here for whoever takes them.
- Livery (Ambulanslar ... Ambulans Hizmetleri Yonetmeligi, Resmi Gazete 08.01.2025,
  article 8): emergency and intensive-care ambulances white with fluorescent red
  reflective strips; **patient-transport ambulances grey (RAL 7004) with blue strips**.
  The Health Minister announced the grey transport livery from the start of 2026.
- Decision (21.11.2026): grey patient-transport ambulances are out of scope. The model
  targets white emergency ambulances with red strips; grey transport ambulances are not
  labelled as ambulance and are not expected to raise an alarm. Stated as a limit in the
  results note.

## Sources

Akdag et al., Critical Vehicle Detection for ITS, VEHITS 2022 ·
Fan et al., Detection of Active Emergency Vehicles, arXiv 2212.13696 ·
Zhang et al., ByteTrack, ECCV 2022 · Jin et al., Unsupervised Hard Example Mining from
Videos, ECCV 2018 · Kisantal et al., arXiv 1902.07296 · Dwibedi et al., ICCV 2017 ·
Akyon et al., SAHI, arXiv 2202.06934 · Keles et al., arXiv 2203.04799 ·
Sochor et al., BoxCars, CVPR 2016 · Tas et al., Sensors 22(13) 4740, 2022 ·
Peng et al., arXiv 1605.06695 · ABLDataset, arXiv 2603.05058 · FlashLightNet,
Sensors 25(20) 6423, 2025 · Self-Supervised Scene Adaptation, CVPR 2023 ·
IEC 62676-4 pixel density (Axis white paper) · Resmi Gazete 08.01.2025 no. 32776.
