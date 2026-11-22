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

## Clip-level results, first run (21.11.2026)

Two-stage system with ByteTrack, decision = mean p(ambulance) over the track, 9 Turkish
test videos (7 with ambulances, 2 MOBESE compilations without). Passages = labelled
ambulance frames of one shot (shot cuts by HSV histogram correlation).

| Setting | Passages found | Videos with ambulance found | Videos without ambulance with an alarm |
| --- | --- | --- | --- |
| min 5 frames, mean >= 0.5 (fixed before looking at test) | 12/16 | 6/7 | 0/2 |
| min 5 frames, mean >= 0.8 | 9/16 | 5/7 | 0/2 |

- Fusing v3 into the track scores changed nothing at clip level.
- Alarm spans outside labelled passages in videos with ambulances were checked by eye:
  nearly all show real ambulances in frames that are not labelled (labels are one frame
  per second, and frames between labelled empty frames are not all empty). So false
  alarms are counted only on videos checked to have no ambulance; 3.7 min is far too
  little for NFR-05 and more ambulance-free footage is needed.
- Missed passages: Sivas (534 px video, ambulance tiny and overturned), Usak 21 s
  (blurred, behind a car), Kutahya 27 s (overturned ambulance seen from below),
  a one-frame passage in the night crash. Hard or borderline cases.
- Thresholds: 5 frames and 0.5 were fixed in advance; the other settings are a
  sensitivity check only, not a choice made on test.

## Classifier calibration and threshold (21.11.2026)

`calibrate.py` on the val crops of `crops_v5e` (3,465 crops, 385 ambulances), classifier
trained with low-resolution copies:

- The classifier is already well calibrated: temperature 1.47, expected calibration
  error 0.010 before and 0.006 after. Temperature scaling is not needed in the pipeline.
- Crop level on val: threshold 0.5 gives precision 0.94 / recall 0.85, 0.6 gives
  0.96 / 0.83. The difference is small, so the track threshold stays at the 0.5 fixed in
  advance; 0.6 is reported only as a sensitivity check.
- Gap found: 363 of the 385 val ambulances are wider than 100 px (mostly foreign
  street-level images), so val says almost nothing about small or far ambulances.
  Size-dependent results must come from the Turkish test videos.

## Where the two-stage system still misses (Turkish test frames, classifier v5e, p >= 0.5)

Recall by ambulance width: 40-64 px 2/24, 64-100 px 12/21, 100-200 px 23/50, > 200 px
36/64. So below ~64 px it fails, but above that recall stays near 0.5 whatever the size:
size is not the main cause. Of the 55 missed ambulances wider than 100 px, 19 have no
COCO vehicle box and 36 have a box with a low classifier score. Looked at by eye:

1. Head-on, moving ambulance (Ankara): clearly an ambulance with the red crescent, scored
   0.00-0.27. The training crops have few frontal Turkish ambulances.
2. Night scenes with blue/red glare: the COCO detector gives no box at all ("no box").
3. Overturned or crashed ambulances (Kutahya, night crash), and one carried on a tow
   truck: low scores or boxes that do not match. These are not ambulances on duty that
   drivers must give way to; whether they belong in the FR-06 test is a scope question.
   Results are reported both with and without them, never by silently dropping them.

Next steps from this: more frontal Turkish ambulance crops (mined from training videos),
and for night glare, let the one-stage detector add boxes the COCO detector misses.

## Held-out test (test2) and the next round (22.11.2026)

New held-out set `D:/code/atlas-data/clips/test2`: 14 Turkish videos never used for
training or tuning, labelled as time intervals (`labels.json`): 8 with an ambulance,
6 without (MOBESE compilations, 23 min in all). Old 9 test videos become the
development set on which settings are chosen; test2 is scored once per model.

Classifier v5e, 12.5 frames/s read, fixed in advance (>= 5 frames, mean >= 0.5):
videos with an ambulance found 7/8 (87.5%, Wilson 95% interval 53-98%), passages 8/9,
ambulance-free videos with an alarm 1/6, 8 alarm spans outside passages in 32 min.
All 8 checked by eye: real false alarms (white minibuses and vans, a truck with red
advertising, red/blue lights at night, an emblem van); 7 of 8 last under 1.6 s.
Chosen on the development set: mean >= 0.6 (false alarm spans 5 -> 2 at equal recall).

Next round:
- Positives: 2,252 more Turkish ambulance crops from labelled ambulances followed along
  their tracks in the training videos (many frontal views), checked on a sample.
- Negatives: false alarms mined from the Sariyer recording (different camera).
- Frames read at 5 per second (NFR-02) instead of 12.5, so the track rule means the
  same time on every clip and long footage can be processed.
- False alarms per hour on 5 h of the Izmir square camera (2.5 h day, 2.5 h night),
  measurement only: this camera is never used for training.

## Frame rate (development set, classifier v5e, 22.11.2026)

| Reading | Frames read | Passages | Videos with ambulance | False alarm spans |
| --- | --- | --- | --- | --- |
| 12.5/s, >= 5 frames, mean >= 0.5 | ~8,800 | 11/16 | 6/7 | 5 |
| 5/s fixed (NFR-02 minimum), same rule | 3,263 | 10/16 | 5/7 | 5 |
| 5/s, 12.5/s for 3 s after a track scores >= 0.3 | 5,517 | 13/16 | 6/7 | 9 |

Reading only 5 frames per second loses ambulances (tracks break, fewer frames to vote).
Raising the rate while a candidate is in view, as NFR-02 already allows for cameras with
an emergency-vehicle candidate, recovers them at 37% fewer frames than 12.5/s, at the
cost of more short false alarms. The false alarms are what the next classifier targets.

## Track decision rule: mean vs sequential test (development set, v5e, adaptive rate)

| Rule | Passages | Videos with ambulance | False alarm spans |
| --- | --- | --- | --- |
| mean >= 0.5, >= 5 frames | 13/16 | 6/7 | 9 |
| mean >= 0.6, >= 5 frames | 11/16 | 5/7 | 4 |
| SPRT, evidence >= 3, per-frame log-odds clipped to +/- 2 | 12/16 | 6/7 | 5 |

The SPRT (Wald; multiple-look fusion, Koch et al. CVPR-W 2013) sums each frame's
log-odds, clipped so that one over-confident frame cannot decide and correlated frames
are not over-counted. It sits between the two mean thresholds: one passage less than
mean >= 0.5 for four false alarms less. With 16 passages a difference of one is within
noise, so the rule is chosen again with the next classifier, on this set only.

## False alarms on the İzmir square camera (classifier v5e, 22.11.2026)

2.5 h of daytime footage at 5 frames/s (45,001 frames), >= 5 frames, mean >= 0.5:
3 alarm spans, all checked by eye and all false (a distant white van, a parked white
van behind a tree next to a turquoise car). That is 1.2 per hour, 95 % upper bound
3.1 per hour; NFR-05 asks for at most 0.125 per hour (1 per 8 h). The spans are short
(0-4.4 s). Proving 0.125 per hour with zero alarms would need 24 h of footage (rule of
three), so NFR-05 cannot be shown with the footage we have even at zero alarms; the
report gives the measured rate and its bound.

## Classifier v5f rejected (22.11.2026)

v5f = v5e training set + 2,252 ambulance crops propagated along tracks + 48 Sariyer
negatives. Validation accuracy 97.4 % (v5e 97.7 %). On the ambulance-free MOBESE
compilation of the development set it raised 13 alarm spans at mean >= 0.5, v5e none;
the spans are white minibuses and vans. Likely cause: the tracker sometimes jumps from
the ambulance to a nearby white vehicle, so some propagated "ambulance" crops show vans;
a sample check (a few per cent wrong) was not strict enough, because those few crops
teach exactly the confusion we want to remove. v5e stays the classifier. A safer way to
propagate would keep only crops whose box stays close to the box interpolated between
two labelled frames; not done for this sprint.
Flip TTA with v5f: same passages, slightly fewer false alarms at mean >= 0.6 (11 vs 14);
not adopted with v5e because it was not measured there.

## Parked vehicles and a motion rule (22.11.2026)

Held-out test with the adaptive rate (chosen on the development set): videos with an
ambulance found 7/8, but ambulance-free videos with an alarm 3/6 (1/6 at a fixed
12.5 frames/s). The development set has only two ambulance-free videos, so it could not
show this. The Sariyer recording (53 min, a camera v5e never saw) is therefore used as
a false-alarm development set: 44 alarm spans at mean >= 0.5, nearly all on parked
vehicles (a yellow taxi, a white van, a red car) under blue shop lighting. The tracker
loses and finds them again, and every new track is a new alarm.

Rule tested next: a track can alarm only after its box centre has moved at least one
box width (`only_moving`). Drivers are alerted with the vehicle's speed and direction
(FR-08), so a parked vehicle is not one they must give way to. Caveat: several test
ambulances stand still at crash scenes; results are reported with and without the rule.
The frame-count rule is also replaced by a time rule (`alarm_from_tracks_time`), so a
faster read rate does not make the rule easier to meet.

### How others suppress alarms on things that do not move (search, 22.11.2026)

Read from abstracts, patent claims and summaries, not full re-implementations:

- **Background memory per fixed camera** (BEM, arXiv 2604.11714): keeps embeddings of the
  quasi-static background and suppresses detections that match it, without retraining.
  This fits our case well: the Sariyer alarms are the same parked vehicles in the same
  places. Our version would be a per-camera memory of classifier crops that alarmed
  while not moving; a new track that matches one closely (embedding distance, same
  place) is held back. Not built yet. It needs a few minutes of footage per camera and
  must be cleared when the scene changes.
- **Memory of false-positive regions** (patent US12430919) and **per-camera masks** over
  places that keep alarming (US12387480): the same idea in a simpler form. A mask is
  cheap but would also hide a real ambulance parked at the same spot.
- **Tracker feedback for stopped objects** (Senior, Tian, Lu, CCNY, AVSS 2010): without
  feedback a stopped object fades into the background and comes back as a new target,
  as our re-found tracks do. Their fix keeps the stopped object's identity. For us that
  means a longer track buffer or re-identification, so one parked taxi is one alarm and
  not twenty.
- **Hard negatives from recurring background** (SciTePress 2019, 73768): the recurring
  false detections of a camera are the best negatives for it. We did this with İzmir
  hard negatives; Sariyer stays unseen because it is the false-alarm development set.
- **Self-supervised scene adaptation** (CVPR 2023): fine-tunes per camera from tracker
  pseudo-labels. Too heavy for this sprint.

Order chosen: (1) motion rule, already built; (2) count parked-vehicle alarms once per
place (the tracker-feedback idea, without changing the detector); (3) per-camera memory
only if (1) and (2) are not enough. All three are judged on the development set and
Sariyer only, never on test2 or İzmir.

### Motion rule rejected; what separates false alarms (22.11.2026, evening)

- Motion rule on the development set: passages 13/16 → 8/16 (half a box width) and 3/16
  (one width). An approaching ambulance grows instead of moving sideways, and news
  cameras follow it. Rejected.
- Once per place (`alarm_once_per_place`): development set unchanged at 13/16, false spans
  9 → 6. The Sariyer result decides whether it is kept.
- The development set cannot measure false alarms: most of its "false" alarming tracks
  score p = 1.00 for every frame (PGcW tracks 6, 7, 9; D7UW 360), so they are almost
  certainly real ambulances in unlabelled frames. Score statistics (median, share of
  frames above 0.5) do not separate real from false there.
- Sariyer (earlier run): 11 of 12 alarming tracks lie in one 5.5-minute window (parked
  vehicles under blue shop light), with mean 0.53-0.67 and never near 1.0.

### Flashing-light evidence

Right of way needs light and sound signals together (Geçiş Üstünlüğüne Sahip Araçlar
Hakkında Yönetmelik, from 2918 KTK art. 71), so a driver must give way only to an ambulance
with its beacon on. Shop light and red advertising are steady; a beacon flashes.
`atlas_training.lights` saves the lit red and blue share of each vehicle box per frame;
the rule `alarm_with_lights` lets a track that has flashed (range of either share within
1 s at least `min_flicker`) alarm at mean 0.5 and asks tracks that have not for a
stricter mean (0.7, 0.8, or a requirement at 1.01). Similar ideas: per-frame light
classification with a temporal smoother (Fan et al., Aurora, arXiv 2212.13696), matching
the flash frequency (US11341753 family), a cascade of colour, persistence and per-place
cooldown before an alarm (arXiv 2607.03131).
Thresholds: on labelled ambulance boxes, S >= 120 / V >= 200 found lit pixels in only
0-46 % of boxes per clip (beacon cores are nearly white in compressed video); S >= 80 /
V >= 160 is used. Open question for the team: making the light a requirement (strict
1.01) changes FR-06 scope from "ambulance" to "ambulance on duty", so it is reported
but not chosen without the team. Decision (Eren, 22.11.2026): light lowers the bar, it is not
required; only strict 0.7 / 0.8 can be chosen, 1.01 is reported for reference.

Next if these are not enough: hard negatives of Turkish white minibuses and panel vans
from new footage (not İzmir or Sariyer, which stay for measuring), as the site-negative
mining in the vehicle-detection literature (e.g. US8971581).

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
IEC 62676-4 pixel density (Axis white paper) · Resmi Gazete 08.01.2025 no. 32776 ·
BEM, arXiv 2604.11714 · US12430919 · US12387480 · Senior, Tian, Lu, AVSS 2010 ·
SciTePress 2019 paper 73768 · US11341753 · arXiv 2607.03131 · US8971581 ·
Geçiş Üstünlüğüne Sahip Araçlar Hakkında Yönetmelik (2918 KTK art. 71).
