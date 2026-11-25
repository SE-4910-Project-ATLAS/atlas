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
- Once per place (`alarm_once_per_place`): development set unchanged at 13/16 passages and
  6/7 clips, false spans 9 → 6. Sariyer (53 min): **44 → 8 alarm spans** with a 10-minute
  memory, **→ 5 with 30 minutes** (radius 0.5 box widths); the parked vehicles came back
  every 7-8 minutes. Radius 1.0 or 1.5 loses two development passages and gains nothing
  on Sariyer. Chosen: radius 0.5, window 30 min. The motion rule gets Sariyer to 0-1 but
  costs development recall (above), so it stays rejected.
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

Development set with light fractions (l_dev, v5e): a flashing track is seen in 14/16
passages at min_flicker 0.003, 12/16 at 0.01, 11/16 at 0.03. Light rule with strict 0.7
or 0.8 keeps 13/16 passages and 6/7 clips only at 0.003; at 0.01 or 0.03 it loses 1-2
passages. Light as a requirement (1.01): 12/16, 9/16, 6/16, which supports the decision
not to require it. Whether 0.003 is above the flicker that compression and box jitter
give a parked vehicle is decided on Sariyer.

Next if these are not enough: hard negatives of Turkish white minibuses and panel vans
from new footage (not İzmir or Sariyer, which stay for measuring), as the site-negative
mining in the vehicle-detection literature (e.g. US8971581).

## Small and night ambulances; classifier v5g (23.11.2026)

Diagnosis (`atlas_training.diagnose`, development frames, v5e): for each labelled ambulance,
is it boxed by the COCO detector, and what does the classifier say on its crop?

| group | n | boxed (960 px) | p >= 0.5 on the labelled crop |
| --- | --- | --- | --- |
| width 40-64 px | 24 | 16 | 6 |
| width 64-100 px | 21 | 21 | 12 |
| night | 33 | 27 | 21 |
| all | 166 | 146 | 95 |

The classifier, not the detector, loses small and night ambulances. A 1280 px detector
input boxes fewer (142), and a lower detector confidence (0.03, 0.01) adds 2 night boxes
but no more correct ones. Only ~3 % of v5e's ambulance crops are dark.

Fix: more real Turkish ambulance crops, mined along vehicle tracks and checked by eye
(`atlas_training.trackmine`, `sheets`, `buildmined`). Jin et al. (ECCV 2018): the frames a
tracked object is missed in are the most useful new examples. 54 new Turkish clips
(`clips/tr3`, local only), searched for night, head-on and top-down views; never used for
testing. Every candidate track was looked at and marked ambulance / other / skip, with a
time range where the track jumped to another vehicle (211 ambulance tracks, 40 of them at
night; 144 other; 124 skipped: studio graphics, picture-in-picture, mixed crops, a foreign
livery). Then every low-scoring positive crop and every negative crop was checked tile by
tile (1,253 crops) and 123 were dropped. Dropped on purpose: night crops that show only
two headlights (they would teach "any car at night"), and plain white vans seen from above
with no visible marking (they would teach the white-van false alarm).
Up to 8 crops per track (half the lowest-scoring, half spread), each with a degraded
low-resolution copy for small ambulances. Ratio ambulance:other kept near 1:3 (v5e
1:5.9) so the classifier is not pushed towards saying "ambulance".
v5h: v5g plus synthetic night copies of 30 % of the ambulance crops (`atlas_training.night`:
gamma, cool cast, noise). Published evidence is for learned day-to-night translation
(Lin et al., IEEE T-ITS 2021; Yang et al., arXiv 2412.16478), not for gamma and noise, so
v5h is kept only if it beats v5g on the night frames of the development set.

Result on the development frames (ambulances boxed by the detector and p >= 0.5):

| group | v5e | v5g | v5h |
| --- | --- | --- | --- |
| all (166) | 81 | **104** | 100 |
| night (33) | 17 | **20** | 16 |
| width 64-100 px (21) | 12 | **18** | 19 |
| width 40-64 px (24) | 5 | **6** | 6 |

v5h (synthetic night copies) is rejected: worse at night than v5g. Real night crops help,
gamma-and-noise copies do not. The smallest ambulances hardly improve: only 16 of 24 are
boxed by the detector, and at 40-64 px there is little left to see. v5g goes on to the
clip-level runs on the development set and Sariyer (false alarms).

Clip level, v5g (development set and Sariyer only; chosen before test2 and İzmir):

| rule | dev passages | dev clips | dev ambulance-free clips with alarm | Sariyer alarms (53 min) |
| --- | --- | --- | --- | --- |
| v5e mean 0.5 | 13/16 | 6/7 | 0/2 | 44 |
| v5e once per place 0.5 | 13/16 | 6/7 | 0/2 | 5 |
| v5g mean 0.5 | 14/16 | 6/7 | 1/2 | 12 |
| v5g mean 0.6 | 14/16 | 6/7 | 0/2 | 1 |
| **v5g mean 0.7** | **14/16** | **6/7** | **0/2** | **0** |

v5g scores real ambulances higher, so a stricter threshold costs no development passage
and removes the parked vehicles. Most of v5g's "false" development spans at 0.5 are real
ambulances in unlabelled frames (CnPH crash scene, PGcW, D7UW, checked on the frames);
the real ones are an overturned white van (6E7c, Çorum MOBESE) and a bus (QiYl).
**Chosen: v5g, mean >= 0.7 over >= 5 frames, once per place (radius 0.5, 30 min).**
At 0.7 the place rule changes nothing here; it is kept for parked vehicles on other cameras.
Caveat: 0 alarms in 0.88 h only bounds the rate at 3.4/h (rule of three).

**Held-out test2, applied once (23.11.2026, v5g, mean >= 0.7, once per place):** passages
7/9, videos with an ambulance 7/8 (Wilson 53-98 %), ambulance-free videos with an alarm
3/6, false spans 7. Same as v5e (7/8, 3/6; 4 false spans with once per place). The gain on
the development set and Sariyer did not carry over to test2; with 8 + 6 videos the
difference is within noise either way. Reported as is, not re-tuned on test2.
(Reference only, not chosen: v5g at mean 0.5 finds 8/8 with 4/6.) İzmir day and night
with v5g were stopped (23.11, 17:50) and are still to run.

False alarms on İzmir day with the adaptive read rate (light run, v5e): 10 in 2.5 h
(4.0/h) with the frame rule, 9 with once per place; the fixed 5 frames/s run had 3. These
are moving vehicles, so the place rule cannot help; the cause is probably that reading at
12.5 frames/s around a candidate makes 5 frames easy to reach (to be checked on Sariyer,
not on İzmir). The classifier is the real fix, hence v5g.

## Held-out results and an alarm audit (23.11.2026, evening)

Chosen rule (v5g, mean >= 0.7 over >= 5 frames, once per place in 30 min), applied once
with `atlas_training.report` to test3 (35 videos), and v5e with the same rule as the
comparison:

| test3, labels as made | v5g | v5e |
| --- | --- | --- |
| Ambulance videos found | 24/25 | 22/25 |
| Passages found | 40/55 | 33/55 |
| Alarms on a labelled passage | 40/64 | 33/39 |
| Ambulance-free videos with an alarm | 3/10 | 1/10 |

Every alarm span outside the labels was then looked at, for both classifiers (crops along
the track and full frames). The one-second sheets had missed real ambulances: a parked
ambulance (qreCPCILYU0 56–63 s), one behind police officers and one just before a
labelled passage (xUgTyUAh3jg), and one entering a Sakarya KGYS camera at the frame edge
in a video labelled ambulance-free (ZwRF38P5K8c 38–41 s). These became positive; a night
van with a blue beacon right after an ambulance scene (S6AxT_6hQ0w 58–90 s, 12 of v5g's
spans) could not be told apart from a gendarmerie van and became ignore. The audited
copy is `clips/test3/labels_audited.json`, scored with `report --intervals`. test2's
alarms outside the labels were all false (fire-engine detail, advertising truck, a
monument, a red lorry, an overturned car in a caption, a van at night), so its labels
stand. Caveat: only alarms can be audited, so the audit can add found passages but never
missed ones; both label sets are reported.

| test2 + test3 (49 videos), audited | v5g | v5e |
| --- | --- | --- |
| Ambulance videos found | 32/34 (94 %, 81–98) | 29/34 (85 %, 70–94) |
| Passages found | 51/68 | 42/68 |
| Alarms that are real | 51/61 (84 %, 72–91) | 42/45 (93 %, 82–98) |
| Ambulance-free videos with an alarm | 5/15 | 2/15 |

v5g trades precision for recall against v5e on held-out data, while it was better on both
on the development set; the choice stays v5g (chosen before these numbers) and the gap is
reported. What the remaining false alarms are says where to work: white and red vans and
trucks, a fire engine, a red and white monument, lights at night.

İzmir (5 h, never used for training or tuning) had an ambulance after all: it crosses
the square around minutes 119–127 of the day recording. Each alarm was looked at
(`clips/fa/alarm_review.csv`, `report --reviewed`): v5g 3 spans on the real ambulance and
2 false (one event, a small blue light at night) = 0.4 false/h (95 % upper bound 1.26/h);
v5e 1 span on the ambulance (it missed the one at minute 119.5) and 3 false (a white
panel van, a city bus) = 0.6/h. Target 0.125/h; showing it needs ~24 h with no false
alarm.

Next (v5i): more Turkish hard negatives from cameras never used for testing (Şerifali–
Çamlıca and Kadıköy recordings in `clips/hn_new`, mined along tracks and checked by eye
as for v5g), aimed at white vans and trucks; compared with v5g on the development set
and Sarıyer only.

## Classifier v5i and a foundation-model probe (23.11.2026, night)

Hard negatives from new cameras: the two YouTube recordings found as "traffic cameras"
were not (Şerifali: a black screen with a company slide show; Kadıköy: a pier camera with
few vehicles), so mining them gave little. The cameras that do show traffic (İzmir,
Sarıyer) are our test and development cameras and stay out of training.

Public van crops instead: Roboflow `car-lmfdq/van-2s73c` v5 (CC BY 4.0, licence read from
the Roboflow API), 1,281 van boxes cropped as for the classifier and scored with v5g. A
public "van" set can hold ambulances, so the 32 crops with p >= 0.3 were looked at: 4 were
(or could be) ambulances and are excluded (`labeling/van_2s73c_exclude.txt`), 28 are hard
negatives (white and beige Hiace-type vans, a school bus). Most photos are used-car
listings outside Turkey, so this teaches the white-van shape, not Turkish streets.
v5i = v5g + 1,275 van crops in train/other (`atlas_training.addneg`), trained with the v5g
settings: val top-1 0.980 (v5g 0.976) on the same val crops.

Development set and Sarıyer with the chosen rule (place, mean >= 0.7), as v5g was chosen:

| | v5g | v5i |
| --- | --- | --- |
| Passages found (dev) | 14/16 | 13/16 |
| False spans in labelled dev videos | 13 | 4 |
| Ambulance-free dev videos with an alarm | 0/2 | 1/2 |
| Sarıyer alarms in 0.88 h | 0 | 2 |

The new alarms were looked at: two 23–25 px red and blue light blobs at night on
Sarıyer, and a white minibus partly hidden by a red lorry. v5i finds one passage fewer
and alarms on Sarıyer, the two criteria v5g was chosen on, so **v5g stays**; the drop in
false spans on the labelled dev videos is noted but not used to override the criteria
set in advance. One training run per classifier cannot separate the data change from
run-to-run noise (the new false alarm is again a white minibus); a fair comparison of
small data changes needs several seeds per classifier, which is the next step for
v5i-style additions.

Second seed (seed 1, same data and settings, 24.11.2026), chosen rule:

| | Passages | Ambulance-free dev videos with an alarm | False spans (dev) | Sarıyer alarms |
| --- | --- | --- | --- | --- |
| v5g seed 42 (final) | 14/16 | 0/2 | 13 | 0 |
| v5g seed 1 | 14/16 | 1/2 | 15 | 2 |
| v5i seed 42 | 13/16 | 1/2 | 4 | 2 |
| v5i seed 1 | 12/16 | 1/2 | 10 | 7 |

v5i finds fewer passages with both seeds, so the van crops cost recall; that holds. The
false-alarm counts move a lot between seeds of the same data (v5g: 0 or 2 on Sarıyer),
so the zero Sarıyer alarms of the final v5g were partly luck of the training run, and the
development false-alarm numbers carry about ±2 alarms of run-to-run noise. Val top-1:
v5g 0.976 / 0.975, v5i 0.980 / 0.977. v5g stays; future data changes are compared over at
least two seeds.

Kadıköy (5 h, pier camera) gave 14 candidate tracks, all one lit white kiosk with a red
stripe at night; marked `other` in `labeling/trackmine_v4/verdicts.csv` and kept for a
later set (none of them meets the rule, mean 0.07–0.35).

Frozen DINOv2 features (ViT-S/14, Apache 2.0) with a logistic regression on the same
v5g train/val crops (`atlas_training.embedprobe`): val accuracy 0.909 (ambulance recall
0.86), against 0.976 for the fine-tuned YOLO11s-cls. The reported strength of DINOv2
linear probes on general fine-grained sets (Oquab et al., arXiv 2304.07193) does not carry
over to ambulance vs white van with our data. The two make partly different errors (83
and 315, 36 shared), but averaging gives 0.973 and down-weighting v5g by the probe 0.977
with fewer ambulances found; not worth a second large model per crop. Not used.

## Literature pass and two cheap tests (24.11.2026)

Searched five questions tied to our weak points and read the key papers in full:

- **Track-level classification.** Hwang et al., "Cannot See the Forest for the Trees"
  (CVPR 2022, arXiv 2206.02116) classify a whole tracklet with a transformer over its
  RoI tokens (3 encoder layers, 8 heads). Gains are over QDTrack's own per-frame classes
  (TAO val TrackAP50 15.8 → 19.9; YouTube-VIS 34.4 → 37.7 AP), not over plain score
  averaging, and it is trained end to end with the tracker on video tracklets (sparse
  video labels alone: 6.5 TrackAP50). Not usable with our data now; our mean over the
  track stays.
- **Train/test resolution.** Touvron et al., FixRes (NeurIPS 2019, arXiv 1906.06423):
  random-resized-crop training makes objects look larger in training than at test
  (E[r_test/r_train] ≈ 0.80); a ResNet-50 trained at 224 scores 77.0 / 78.0 / 78.4 / 78.3
  at test 224 / 256 / 288 / 320, and 79.1 at 384 after fine-tuning the last layers at the
  test resolution. The paper is about scale statistics, not about pixels lost at the
  source, so it does not promise better small ambulances.
- **Higher input resolution for fine-grained classes.** FungiTastic (arXiv 2408.13632)
  reports about +5 points from 224 to 384 for almost all architectures; Cui et al.
  (arXiv 1806.06193) large gains from 299 to 560 on iNat. Not testable cheaply here: the
  crop sets are stored at 224 px and the mined crops cannot be cut again without
  re-running the mining, so training at 288 or 320 would only upsample the same pixels.
  If the crop pipeline is rebuilt, store crops at 320 px; most ambulances that fail are
  smaller than 224 px anyway, where this gain cannot apply.
- **Low-resolution vehicles.** Tas et al. (Sensors 2022) and arXiv 2309.16390 train on or
  distil to low-resolution vehicle images; together with Peng et al. (1605.06695) this
  supports our degraded low-resolution copies rather than super-resolution.
- **False positives on fixed cameras.** Training-free background embedding (BEM, arXiv
  2604.11714), empty-scene negatives (US8385632), per-region thresholds (US12682481),
  adversarial masks for repeat false detections (US11978247 / US12387480), per-location
  classifiers (Roth et al., CVPR 2009) and tracking-based adaptation (Spinello et al.,
  > 50 % fewer false positives on RGB-D). Our once-per-place memory is the cheapest of
  these; per-camera negatives from empty footage are the next step once a real camera
  (İZUM) is available.
- **Emergency-vehicle literature.** Little CCTV work: most papers use web images, aerial
  views (arXiv 2412.05394) or the beacon alone (ABLDataset, arXiv 2603.05058); sound
  fusion is reported to cut vision false alarms (ETASR 2026, "False Positive Reduction
  in Emergency Vehicle Detection Using a Multimodal Edge-Based System"). The MDPI 2024
  GAN paper (Vehicles 6(3), 53) could not be read (HTTP 403) and is not cited for content.

Two tests that need no training, on the development set and Sarıyer only:

| Test | Result | Decision |
| --- | --- | --- |
| v5g at test size 192 / 224 / 256 / 288 / 320 (val crops, FixRes idea) | accuracy 0.974 / 0.976 / 0.976 / 0.976 / 0.970; ambulance recall falls 0.85 → 0.76 as size grows, other recall rises 0.991 → 0.996 | No gain, it acts like a stricter threshold; stay at 224 |
| Track mean weighted by box width (weight min(1, max(0.25, w/96))) | v5g seed 42: false spans 13 → 12, passages and Sarıyer unchanged; seed 1: Sarıyer 2 → 1 | Within run-to-run noise; not used |

## A benchmark that can tell classifiers apart (24.11.2026)

Counting alarms on the development videos cannot separate two classifiers: two training
runs of the same data differ by ±2 alarms. So a fixed set of hard tracks was built from
the development videos and Sarıyer only (test sets untouched): `atlas_training.trackmine`
with v5g at candidate p >= 0.1 proposed 205 tracks; each was looked at and marked
ambulance (28), other (123: white vans and minibuses, buses, a fire engine, a police car,
billboards, a road sign, night tail lights) or skip (54: the track jumped between
vehicles, overturned ambulances, unclear night scenes). Data:
`labeling/bench_v1{,_sariyer}/verdicts.csv`. `atlas_training.bench` scores every crop
offline (40 s for five classifiers, at most 60 crops per track spread evenly), applies
the track rule (mean >= 0.7 over >= 5 crops) and bootstraps over tracks, paired between
classifiers (* = 95 % interval of the difference excludes 0):

| Classifier | False-alarm tracks (of 123) | Ambulance tracks raised (of 28) | FP crops | Recall crops |
| --- | --- | --- | --- | --- |
| v5e | 0.024 | 0.571 | 0.145 | 0.730 |
| v5g (final) | 0.008 | 0.786 | 0.051 | 0.942 |
| v5g seed 1 | 0.016 | 0.786 | 0.043 | 0.916 |
| v5i | 0.008 | 0.786 | 0.058 | 0.916 |
| v5i seed 1 | 0.041 | 0.714 | 0.055 | 0.926 |

- v5g vs v5e: +0.214 ambulance tracks [+0.07, +0.36] *, FP crops −0.095 [−0.16, −0.04] *,
  recall crops +0.21 *. The mined crops (v5g) are a real gain, now shown with an interval.
- v5g vs its second seed and vs v5i: no clear difference on any measure except v5i's
  recall crops (−0.027 [−0.055, −0.006] *), which agrees with the development videos
  (fewer passages). v5i seed 1 raises more false-alarm tracks than v5g (+0.033 *).
- So the earlier conclusions hold, now with intervals: v5g stays; v5i is not better.

Limits: the tracks were proposed by v5g, so the set holds more of v5g's mistakes than of
another classifier's (it is harder on v5g, and v5g's ambulance tracks all reached
p >= 0.1 somewhere); 28 ambulance tracks give intervals of about ±0.15 on track recall.
Crop-level rates are the more sensitive measure for small changes. To grow the set,
mine new development footage with several classifiers and keep the union.

## Classifier v5j and a second seed (25.11.2026)

New data: 25 Turkish videos (`clips/tr4`, leakage-checked against test2/test3 with
`atlas_training.leakcheck`; five dropped), mined with v5g and looked at track by track
(`labeling/trackmine_v5`: 35 ambulance, 34 other, 32 skip). v5j = v5g + 510 ambulance
crops + 64 hard-negative crops. The bench was grown with seven more development videos
(`clips/dev2`, bench only; `labeling/bench_v2`) to 48 ambulance / 188 other tracks.

| Classifier | False-alarm tracks | Ambulance tracks raised | FP crops | Recall crops |
| --- | --- | --- | --- | --- |
| v5g (final) | 0.011 | 0.750 | 0.052 | 0.919 |
| v5g seed 1 | 0.011 | 0.750 | 0.047 | 0.886 |
| v5j | 0.016 | 0.688 | 0.058 | 0.834 |
| v5j seed 1 | 0.011 | 0.729 | 0.048 | 0.884 |

- The first v5j run lost recall crops against v5g (−0.086 [−0.18, −0.02] *), mostly on
  night ambulances. The second seed does not: v5j seed 1 vs v5g seed 1 is −0.001
  [−0.04, +0.04]. Against v5g (seed 42) both v5j runs are lower, but so is v5g's own
  second seed (−0.034 [−0.065, 0.000]); the bench tracks were proposed by v5g seed 42,
  which favours that one run. So the first v5j result was mostly run-to-run noise.
- v5j is not adopted: over two seeds it is not better than v5g on any measure, and the
  new data did not buy a measurable gain. v5g stays.
- The night red-light negatives first suspected as the cause are only one or two of the
  64. Looking at all 64 found one real label error: a crop where the tracker had jumped
  from a grey van to an ambulance (track `p2Sx_COlRrI/86`), and two crops of a bus track
  with a whole ambulance in frame (`/157`). Track-level verdicts can hide such frames;
  the three crops are now in `trackmine_v5/check/exclude.txt`. v5g scores do not single
  them out (mined negatives score high by design), so the check stays by eye.
- A planned ablation (v5k: v5g + the new positives only, `datasets/crops_v5k`) is not
  trained, since there is no drop left to explain.

Lesson for the method: one seed per data change is not enough even on the bench; compare
the mean of two seeds, and prefer a bench proposed by several classifiers.

## Short, confident tracks (25.11.2026, night)

Of the 9 bench ambulance tracks v5g misses with the running-mean rule, 6 have only 3–4
crops, all at p ≈ 1.0 (brief glimpses, or pieces of a track the tracker broke after a
glare or an occlusion; seen in a night traffic demo as an ambulance turning orange each
time it got a new track number). A second path was added to the place rule
(`alarm_once_per_place(..., short_frames, short_threshold)`, off by default): a track
with fewer than 5 frames also alarms once it has `short_frames` frames at mean >=
`short_threshold`. A sliding-window mean was tried as well and was worse at every
setting (window 5 raises 13 of 188 other tracks at 0.7; windows >= 8 lose recall, as most
tracks are short).

| Rule (v5g) | Bench ambulance tracks | Bench other tracks | Dev passages | Dev false spans | Sarıyer spans |
| --- | --- | --- | --- | --- | --- |
| mean >= 0.7, >= 5 frames (chosen) | 39/48 | 4/188 | 14/16 | 13 | 0 |
| + >= 3 frames at >= 0.9 | 45/48 | 4/188 | 14/16 | 14 | 0 |
| + >= 2 frames at >= 0.8 | 45/48 | 5/188 | 14/16 | 17 | 0 |

Not adopted. The gain is per track, not per passage: on the development videos every
passage the short tracks belong to is already found by a longer track of the same
ambulance, while each setting adds 0–4 false spans. The bench counts tracks, so it
overstates rule changes that only help broken tracks; rule changes are judged on
passages. (A first run of this comparison merged two seeds' runs of the same videos,
which overwrite each other by video name, and used no shot cuts; it was discarded.)

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
