# Ambulance detection results (S4PT-52) — DRAFT

Issue: S4PT-52 · Sprint: S4PT Sprint 4 · Author: Eren Türkan (22070006020)
Related: FR-06, UC-2.1, NFR-04, NFR-05 (false alerts), NFR-02 (frame rate) · Code: branch `task/S4PT-52`,
`vision-service/training/` (README.md, RESEARCH.md)

## 1. Purpose
Test a pre-trained ambulance detector, fine-tune with transfer learning, evaluate on the
same test set, and report against NFR-04 (precision and recall >= 90 % at clip level)
with the remaining gap.

## 2. Data
- Public datasets (Roboflow Universe, CC BY 4.0): emergency_vehicles_detection,
  special_vehicles_small_ds, emergency_vehicles_night (licences checked 12.11 and 19.11).
- Own labels (Label Studio, local): Turkish traffic and news videos, standard YouTube
  licence, local testing only, never published. Labelling rule: only the physical
  vehicle, also partly hidden or cut off; captions and graphics are never labels.
- Splits by video, so no test video is seen in training. Test sets:
  - frame-level test split (1,099 images, 322 ambulance boxes; 9 Turkish videos),
  - development set: the same 9 Turkish videos, used to choose settings,
  - held-out clip tests: test2 (14 videos) and test3 (35 videos), Turkish videos never
    used for training or tuning (34 with an ambulance, 15 without), labelled as time
    intervals,
  - false-alarm development: 0.88 h of a Sarıyer camera, used to choose the rule,
  - false alarms: 5 h of the İzmir square camera (day and night), never used for
    training or tuning.
- Out of scope: grey patient-transport ambulances (2026 livery), beacon state (FR-07).

## 3. Models compared (frame level, same test set)
| Model | Precision | Recall | AP@0.5 |
| --- | --- | --- | --- |
| Pre-trained (SmartTrafficAI) | 0.44 | 0.59 | 0.49 |
| v1 – v4 fine-tuned YOLO11s | see RESEARCH.md | | |
| v3 (best one-stage) | 0.82 | 0.53 | 0.62 |
| Two-stage, classifier v5e (conf 0.25) | 0.66 | 0.59 | 0.59 |
| Two-stage, classifier v5g (final, conf 0.25) | 0.69 | 0.64 | 0.64 |
| Two-stage v5g at best-F1 threshold | 0.86 | 0.60 | 0.64 |

Turkish test videos only (159 boxes): v3 P 0.78 / R 0.33; two-stage v5e P 0.69 / R 0.54 /
AP 0.58; two-stage v5g P 0.77 / R 0.62 / AP 0.67. Frame-level numbers count every box in
every frame; NFR-04 is judged per clip (Section 6), where one recognised frame run is
enough.

## 4. Key finding
A generic vehicle detector finds 129 of 148 Turkish test ambulances as vehicles; the
one-stage ambulance detector finds 52. The miss is recognition, not detection, so the
final system classifies tracked vehicle crops and decides per track.

## 5. Final system
COCO YOLO11s vehicles (960 px) → ByteTrack → crop classifier v5g (YOLO11s-cls, 224 px,
trained with track-mined crops checked by eye) → track decision: mean p(ambulance) >= 0.7
over >= 5 frames, at most one alarm per place in 30 min (a parked ambulance does not alarm
again) → one alarm per track. Frames read at 5/s, 12.5/s for 3 s while a candidate is in
view (NFR-02).

The rule was chosen only on the development set (9 videos, 16 passages) and on 0.88 h of
a Sarıyer camera without ambulances: 14/16 passages found, 0 false alarms (v5e with the
earlier rule: 13/16 and 44 alarms). It was then applied once, unchanged, to the held-out
sets below. Rejected after measurement: a motion requirement (recall 13→3 of 16),
synthetic night copies (worse at night), requiring a visible beacon light (6–12 of 16).

## 6. NFR-04 at clip level (held-out, 49 videos)
Held-out sets: test2 (14 videos) and test3 (35 videos; 25 with an ambulance, 7 of them at
night), Turkish videos never used for training or for choosing the rule, labelled as
time intervals from one-second contact sheets. The chosen rule was applied once
(`atlas_training.report`).

Every alarm outside the labelled intervals was then looked at, for both classifiers. In
test3 the sheets had missed real ambulances (a parked ambulance, one behind police
officers, one entering a KGYS camera at the edge of a video labelled ambulance-free);
those were added as positive and an unclear night van as ignored
(`clips/test3/labels_audited.json`). test2 needed no change: all its alarms outside the
labels are false (a fire engine, an advertising truck, a monument, a red lorry). Only
alarms can be audited this way, so the audit can only add passages a model found; both
label sets are reported.

| v5g (final), test2 + test3 | Labels as made | Audited labels | 95 % interval (audited) | Target |
| --- | --- | --- | --- | --- |
| Videos with an ambulance found (recall) | 31/33 (94 %) | 32/34 (94 %) | 81–98 % | >= 90 % |
| Ambulance passages found | 47/64 (73 %) | 51/68 (75 %) | | |
| Alarms that are real (precision) | 47/78 (60 %) | 51/61 (84 %) | 72–91 % | >= 90 % |
| Ambulance-free videos with an alarm | 6/16 | 5/15 | | |

Previous classifier v5e with the same rule, audited labels: recall 29/34 (85 %, 70–94 %),
passages 42/68 (62 %), precision 42/45 (93 %, 82–98 %), ambulance-free videos with an
alarm 2/15. v5g finds more ambulances (3 more videos, 9 more passages), v5e raises fewer
false alarms. The rule was chosen on the development set, where v5g was better on both
counts, so v5g stays the final classifier; the precision gap is reported, not tuned away
on the test sets. test3 alone (audited): v5g recall 25/26, precision 44/47; v5e 23/26,
36/36.

## 7. False alarms (NFR-05)
Chosen rule, applied once to 5 h of the İzmir square camera (2.5 h day, 2.5 h night),
which was never used for training or for choosing the rule:

Every alarm was looked at (`clips/fa/alarm_review.csv`, `atlas_training.report --reviewed`).
The recording was assumed to be ambulance-free, but an ambulance crosses the square in
the daytime part (around minutes 119–127), so some alarms are correct detections:

| Classifier | Alarm spans | On a real ambulance | False (events) | False per hour | 95 % upper bound |
| --- | --- | --- | --- | --- | --- |
| v5g (final) | 5 | 3 | 2 (1: a small blue light at night, flickering) | 0.40 | 1.26 |
| v5e (previous), same rule | 4 | 1 | 3 (2: a white panel van, a city bus) | 0.60 | 1.55 |

Target: <= 1 per camera per 8 h (0.125/h). v5g is about 3 times above it (0.2/h if the
two spans of the same light count as one event); 5 h is too short to show that the
target is met, which would need about 24 h with no false alarm (rule of three). v5g also
found the ambulance at minute 119.5 that v5e missed. Most of the drop from the earlier
rule (v5e, mean >= 0.5: 10 alarms in the 2.5 h of daytime alone) comes from the higher
threshold and the place memory.

## 8. Gap to the targets and limits
- NFR-04 recall: met as a point estimate (94 % of ambulance videos), but the 95 % lower
  bound is 81 %; about 100 held-out ambulance videos would be needed to show >= 90 %.
- NFR-04 precision: 84 % (72–91 %), below 90 %. False alarms are white or red vans and
  trucks, a fire engine, a monument, lights at night.
- NFR-05: 0.4 false alarms per hour on the İzmir camera (target 0.125/h); 5 h is too
  short to show the target.
- Small ambulances (< ~64 px wide) are rarely recognised (2 of 24 on the test frames);
  the İZUM stream (~5 fps, 240x180) is in this range (S4PT-50 measurement).
- Night scenes with strong glare: the vehicle detector sometimes gives no box.
- Overturned or crashed ambulances: reported separately.
- Test labels were checked only where a model alarmed (Section 6), so missed passages
  outside the labels stay unknown; intervals are wide with 49 videos.
- FR-07 (beacon state) not covered; method notes in RESEARCH.md.

## 9. Licences
Datasets CC BY 4.0; YouTube clips local only; Ultralytics models AGPL-3.0 (same as the
service).
