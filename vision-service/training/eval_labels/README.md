# Held-out labels (S4PT-52)

Time-interval labels of the held-out test videos and the eye check of the camera alarms,
used for the numbers in `RESULTS.md` (Sections 6 and 7). They contain video IDs, seconds
and notes only; the videos themselves are kept outside Git and are never published.

| File | Content |
| --- | --- |
| `test2/labels.json` | test2 (14 videos), labels as made on 21.11.2026; no change after the audit |
| `test3/labels.json` | test3 (35 videos), labels as made on 23.11.2026 |
| `test3/labels_audited.json` | test3 after every alarm outside the labels was looked at: missed ambulances added as positive, one unclear night van as ignored |
| `fa/alarm_review.csv` | Every alarm on the 5 h İzmir camera recording, checked by eye (`atlas_training.report --reviewed`) |

The rule of each label set is written in its `_rule` field. Compare `test3/labels.json`
and `test3/labels_audited.json` to see what was relabelled.
