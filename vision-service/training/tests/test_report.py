from atlas_training.report import summarise


def test_summarise_counts_alarm_precision_and_false_alarm_rate():
    r = {"found": 7, "events": 9, "false_spans": 7, "pos_found": 7, "pos_clips": 8,
         "neg_alarm": 3, "neg_clips": 6, "fa_spans": 0, "fa_hours": 0.88, "per_run": {}}
    s = summarise(r)
    assert s["recall_videos"][:2] == [7, 8]
    assert s["precision_alarms"][:2] == [7, 14]
    assert 0.2 < s["precision_alarms"][2] < s["precision_alarms"][3] < 0.8
    assert s["false_alarms"][0] == 0 and abs(s["false_alarms"][3] - 3 / 0.88) < 0.01


def test_summarise_without_camera_hours_has_no_false_alarm_rate():
    r = {"found": 1, "events": 1, "false_spans": 0, "pos_found": 1, "pos_clips": 1,
         "neg_alarm": 0, "neg_clips": 0, "fa_spans": 0, "fa_hours": 0.0, "per_run": {}}
    assert "false_alarms" not in summarise(r)


def test_reviewed_false_alarms_counts_unchecked_spans_as_false(tmp_path):
    from atlas_training.report import reviewed_false_alarms

    # Track 1 meets the chosen rule from the 5th frame on; one alarm span starting at t=4.
    run = {"times": [float(t) for t in range(10)],
           "frames": [{"1": 0.9}] * 10,
           "boxes": [{"1": [100.0, 100.0, 50.0]}] * 10}
    review = tmp_path / "review.csv"
    review.write_text("run,start_s,verdict,note\ncam,4.0,ambulance,x\n", encoding="utf-8")
    r = reviewed_false_alarms({"cam": run}, review)
    assert (r["true"], r["false"], r["unchecked"]) == (1, 0, 0)
    review.write_text("run,start_s,verdict,note\n", encoding="utf-8")
    r = reviewed_false_alarms({"cam": run}, review)
    assert (r["true"], r["false"], r["unchecked"]) == (0, 1, 1)
