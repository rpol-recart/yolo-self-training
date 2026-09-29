import numpy as np
import pytest

from selftrain.boxes import fuse_tta, iou_matrix, nms_per_class, unflip_x, xyxy_to_yolo, yolo_to_xyxy
from selftrain.calibrate import calibrate, match_predictions, threshold_for_precision
from selftrain.config import DEFAULTS, deep_merge, validate
from selftrain.decide import accept_round, should_stop
from selftrain.filter import ImageDecision, decide, review_queue, split_by_confidence, temporal_consistency
from selftrain.merge import select_pseudo
from selftrain.split import assign_groups, check_no_leakage

CAL = {"iou": 0.5, "target_precision": 0.9, "low_precision": 0.5, "min_support": 1,
       "fallback_high": 0.6, "fallback_low": 0.25}
THR = {0: {"t_high": 0.8, "t_low": 0.3}, 1: {"t_high": 0.6, "t_low": 0.2}}


# ---------- boxes ----------

def test_iou_basic():
    a = np.array([[0, 0, 0.5, 0.5]])
    b = np.array([[0, 0, 0.5, 0.5], [0.25, 0, 0.75, 0.5], [0.6, 0.6, 1, 1]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-9)


def test_yolo_roundtrip():
    boxes = np.array([[0.1, 0.2, 0.3, 0.6]])
    rows = xyxy_to_yolo(boxes, np.array([1]))
    np.testing.assert_allclose(yolo_to_xyxy(rows), [[1, 0.1, 0.2, 0.3, 0.6]], atol=1e-9)


def test_nms_is_per_class():
    d = np.array([[0, 0, 1, 1, 0.9, 0], [0, 0, 1, 1, 0.8, 0], [0, 0, 1, 1, 0.7, 1]])
    out = nms_per_class(d, 0.5)
    assert len(out) == 2 and set(out[:, 5]) == {0, 1}


# ---------- split ----------

def test_split_is_by_group_deterministic_and_complete():
    groups = [f"g{i}" for i in range(20)]
    a = assign_groups(groups, 0.15, 0.15, seed=1)
    assert a == assign_groups(groups, 0.15, 0.15, seed=1)
    assert set(a) == set(groups)
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "val", "test")}
    assert counts == {"train": 14, "val": 3, "test": 3}


def test_split_too_few_groups():
    with pytest.raises(ValueError):
        assign_groups(["a", "b"], 0.5, 0.5, seed=0)


def test_leakage_guard():
    check_no_leakage({"x1": "a", "x2": "a"}, {"a": "train"})


# ---------- calibration ----------

def test_threshold_lowest_cut_meeting_target():
    # conf-desc: TP TP TP FP TP FP FP ; precision after each: 1,1,1,.75,.8,.67,.57
    scored = [(0.95, True), (0.9, True), (0.85, True), (0.7, False), (0.6, True), (0.4, False), (0.3, False)]
    t, p, tp = threshold_for_precision(scored, 0.9)
    assert t == 0.85 and p == 1.0 and tp == 3
    t, p, _ = threshold_for_precision(scored, 0.75)
    assert t == 0.6 and p == pytest.approx(0.8)


def test_threshold_unreachable():
    assert threshold_for_precision([(0.9, False)], 0.9)[0] is None


def test_matching_one_gt_one_tp():
    preds = {"a": np.array([[0, 0, 0.5, 0.5, 0.9, 0], [0, 0, 0.5, 0.5, 0.8, 0]])}
    gts = {"a": np.array([[0, 0, 0, 0.5, 0.5]])}
    scored, n_gt = match_predictions(preds, gts, 0.5)
    assert n_gt == {0: 1}
    assert scored[0] == [(0.9, True), (0.8, False)]  # duplicate is a false positive


def test_calibrate_fallback_for_class_without_support():
    preds = {"a": np.array([[0, 0, 0.5, 0.5, 0.9, 0]])}
    gts = {"a": np.array([[0, 0, 0, 0.5, 0.5]])}
    th = calibrate(preds, gts, 2, CAL, conf_floor=0.05)
    assert th[0]["fallback"] is False and th[0]["t_high"] == 0.9
    assert th[1]["fallback"] is True and th[1]["t_high"] == 0.6
    assert th[0]["t_low"] <= th[0]["t_high"]


# ---------- filtering ----------

def test_split_by_confidence_three_zones_and_min_area():
    d = np.array([
        [0, 0, 0.5, 0.5, 0.9, 0],      # confident
        [0, 0, 0.5, 0.5, 0.5, 0],      # grey zone
        [0, 0, 0.5, 0.5, 0.1, 0],      # background
        [0, 0, 0.01, 0.01, 0.99, 0],   # too small
    ])
    c, u = split_by_confidence(d, THR, min_area=0.001)
    assert len(c) == 1 and len(u) == 1


def test_uncertain_policy():
    preds = {"img": np.array([[0, 0, 0.5, 0.5, 0.9, 0], [0.5, 0.5, 1, 1, 0.5, 0]])}
    fcfg = {"min_box_area": 0.0, "uncertain_policy": "drop_image", "temporal": {"enabled": False}}
    assert decide(preds, {}, THR, fcfg)[0].status == "uncertain"
    fcfg["uncertain_policy"] = "keep_confident"
    assert decide(preds, {}, THR, fcfg)[0].status == "pseudo"


def test_background_status():
    preds = {"img": np.zeros((0, 6))}
    fcfg = {"min_box_area": 0.0, "uncertain_policy": "drop_image", "temporal": {"enabled": False}}
    assert decide(preds, {}, THR, fcfg)[0].status == "background"


def test_temporal_removes_one_frame_flash():
    box = np.array([[0.1, 0.1, 0.3, 0.3, 0.9, 0]])
    flash = np.array([[0.6, 0.6, 0.8, 0.8, 0.9, 1]])
    recs = {f"v/f_{i}.jpg": box.copy() for i in range(5)}
    recs["v/f_2.jpg"] = np.concatenate([box, flash])
    kept, removed = temporal_consistency(recs, r"(?P<seq>.+)/f_(?P<idx>\d+)\.jpg", window=1, iou_thr=0.3, min_support=1)
    assert len(kept["v/f_2.jpg"]) == 1 and removed["v/f_2.jpg"][0, 5] == 1
    assert all(len(kept[f"v/f_{i}.jpg"]) == 1 for i in range(5))


def test_temporal_ignores_neighbours_across_gap():
    box = np.array([[0.1, 0.1, 0.3, 0.3, 0.9, 0]])
    recs = {"v/f_0.jpg": box, "v/f_1.jpg": box, "v/f_100.jpg": box}
    kept, _ = temporal_consistency(recs, r"(?P<seq>.+)/f_(?P<idx>\d+)\.jpg", window=1, iou_thr=0.3, min_support=1)
    assert len(kept["v/f_100.jpg"]) == 0 and len(kept["v/f_0.jpg"]) == 1


def test_review_queue_round_robin_over_groups():
    ds = [ImageDecision(path=f"{g}{i}", group=g, uncertain=np.ones((1, 6)), uncertainty=1.0 - i / 10)
          for g in ("a", "b") for i in range(5)]
    q = review_queue(ds, 4)
    assert sorted(d.group for d in q) == ["a", "a", "b", "b"]


# ---------- merge ----------

def test_rare_class_first_and_group_cap():
    def img(path, group, cls):
        return ImageDecision(path=path, group=group, status="pseudo",
                             confident=np.array([[0, 0, 0.5, 0.5, 0.9, cls]]))
    cands = [img(f"c{i}", "g1", 0) for i in range(10)] + [img("r1", "g2", 1), img("r2", "g1", 1)]
    gt_counts = np.array([100, 2])
    picked = select_pseudo(cands, gt_counts, max_images=3, max_per_group=None, seed=0)
    assert {p.path for p in picked[:2]} == {"r1", "r2"}
    picked = select_pseudo(cands, gt_counts, max_images=10, max_per_group=2, seed=0)
    assert sum(p.group == "g1" for p in picked) == 2


# ---------- decision rules ----------

def _m(v, groups=None):
    return {"overall": {"map50_95": v}, "groups": {g: {"map50_95": x} for g, x in (groups or {}).items()}}


ACC = {"metric": "map50_95", "min_delta": 0.01, "group_tolerance": 0.02}


def test_accept_requires_delta():
    assert accept_round(_m(0.505), _m(0.5), ACC)[0] is False
    assert accept_round(_m(0.52), _m(0.5), ACC)[0] is True


def test_accept_rejects_group_regression_even_if_mean_improves():
    ok, reasons = accept_round(_m(0.6, {"a": 0.9, "b": 0.3}), _m(0.5, {"a": 0.5, "b": 0.5}), ACC)
    assert ok is False and any("group b" in r for r in reasons)


def test_stop_rules():
    lc = {"max_rounds": 4, "patience": 2}
    assert should_stop([{"accepted": True}], lc)[0] is False
    assert should_stop([{"accepted": True}, {"accepted": False}, {"accepted": False}], lc)[0] is True
    assert should_stop([{"accepted": True}] * 4, lc)[0] is True


# ---------- config ----------

def test_config_merge_and_validation():
    cfg = deep_merge(DEFAULTS, {"names": ["x"], "data": {"gt_images": "d"}, "accept": {"min_delta": 0.1}})
    assert cfg["accept"]["min_delta"] == 0.1 and cfg["accept"]["metric"] == "map50_95"
    validate(cfg)
    with pytest.raises(ValueError):
        validate(deep_merge(cfg, {"filter": {"uncertain_policy": "maybe"}}))


def test_t_low_extends_to_floor_when_all_val_predictions_are_correct():
    preds = {"a": np.array([[0, 0, 0.5, 0.5, 0.95, 0]])}
    gts = {"a": np.array([[0, 0, 0, 0.5, 0.5]])}
    th = calibrate(preds, gts, 1, CAL, conf_floor=0.05)
    assert th[0]["t_high"] == 0.95 and th[0]["t_low"] == 0.05


# ---------- TTA ----------

def test_unflip_x_roundtrip():
    d = np.array([[0.1, 0.2, 0.3, 0.4, 0.9, 1]])
    np.testing.assert_allclose(unflip_x(d), [[0.7, 0.2, 0.9, 0.4, 0.9, 1]])
    np.testing.assert_allclose(unflip_x(unflip_x(d)), d)


def test_fuse_tta_agreement_keeps_confidence_single_view_is_diluted():
    obj = np.array([[0.1, 0.1, 0.3, 0.3, 0.8, 0]])
    ghost = np.array([[0.6, 0.6, 0.7, 0.7, 0.8, 0]])
    views = [np.r_[obj, ghost], obj + [0.005, 0, 0.005, 0, 0, 0], obj]
    fused = fuse_tta(views, 0.55)
    assert len(fused) == 2
    assert fused[0, 4] == pytest.approx(0.8)          # seen in all 3 views
    assert fused[1, 4] == pytest.approx(0.8 / 3)      # seen in 1 of 3 views
    np.testing.assert_allclose(fused[0, :4], [0.101667, 0.1, 0.301667, 0.3], atol=1e-4)


def test_fuse_tta_does_not_merge_classes():
    a = np.array([[0.1, 0.1, 0.3, 0.3, 0.8, 0]])
    b = np.array([[0.1, 0.1, 0.3, 0.3, 0.8, 1]])
    assert len(fuse_tta([a, b])) == 2
