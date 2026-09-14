"""Task 8: score the whole system on REAL smear photos labelled by experts (BBBC041).

Download the sample first:  python -m scripts.fetch_real_smears
Run:                        python -m src.real_eval
"""
import json

import numpy as np
import torch
from PIL import Image

import src.smear as smear_module
from src import config
from src.model import load_trained_model

REAL_DIR = config.PROJECT_DIR / "real_smears"
RESULTS_DIR = config.OUTPUTS_DIR / "real"
INFECTED = {"ring", "trophozoite", "schizont", "gametocyte"}
MATCH_IOU = 0.5  # a detection "finds" an expert box if they overlap at least this much


def expert_boxes(entry):
    """Convert the dataset's labels into (x, y, w, h, kind) with kind = healthy/infected/other."""
    boxes = []
    for obj in entry["objects"]:
        top_left, bottom_right = obj["bounding_box"]["minimum"], obj["bounding_box"]["maximum"]
        x, y = top_left["c"], top_left["r"]
        w, h = bottom_right["c"] - x, bottom_right["r"] - y
        if obj["category"] == "red blood cell":
            kind = "healthy"
        elif obj["category"] in INFECTED:
            kind = "infected"
        else:
            kind = "other"  # white blood cells and cells the experts marked "difficult"
        boxes.append((x, y, w, h, kind))
    return boxes


def iou(a, b):
    """Intersection over union: overlap area / combined area. 1 = identical boxes."""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    overlap_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    overlap = overlap_w * overlap_h
    return overlap / (aw * ah + bw * bh - overlap) if overlap else 0.0


def match(detections, experts):
    """Pair each expert box with at most one detection, best overlaps first."""
    pairs = sorted(((iou(d["box"], e), di, ei)
                    for di, d in enumerate(detections) for ei, e in enumerate(experts)),
                   reverse=True)
    used_d, used_e, matches = set(), set(), []
    for score, di, ei in pairs:
        if score < MATCH_IOU:
            break
        if di not in used_d and ei not in used_e:
            used_d.add(di)
            used_e.add(ei)
            matches.append((di, ei))
    return matches


def evaluate(entries, model, device, save_images=False):
    """Run the full pipeline on every photo and compare with the experts."""
    totals = {"expert_cells": 0, "detections": 0, "found": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0}
    parasitemia_errors, rows = [], []

    for entry in entries:
        name = entry["image"]["pathname"].split("/")[-1]
        image = np.array(Image.open(REAL_DIR / "images" / name).convert("RGB"))
        results, summary, annotated = smear_module.analyze_smear(image, model, device)
        experts = expert_boxes(entry)
        red_cells = [e for e in experts if e[4] != "other"]

        # Detections that landed on a white blood cell or "difficult" cell are
        # neither right nor wrong, so they are left out of the precision count
        matches = match(results, experts)
        on_other = sum(experts[ei][4] == "other" for _, ei in matches)
        found_red = [(di, ei) for di, ei in matches if experts[ei][4] != "other"]
        totals["expert_cells"] += len(red_cells)
        totals["detections"] += len(results) - on_other
        totals["found"] += len(found_red)

        for di, ei in found_red:
            said_infected = results[di]["label"] == 1
            truly_infected = experts[ei][4] == "infected"
            key = {(True, True): "tp", (False, True): "fn",
                   (True, False): "fp", (False, False): "tn"}[(said_infected, truly_infected)]
            totals[key] += 1

        true_parasitemia = sum(e[4] == "infected" for e in red_cells) / max(1, len(red_cells))
        parasitemia_errors.append(abs(summary["parasitemia"] - true_parasitemia))
        rows.append((name, len(red_cells), summary["total_cells"], len(found_red),
                     true_parasitemia, summary["parasitemia"]))
        if save_images:
            Image.fromarray(annotated).save(RESULTS_DIR / name.replace(".jpg", ".png"))

    t = totals
    report = {
        "detection_recall": t["found"] / t["expert_cells"],           # expert cells we found
        "detection_precision": t["found"] / max(1, t["detections"]),  # our boxes that were real cells
        "sensitivity": t["tp"] / max(1, t["tp"] + t["fn"]),
        "specificity": t["tn"] / max(1, t["tn"] + t["fp"]),
        "mean_parasitemia_error": float(np.mean(parasitemia_errors)),
        **t,
    }
    return report, rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(device)
    entries = json.loads((REAL_DIR / "labels.json").read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compare classification with and without colour matching (detection is the same)
    reports = {}
    for normalize in (False, True):
        smear_module.STAIN_NORMALIZE = normalize
        reports[normalize], rows = evaluate(entries, model, device, save_images=normalize)

    print(f"{'image':<14}{'expert':>7}{'found':>7}{'matched':>8}{'true %':>8}{'ours %':>8}")
    for name, n_exp, n_det, n_match, true_p, our_p in rows:
        print(f"{name[:12]:<14}{n_exp:>7}{n_det:>7}{n_match:>8}{true_p:>8.1%}{our_p:>8.1%}")

    print(f"\n{'metric':<24}{'no colour match':>16}{'colour match':>14}")
    for key in reports[True]:
        before, after = reports[False][key], reports[True][key]
        fmt = (lambda v: f"{v:.1%}") if isinstance(after, float) else str
        print(f"{key:<24}{fmt(before):>16}{fmt(after):>14}")

    output = {"images": len(entries), "without_colour_matching": reports[False],
              "with_colour_matching": reports[True]}
    (RESULTS_DIR / "real_results.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
