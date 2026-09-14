"""Task 4: analyse a whole blood smear: find each cell, classify it, report parasitemia.

Run:  python -m src.smear
"""
import random

import cv2
import numpy as np
import torch
from PIL import Image

from src import config
from src.dataset import eval_transform, list_images, split_samples
from src.model import load_trained_model

BACKGROUND_COLOR = (245, 238, 243)  # pale pink, like an empty area of a stained slide
GAP = 12                # pixels kept between cells when building a synthetic smear

# Cell detection settings. Sizes are relative to a "typical cell" in the image,
# so the same code works for zoomed-in and zoomed-out photos.
MIN_IMAGE_SIDE = 64       # anything smaller is too tiny to be a smear
BACKGROUND_TOLERANCE = 8  # pixels this close in brightness to the slide count as slide
MIN_CONTRAST = 8          # cells must differ from the slide by this much on average
CONTRAST_CAP = 40         # differences above this are treated as equal (see foreground_mask)
MIN_CELL_AREA = 80        # the smallest blob (in pixels) worth looking at
MAX_AREA_FRACTION = 0.9   # a blob covering almost the whole image is a lighting problem
SPECK_FRACTION = 0.25     # blobs under 25% of a typical cell's area are debris
CLUMP_FACTOR = 1.6        # blobs over 1.6x a typical cell's area are probably touching cells
DENSE_FACTOR = 3.0        # pieces still over 3x a typical cell after splitting are skipped

# Average colour and spread of cells in the TRAINING images, in LAB colour space
# (measured once on 2,000 training images). Used by match_training_colours.
STAIN_NORMALIZE = True
TRAINING_LAB_MEAN = np.array([170.27, 143.83, 128.57])
TRAINING_LAB_STD = np.array([15.31, 9.74, 10.20])

# Colours are (Red, Green, Blue)
GREEN, RED, ORANGE, GREY = (0, 170, 0), (220, 0, 0), (255, 140, 0), (110, 110, 110)


def make_synthetic_smear(samples, infected_fraction, n_cells=24, size=(1400, 900), seed=None):
    """Paste single-cell images onto a blank slide to imitate a real smear.

    Returns the image and the true (x, y, w, h, label) of every cell placed,
    so we can check the analyser's answers against the truth.
    """
    rng = random.Random(seed)
    infected = [s for s in samples if s[1] == 1]
    healthy = [s for s in samples if s[1] == 0]
    n_infected = round(n_cells * infected_fraction)
    chosen = rng.sample(infected, n_infected) + rng.sample(healthy, n_cells - n_infected)
    rng.shuffle(chosen)

    width, height = size
    canvas = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)
    placed = []

    for path, label in chosen:
        cell = np.array(Image.open(path).convert("RGB"))
        h, w = cell.shape[:2]
        for _attempt in range(50):  # try random spots until one doesn't overlap
            x, y = rng.randint(0, width - w), rng.randint(0, height - h)
            overlaps = any(
                x < px + pw + GAP and px < x + w + GAP and y < py + ph + GAP and py < y + h + GAP
                for px, py, pw, ph, _ in placed
            )
            if not overlaps:
                is_cell = cell.sum(axis=2) > 30  # True on the cell, False on its black border
                canvas[y:y + h, x:x + w][is_cell] = cell[is_cell]
                placed.append((x, y, w, h, label))
                break
    return canvas, placed


def estimate_background(image):
    """Guess what the empty slide looks like everywhere, including uneven lighting.

    1. Split the image into a 6x6 grid. Empty slide is very evenly coloured, so in
       each tile the most common brightness is probably the slide.
    2. Fit a flat surface through those guesses, ignoring tiles that disagree
       (e.g. tiles completely covered by cells). This follows lighting that
       changes from one side of the image to the other.
    3. Take the pixels close to that surface as slide, fit a smooth curved surface
       through them, and repeat a few times.
    Unlike a local average, this still works when cells are packed so tightly that
    they cover most of the image. Everything runs on a 1/8-size copy, because the
    surface is smooth anyway.
    """
    height, width = image.shape[:2]
    small_size = (max(8, width // 8), max(8, height // 8))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    values = small.mean(axis=2).ravel()  # brightness of each pixel

    # Pixel positions scaled to -1..1, and the terms of a flat and a curved surface
    ys, xs = np.mgrid[0:small.shape[0], 0:small.shape[1]]
    x = xs.ravel() / (small.shape[1] - 1) * 2 - 1
    y = ys.ravel() / (small.shape[0] - 1) * 2 - 1
    flat = np.stack([np.ones_like(x), x, y], axis=1)
    curved = np.concatenate([flat, np.stack([x * x, x * y, y * y], axis=1)], axis=1)

    # Step 1: one guess per tile, plus how evenly coloured the tile is
    brightness = values.reshape(small.shape[:2])
    guesses, evenness, tile_x, tile_y = [], [], [], []
    for rows in np.array_split(np.arange(small.shape[0]), 6):
        for cols in np.array_split(np.arange(small.shape[1]), 6):
            tile = brightness[np.ix_(rows, cols)]
            counts, edges = np.histogram(tile, bins=64, range=(0, 256))
            peak = counts.argmax()
            guesses.append(edges[peak] + 2)  # centre of the fullest 4-wide bin
            evenness.append(counts[max(0, peak - 1):peak + 2].sum() / tile.size)
            tile_x.append(cols.mean() / (small.shape[1] - 1) * 2 - 1)
            tile_y.append(rows.mean() / (small.shape[0] - 1) * 2 - 1)
    guesses, evenness = np.array(guesses), np.array(evenness)
    tile_terms = np.stack([np.ones_like(guesses), tile_x, tile_y], axis=1)

    # Step 2: start from the more evenly coloured half of the tiles, then keep
    # only tiles that agree with the fitted surface
    trusted = evenness >= np.median(evenness)
    for _ in range(4):
        coefficients = np.linalg.lstsq(tile_terms[trusted], guesses[trusted], rcond=None)[0]
        disagreement = np.abs(guesses - tile_terms @ coefficients)
        trusted = disagreement < 2 * BACKGROUND_TOLERANCE
        if trusted.sum() < 3:
            trusted = disagreement <= np.median(disagreement)
    is_slide = np.abs(values - flat @ coefficients) < BACKGROUND_TOLERANCE

    for step in range(6):
        if is_slide.sum() < 10:  # too few slide pixels to fit anything: use them all
            is_slide[:] = True
        # A flat surface first is more stable; then allow curves (e.g. dark corners)
        terms = flat if step < 2 else curved
        coefficients = np.linalg.lstsq(terms[is_slide], values[is_slide], rcond=None)[0]
        is_slide = np.abs(values - terms @ coefficients) < BACKGROUND_TOLERANCE

    if is_slide.sum() < 10:
        is_slide[:] = True
    # Fit red, green and blue separately, using the final slide pixels
    colours = small.reshape(-1, 3)
    background = curved @ np.linalg.lstsq(curved[is_slide], colours[is_slide], rcond=None)[0]
    background = np.clip(background, 0, 255).reshape(small.shape).astype(np.uint8)
    return cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)


def foreground_mask(image):
    """Return a black-and-white mask: white where something differs from the slide."""
    # How different is each pixel from the slide behind it? Taking the biggest
    # difference across red, green and blue catches cells that differ only in colour.
    difference = cv2.absdiff(image, estimate_background(image)).max(axis=2)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)  # smooth out camera noise

    # Most of a smear is empty slide, so the median difference is the camera's
    # noise level. Subtracting it stops grainy photos from looking like cells.
    noise_floor = np.median(difference)
    difference = cv2.subtract(difference, np.full_like(difference, noise_floor))

    # Very dark objects (stain clumps, white blood cells) would drag Otsu's cut-off
    # so high that pale red cells fall below it. Capping the difference stops that.
    difference = np.minimum(difference, CONTRAST_CAP)

    # Otsu's method picks the best "different enough" cut-off automatically
    _, mask = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu always splits the image somehow, even an empty slide. Only trust the
    # split if the "different" pixels really differ from the rest on average.
    above, below = difference[mask > 0], difference[mask == 0]
    if above.size == 0 or below.size == 0 or above.mean() - below.mean() < MIN_CONTRAST:
        return np.zeros_like(mask)

    # "Opening" removes tiny specks; "closing" fills small gaps in cell edges.
    # Only one closing step, so the thin bright gaps between packed cells survive.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def split_clump(clump_mask, cell_radius):
    """Split a blob of touching cells into separate cells.

    1. Distance transform: every pixel gets its distance to the blob's edge, so
       each cell looks like a hill whose peak sits at the cell's centre.
    2. Find the peaks: one per cell.
    3. Every pixel of the blob joins the cell whose edge it is closest to
       (distance to that centre, minus that cell's radius).
    Returns one outline per cell, or [] if the blob is really just one cell.
    """
    distance = cv2.distanceTransform(clump_mask, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (0, 0), 2)

    # A peak is a pixel that is the highest point within a cell-radius window
    window = max(3, int(cell_radius) | 1)
    local_max = cv2.dilate(distance, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window)))
    peaks = ((distance >= local_max) & (distance > 0.5 * cell_radius)).astype(np.uint8)
    peaks = cv2.dilate(peaks, np.ones((3, 3), np.uint8))  # merge peak pixels that touch

    n_labels, labels, _, centroids = cv2.connectedComponentsWithStats(peaks)
    if n_labels <= 2:  # label 0 is "not a peak", so 2 labels means just one cell
        return []

    centres = centroids[1:].astype(np.float32)  # (x, y) of each cell centre
    radii = np.array([distance[labels == i].max() for i in range(1, n_labels)], np.float32)

    ys, xs = np.nonzero(clump_mask)
    # One row per blob pixel, one column per cell: how far is this pixel from that cell's edge?
    gap_to_edge = np.hypot(xs[:, None] - centres[None, :, 0],
                           ys[:, None] - centres[None, :, 1]) - radii[None, :]
    owner = gap_to_edge.argmin(axis=1)

    pieces = []
    for cell_index in range(len(centres)):
        region = np.zeros_like(clump_mask)
        region[ys[owner == cell_index], xs[owner == cell_index]] = 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            pieces.append(max(contours, key=cv2.contourArea))
    return pieces


def looks_like_one_cell(contour):
    """Single cells are roughly round with no big dents; clumps and slivers are not."""
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    _, (width, height), _ = cv2.minAreaRect(contour)
    solidity = area / hull_area if hull_area else 0.0  # 1.0 = no dents at all
    elongation = max(width, height) / max(1.0, min(width, height))
    return solidity > 0.9 and elongation < 2


def find_cells(image):
    """Find the outline of every cell with classic image processing (no AI needed).

    Returns (cells, dense_regions). Dense regions are clumps packed so tightly that
    they can't be split into cells reliably, so they are skipped, not guessed at.
    """
    if min(image.shape[:2]) < MIN_IMAGE_SIDE:
        return [], []

    mask = foreground_mask(image)
    # Contours are the outlines of each white blob; EXTERNAL ignores holes inside
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = MAX_AREA_FRACTION * image.shape[0] * image.shape[1]
    blobs = [c for c in contours if MIN_CELL_AREA <= cv2.contourArea(c) <= max_area]
    if not blobs:
        return [], []

    # "Typical cell" size: the median of the blobs that look like single cells
    # (slivers and clumps would drag the median the wrong way)
    single_looking = [cv2.contourArea(c) for c in blobs if looks_like_one_cell(c)]
    areas = single_looking if len(single_looking) >= 3 else [cv2.contourArea(c) for c in blobs]
    typical_area = float(np.median(areas))
    cell_radius = np.sqrt(typical_area / np.pi)

    cells, dense_regions = [], []
    for blob in blobs:
        area = cv2.contourArea(blob)
        if area < SPECK_FRACTION * typical_area:
            continue  # debris, not a cell
        if area <= CLUMP_FACTOR * typical_area:
            cells.append(blob)
            continue

        # Probably touching cells: work on a small padded patch, then shift back
        x, y, w, h = cv2.boundingRect(blob)
        pad = 2
        clump_mask = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        cv2.drawContours(clump_mask, [blob], -1, 255, thickness=cv2.FILLED,
                         offset=(pad - x, pad - y))
        pieces = [p + np.array([x - pad, y - pad], dtype=np.int32)
                  for p in split_clump(clump_mask, cell_radius)
                  if cv2.contourArea(p) >= SPECK_FRACTION * typical_area]

        for piece in pieces or [blob]:
            piece_area = cv2.contourArea(piece)
            # Over 3x a cell and misshapen, or over 6x a cell whatever its shape:
            # no single red blood cell is that big, so don't pretend it is one
            too_big = piece_area > 2 * DENSE_FACTOR * typical_area
            if too_big or (piece_area > DENSE_FACTOR * typical_area and not looks_like_one_cell(piece)):
                dense_regions.append(piece)
            else:
                cells.append(piece)

    # Sort in reading order: top-to-bottom in rows one cell tall, then left-to-right
    row_height = max(1, int(2 * cell_radius))
    cells = sorted(cells, key=lambda c: (cv2.boundingRect(c)[1] // row_height,
                                         cv2.boundingRect(c)[0]))
    return cells, dense_regions


def detect_cells(image):
    """Just the cell outlines from find_cells."""
    return find_cells(image)[0]


def crop_cell(image, contour):
    """Cut one cell out and blacken everything around it, like the training images."""
    x, y, w, h = cv2.boundingRect(contour)
    cell_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(cell_mask, [contour], -1, 255, thickness=cv2.FILLED)
    crop = image[y:y + h, x:x + w].copy()
    crop[cell_mask[y:y + h, x:x + w] == 0] = 0
    return crop, (x, y, w, h)


def match_training_colours(crops):
    """Recolour cells so their overall colour matches the training images.

    Labs use different stains and cameras: our training cells are pink, while other
    photos can be grey-blue. LAB colour space separates brightness (L) from colour
    (A and B). We shift and stretch each channel so the cells' average and spread
    match the training cells (a method called Reinhard colour transfer).
    Every cell in one smear gets the SAME adjustment, so an infected cell still
    looks different from its healthy neighbours.
    """
    labs = [cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float64) for crop in crops]
    masks = [crop.sum(axis=2) > 0 for crop in crops]  # True on the cell, False on black
    pixels = np.concatenate([lab[mask] for lab, mask in zip(labs, masks)])
    if len(pixels) < 100:  # too little to measure reliably
        return crops

    mean, std = pixels.mean(axis=0), pixels.std(axis=0) + 1e-6
    recoloured = []
    for lab, mask in zip(labs, masks):
        matched = (lab - mean) / std * TRAINING_LAB_STD + TRAINING_LAB_MEAN
        rgb = cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        rgb[~mask] = 0
        recoloured.append(rgb)
    return recoloured


def classify_cells(image, contours, model, device):
    """Ask the model about every detected cell. Returns one result dict per cell."""
    crops, boxes = [], []
    for contour in contours:
        crop, box = crop_cell(image, contour)
        crops.append(crop)
        boxes.append(box)
    if STAIN_NORMALIZE and crops:
        crops = match_training_colours(crops)

    results = []
    # Process in batches so a smear with hundreds of cells still fits in GPU memory
    for start in range(0, len(crops), config.BATCH_SIZE):
        batch = torch.stack([eval_transform(Image.fromarray(c))
                             for c in crops[start:start + config.BATCH_SIZE]])
        with torch.no_grad():
            # softmax turns the model's raw scores into probabilities that add up to 1
            probabilities = torch.softmax(model(batch.to(device)), dim=1).cpu()

        for crop, box, probs in zip(crops[start:], boxes[start:], probabilities):
            infection_probability = float(probs[1])
            label = int(infection_probability >= config.INFECTED_THRESHOLD)
            results.append({
                "box": box,
                "crop": crop,
                "label": label,
                "class_name": config.CLASSES[label],
                "infection_probability": infection_probability,
                "uncertain": config.REVIEW_ABOVE < infection_probability < config.INFECTED_THRESHOLD,
            })
    return results


def summarize(results, dense_regions=()):
    """Count cells and compute parasitemia: the % of red blood cells infected."""
    total = len(results)
    infected = sum(1 for r in results if r["label"] == 1)
    return {
        "total_cells": total,
        "infected_cells": infected,
        "healthy_cells": total - infected,
        "parasitemia": infected / total if total else 0.0,
        "uncertain_cells": sum(1 for r in results if r["uncertain"]),
        "dense_regions_skipped": len(dense_regions),
    }


def draw_results(image, results, dense_regions=()):
    """Draw a numbered box on every cell: green healthy, red infected, orange unsure.

    Dense regions that were skipped get a grey outline, so the user can see them.
    """
    annotated = image.copy()
    cv2.drawContours(annotated, list(dense_regions), -1, GREY, 3)
    for region in dense_regions:
        x, y, _, _ = cv2.boundingRect(region)
        cv2.putText(annotated, "too dense", (x + 4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
    for number, r in enumerate(results, start=1):
        x, y, w, h = r["box"]
        color = ORANGE if r["uncertain"] else (RED if r["label"] == 1 else GREEN)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        cv2.putText(annotated, str(number), (x + 4, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return annotated


def analyze_smear(image, model, device):
    """The full pipeline in one call: detect -> classify -> count -> draw."""
    cells, dense_regions = find_cells(image)
    results = classify_cells(image, cells, model, device)
    return results, summarize(results, dense_regions), draw_results(image, results, dense_regions)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(device)

    # Build demo smears only from TEST cells, which the model never trained on
    test_cells = split_samples(list_images())["test"]
    smear_dir = config.PROJECT_DIR / "sample_smears"
    smear_dir.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)

    for number, infected_fraction in enumerate([0.25, 0.10, 0.0], start=1):
        smear, truth = make_synthetic_smear(test_cells, infected_fraction, seed=number)
        Image.fromarray(smear).save(smear_dir / f"smear_{number}.png")

        results, summary, annotated = analyze_smear(smear, model, device)
        true_infected = sum(label for *_, label in truth)
        print(f"smear_{number}: truth {len(truth)} cells, {true_infected} infected "
              f"({true_infected / len(truth):.1%}) | found {summary['total_cells']} cells, "
              f"{summary['infected_cells']} infected ({summary['parasitemia']:.1%}), "
              f"{summary['uncertain_cells']} uncertain")
        Image.fromarray(annotated).save(config.OUTPUTS_DIR / f"smear_{number}_analyzed.png")

    print(f"Saved demo smears to {smear_dir} and results to {config.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
