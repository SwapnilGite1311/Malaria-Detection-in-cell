# Project Notes: How Everything Works

Plain-language notes for revising the project and answering questions about it.

---

## Which models are pretrained?

**We trained all 4 models ourselves** on the 22,336 Kaggle training cells. The difference
is where each one *started*:

| Model | Starting point | Meaning |
|---|---|---|
| MobileNetV3-Small | Pretrained on ImageNet | Fine-tuned by us (transfer learning) |
| EfficientNet-B0 | Pretrained on ImageNet | Fine-tuned by us (transfer learning) |
| ResNet18 | Pretrained on ImageNet | Fine-tuned by us (transfer learning) |
| Simple CNN | Random numbers | Trained from scratch by us, no prior knowledge |

"Pretrained" means someone (the PyTorch team) already trained the model on ImageNet's
1.2 million everyday photos, so it already knows edges, shapes and textures. We
downloaded those starting weights and taught the model malaria cells. Nobody gave us a
ready-made malaria model.

---

## The project in one paragraph

A doctor looks at a blood smear under a microscope and counts how many red blood cells
contain the malaria parasite. That is slow and tiring. Our app does it automatically:
it **finds** every cell in the image (OpenCV), **classifies** each one as infected or
healthy (a ResNet18 model fine-tuned on 27,558 labelled cell images), **counts** the
percentage infected (parasitemia), **flags** cells it is unsure about for a human, and
**explains** its decisions with heatmaps (Grad-CAM).

**Pipeline:** image → find cells → crop each cell → model predicts → count → report

---

## Task 1: Loading and splitting the data (`src/dataset.py`)

**What:** Lists all 27,558 images, splits them into train (22,336), validation (2,613)
and test (2,609), and feeds them to the model in batches of 64.

**Key idea: split by source photo.** Each cell image was cut from a bigger microscope
photo. Cells from the same photo look alike, so all cells of one photo go into the
same set. Otherwise the test would contain near-copies of training images
(**data leakage**) and the accuracy would be fake.

**Augmentation:** training images are randomly flipped and rotated so the model learns
what a parasite looks like, not which way up the cell was.

**Python to revise:** functions, classes (`CellDataset`), `__len__`/`__getitem__`,
`enumerate`, `defaultdict`, list comprehensions, sets and `&`, f-strings,
`if __name__ == "__main__":`, `pathlib.Path`.

**Cross-questions**
- *Why three sets?* Train = learn, validation = check progress and pick the best
  version, test = the untouched final exam.
- *Why 224×224?* It's the size ResNet18 was originally trained on.
- *What is a batch?* A group of images processed together. Faster than one at a time.

---

## Task 2: Training (`src/model.py`, `src/train.py`)

**What:** Takes ResNet18, already trained on ImageNet (1.2 million everyday photos),
and fine-tunes it to recognise infected cells. This is **transfer learning**.

**The model change:** ResNet18's last layer outputs 1,000 scores (one per ImageNet
object). We replaced it with a layer that outputs 2 scores: Uninfected and Parasitized.

**The training loop: 5 steps, repeated for every batch**
1. The model guesses (`outputs = model(images)`)
2. The loss function measures how wrong the guess was (`CrossEntropyLoss`)
3. Clear old adjustments (`optimizer.zero_grad()`)
4. Work out how each weight caused the error (`backward()`, called backpropagation)
5. Nudge the weights to reduce the error (`optimizer.step()`)

**Settings:** 5 epochs, AdamW optimizer, learning rate 0.0001 (small, so we gently
adjust what the model already knows instead of wiping it). Mixed precision
(`autocast`) uses 16-bit numbers where safe: faster, and fits a 4 GB GPU.

**Results**

| Epoch | Train acc | Val acc | Saved? |
|---|---|---|---|
| 1 | 95.26% | 96.82% | ✅ |
| 2 | 96.84% | 97.13% | ✅ |
| 3 | 97.03% | 97.51% | ✅ |
| 4 | 97.24% | 97.44% | ❌ (worse than epoch 3) |
| 5 | 97.43% | 97.67% | ✅ final model |

Chart: `outputs/training_curves.png`

**Python to revise:** default arguments (`optimizer=None`), `with` blocks (context
managers), `zip`, dictionary of lists, `:.2%` formatting, `json.dump`.

**Cross-questions**
- *Why is validation accuracy higher than training accuracy?* Training images are made
  harder by random flips and rotations, and training accuracy is averaged over the
  whole epoch while the model is still improving. Validation is measured at the end.
- *Why only 5 epochs?* The model was already at 96.8% after 1 epoch because it was
  pretrained. Accuracy was levelling off, and more epochs risk **overfitting**
  (memorising training images instead of learning general patterns).
- *Why save only the best epoch?* Epoch 4 was slightly worse than epoch 3; we keep
  whichever version did best on validation data.
- *What is ResNet?* A CNN (convolutional neural network) with "skip connections" that
  let very deep networks train well. The 18 means 18 layers.

---

## Task 3: Testing (`src/evaluate.py`)

**What:** The final exam on 2,609 cells the model never saw during training.

**Decision rule:** a cell is called infected only if the model is at least **80%** sure
(`INFECTED_THRESHOLD` in `src/config.py`). Cells between 20% and 80% are flagged for review.

| Metric | Score | Meaning |
|---|---|---|
| Accuracy | **95.98%** | Correct answers overall |
| Sensitivity (recall) | **93.04%** | Of 1,250 infected cells, caught 1,163 (87 missed) |
| Specificity | **98.68%** | Of 1,359 healthy cells, cleared 1,341 (18 false alarms) |

At the default 50% rule the scores were 96.59% / 95.76% / 97.35%. See Task 8 for why
we moved to 80%.

**Confusion matrix** (`outputs/confusion_matrix.png`): a 2×2 table of truth vs.
prediction, showing exactly which kind of mistakes the model makes.

**Python to revise:** nested tuple unpacking `(tn, fp), (fn, tp) = ...`,
`list.extend`, `isinstance`, one-line `if/else` expressions.

**Cross-questions**
- *Why is test accuracy lower than validation (97.67%)?* We chose the best epoch using
  validation data, so validation is slightly optimistic. Also, the 80% rule trades a
  little accuracy for fewer false alarms.
- *Which metric matters most for malaria?* Sensitivity. A missed infection (a sick
  patient sent home) is worse than a false alarm (which a doctor re-checks).
- *Why not just report accuracy?* Accuracy hides which mistakes happen. If 95% of
  cells were healthy, a model that always says "healthy" would score 95% and be useless.

---

## Task 4: Whole-smear analysis (`src/smear.py`)

**What:** Analyses a full smear image with many cells.

**Synthetic smears:** the Kaggle dataset only has single-cell crops, so we build test
smears by placing 24 unseen **test** cells on a blank slide. Because we placed them,
we know the true answer for every cell.

**Finding cells (classic image processing, no AI)**
1. **Estimate the empty slide.** Split a shrunken copy into a 6×6 grid; in each tile the
   most common brightness is probably slide. Fit a smooth surface through those guesses
   (ignoring tiles full of cells), then refine it. This follows uneven lighting and
   still works when cells cover most of the image
2. **Measure the difference** between the image and that slide estimate. Cells stand
   out whether the background is light or dark
3. **Remove the noise floor** (the median difference), so grainy photos don't look like cells
4. **Cap big differences**, so one very dark object (a stain clump or white blood cell)
   can't push the cut-off so high that pale red cells disappear
5. **Otsu's threshold** picks the "different enough" cut-off. If the two groups barely
   differ on average, it's an empty slide
6. **Opening** removes tiny specks; one **closing** step fills small gaps in cell edges
7. **Contours** trace each blob. The median size of the blobs that *look like single
   cells* (round, no dents) becomes the "typical cell", so size rules work at any zoom
8. **Split clumps.** Blobs over 1.6× a typical cell are touching cells. A distance
   transform turns each cell into a hill; each hilltop is a cell centre, and every
   pixel joins the nearest centre (adjusted for that cell's size)
9. **Skip dense regions.** Pieces still over 3× a typical cell (and misshapen), or over
   6× whatever their shape, are outlined in grey as "too dense" instead of guessed at
10. Sort cells in reading order (top-to-bottom, left-to-right)

**Classifying**
- Each cell is cropped and the area around it is made black, like the training images
- **Colour matching:** all cells in the smear get the same colour adjustment so their
  average colour matches the training cells (Reinhard colour transfer in LAB colour space)
- **Softmax** turns the model's scores into an infection probability. 80% or more =
  infected; 20–80% = flagged orange for review

**Parasitemia** = infected cells ÷ all cells counted × 100.

**Results on the demo smears**

| Smear | True | Found |
|---|---|---|
| smear_1 | 23 cells, 6 infected (26.1%) | 23 cells, 5 infected (21.7%), 2 for review |
| smear_2 | 24 cells, 2 infected (8.3%) | 24 cells, 1 infected (4.2%), 2 for review |
| smear_3 | 24 cells, 0 infected (0%) | 24 cells, 0 infected (0%), 1 for review |

Cell counts are exact. The stricter 80% rule misses one infected cell in smears 1 and 2
that the 50% rule caught. That's the price of fewer false alarms on real photos.

**Python to revise:** `random.Random(seed)`, `any()` with a generator, NumPy slicing
`canvas[y:y+h, x:x+w]`, boolean masks, `lambda` as a sort key, `range(start, stop, step)`
for batching, `*_` unpacking.

**Cross-questions**
- *Why synthetic smears?* The dataset has no full smear images with per-cell labels.
  Synthetic smears give us a known ground truth to measure against.
- *Why not use AI to find cells too?* Cells on a light background are easy to separate
  with thresholding: fast, needs no training data, and easy to explain.
- *What about touching cells or bad lighting?* The detector estimates the slide's
  lighting everywhere and splits clumps of touching cells. Automated tests cover dark
  backgrounds, uneven lighting, JPEG compression, zoomed images and touching cells.
- *Limitations?* See Task 8: on real photos about 73% of cells are found, and heavily
  packed areas are skipped rather than counted.
- *Why black out the background?* The model only ever saw cells on black. Showing it
  something different could confuse it (a **domain gap**).

---

## Task 5: Grad-CAM heatmaps (`src/gradcam.py`)

**What:** Shows which parts of a cell influenced the model's decision. Red means it
mattered most; blue means it barely mattered.

**How (simply):** it looks at the model's last image-reading layer (`layer4`),
measures how much each area pushed the score toward the predicted class, and paints
that as a heatmap on top of the cell.

**What we saw** (`outputs/gradcam_examples.png`): for infected cells, the red spots sit
right on the purple parasite dots. For healthy cells, attention spreads across the
middle, because there is no single suspicious spot.

**Python to revise:** `with ... as cam:` context managers, `next(model.parameters())`,
`unsqueeze(0)` to make a batch of one, NumPy `dtype` and dividing by 255.

**Cross-questions**
- *Why is Grad-CAM important?* It shows the model is looking at the parasite and not a
  shortcut (like image brightness or a border). Doctors won't trust a black box.
- *Why is the heatmap blurry?* `layer4` sees the image as a 7×7 grid, which gets
  stretched up to 224×224.

---

## Task 6: The web app (`app.py`)

**What:** A Streamlit website with two tabs:
- **Blood smear:** upload or pick a smear, then see counts, parasitemia %, the annotated
  image, flagged cells with heatmaps, a warning if dense areas were skipped, and
  download a CSV report
- **Single cell:** upload one cell image, then see the verdict, infection probability,
  a review warning if the model is unsure, and the heatmap

**Model picker (sidebar):** choose between the 4 trained models. It starts with
MobileNetV3-Small, the best on real photos (Task 9). Below the picker, the sidebar shows
that model's test-set and real-photo scores, file size and CPU speed. Picking the
simple CNN shows a warning that it's only there for comparison. Cell detection is the
same for every model; only the infected/healthy decisions change. The CSV report
records which model was used.

**Grad-CAM for every model:** each model type has a different "last image-reading
layer" (`target_layer` in `src/gradcam.py`): `layer4` for ResNet18, the last `features`
block for MobileNetV3 and EfficientNet, and the last ReLU for the simple CNN.

**Caching with a model picker:** results are cached per image *and* per model name, so
switching models never shows the previous model's answers.

**How Streamlit works:** every time you click something, Streamlit re-runs the whole
script from top to bottom. `@st.cache_resource` makes sure the model loads only once
instead of on every click.

**Privacy settings** (`.streamlit/config.toml`): usage statistics off; the app is only
reachable from this computer.

**Python to revise:** decorators (`@st.cache_resource`), `with column:` blocks,
`i % 4` to place cards in a 4-column grid, `io.StringIO` + `csv.writer`, early
`return` to stop a function, `isinstance` checks, `next(generator, None)` to find the
first match or nothing, `format_func=dict.get` to show friendly names in a dropdown.

**Cross-questions**
- *Why Streamlit?* It turns a Python script into a web app without HTML or JavaScript.
- *Why limit heatmaps to 12?* Grad-CAM is slower than a normal prediction.
- *Why a model picker?* It shows the comparison live: the same image, different models,
  different decisions. And anyone can check our claim that MobileNetV3 is the best choice.
- *Why does the cell count stay the same when you switch models?* Cells are found by
  OpenCV rules, not by the AI model. The model only decides infected or healthy.

---

## Task 7: System audit and automated tests (`tests/test_system.py`)

**What:** 31 automated tests check the entire system with one command:
`python -m pytest -v`. They take about 70 seconds.

| Area | What the tests check |
|---|---|
| Data | All 27,558 images used; no photo shared between sets; same split every run |
| Model | Accuracy above 93% on a sample of test cells |
| Detection | Empty, white, black and tiny images → 0 cells; camera noise → 0 cells |
| Detection | Exact counts on the 3 demo smears; JPEG; dark background; uneven lighting; 0.5× and 3× zoom; touching cells |
| Grad-CAM | Heatmap has the right size and format, for all 4 models |
| Model picker | Starts on MobileNetV3; switching to every model analyses smear_1 without errors |
| App | Broken files don't crash it; RGBA/grayscale/palette images load; CSV report; single-cell prediction; the app itself runs and shows 23 cells for smear_1 |
| Real photos | On 40 expert-labelled photos: cells found ≥ 65%, boxes correct ≥ 65%, sensitivity ≥ 85%, specificity ≥ 95% |

**What the audit found and fixed**

| Problem found | Fix |
|---|---|
| Dark background → whole image became one "cell" | Compare against an estimate of the slide instead of assuming it's bright |
| Uneven lighting → half the image became one "cell" | The slide estimate follows the lighting |
| Touching cells merged (21 of 30 found) | Clump splitting (now 31 of 30) |
| Fixed minimum cell size broke zoomed-out images | Sizes relative to the typical cell |
| Grainy photo → 203 fake "cells" | Subtract the noise floor |
| Broken upload crashed the app | `read_image` returns `None` and the app shows a message |
| Download button re-ran the whole analysis | `st.cache_data` + `on_click="ignore"` |
| Single cell in the smear tab gave wrong results | The app spots small images and points you to the right tab |

**Python to revise:** `pytest` fixtures (`@pytest.fixture`), `@pytest.mark.parametrize`
(one test, many inputs), `monkeypatch` (temporarily change a setting), `assert`.

**Cross-questions**
- *How do you know it works?* 31 automated tests, including tricky images like dark
  backgrounds, noise, compression and touching cells, plus 40 real photos.
- *Why write tests?* After changing code, one command proves nothing else broke.
  The first run of these tests caught 3 real bugs.

---

## Task 8: Testing on real microscope photos (`scripts/fetch_real_smears.py`, `src/real_eval.py`)

**What:** Everything before this used synthetic smears. We tested on **40 real smear
photos** from the BBBC041 dataset (Broad Institute), where experts drew a labelled box
around each of the **1,869 red blood cells**.

**Downloading only what we need:** the full dataset is a 2.26 GB zip. A zip keeps its
table of contents at the end, and the server allows downloading any byte range, so
the script reads the table of contents and fetches only 40 photos (10 MB total).

**How we score it**
- **IoU (intersection over union):** overlap area ÷ combined area of two boxes. A
  detection counts as finding an expert's cell if IoU ≥ 0.5
- **Detection recall:** share of expert-labelled cells we found
- **Detection precision:** share of our boxes that were real cells
- **Sensitivity / specificity:** measured on the cells we found
- **Parasitemia error:** how far our % is from the experts' %, averaged over photos

**Results: before and after fixing real-photo problems**

| Metric | First try | Final |
|---|---|---|
| Cells found (recall) | 50.7% | **72.9%** |
| Boxes that were real cells (precision) | 42.7% | **74.7%** |
| Photos with 0 cells found | 6 | **0** |
| Sensitivity | 67.9% | **95.9%** |
| Specificity | 97.9% | **98.4%** |
| Average parasitemia error | 4.1 points | 4.2 points |

**What went wrong and how we fixed it**

| Problem on real photos | Cause | Fix |
|---|---|---|
| 6 photos: 0 cells found | Real cells are faint; a safety check rejected them as "empty slide" | Check the *average* difference between groups instead of the cut-off value |
| Packed clusters → 150 junk boxes | The old slide estimate (a local average) mistook big clusters for slide | Slide estimate from the most common colour per tile + a smooth surface |
| Pale cells missed next to a dark stain clump | The dark clump pushed the cut-off too high | Cap large differences |
| Clumps too dense to split → giant wrong boxes | Splitting can't separate tightly packed cells | Skip them and show a grey "too dense" outline and a warning |
| Grey-blue cells classified poorly (only 37% of infections caught at the 80% rule) | Training cells are pink: a **domain gap** | **Colour matching** raised sensitivity to 96% |
| Too many false alarms after colour matching | Healthy cells outnumber infected ~25 to 1, so a few % false alarms inflate parasitemia | Stricter 80% rule: specificity 96.3% → 98.4% |

**Honest limitations**
- The parasitemia error didn't improve overall (4.1 → 4.2 points). Detection now finds
  far more real cells, but some extra boxes are still cell fragments, and 2 photos
  that are mostly one packed cluster still go badly.
- About 27% of real cells are still missed, mostly in packed clusters.
- The real photos show **P. vivax**, but we trained on **P. falciparum**. The classifier
  still transferred well, but a model trained on several species would be better.
- The 80% threshold was chosen by looking at these same 40 photos, so their scores
  are slightly optimistic. A fair check would use a fresh set of photos.

**Python to revise:** a class that imitates a file (`seek`, `read`, `tell`), HTTP `Range`
headers with `urllib`, `zipfile`, JSON, sorting with `reverse=True`, sets to avoid
double-matching, dictionary keys that are tuples (`{(True, False): "fp"}`).

**Cross-questions**
- *Does it work on real images?* Yes, with limits: 73% of cells found and 96%
  sensitivity / 98% specificity on the cells found, measured against expert labels.
- *Why did accuracy drop on real photos?* Different lab, stain, camera and malaria
  species. This is called **domain shift**, and it's one of the biggest problems in
  medical AI.
- *What is colour normalization?* Adjusting an image's colours to match the training
  data, so the model judges the parasite and not the stain colour.
- *Why not change the threshold to 50% again?* On real photos, 50% raised false alarms
  so much that parasitemia got worse (6.5 vs 3.3 points at the time we tested).
- *How would you improve detection?* Train a detection model (e.g. YOLO) on BBBC041's
  labelled boxes instead of using hand-written rules.

---

## Task 9: Model comparison (`src/compare_models.py`)

**What:** We trained 3 more models on exactly the same data, for 5 epochs each, and
tested all 4 the same way. Train one with `python -m src.train --model mobilenet_v3_small`,
then compare with `python -m src.compare_models`.

| Model | What it is |
|---|---|
| **Simple CNN** | 4 convolution blocks we wrote ourselves, trained **from scratch** (no pretraining) |
| **MobileNetV3-Small** | Pretrained, designed to run fast on phones |
| **EfficientNet-B0** | Pretrained, designed for the best accuracy per unit of computing |
| **ResNet18** | Pretrained, the model the app uses |

**Results** (infected = 80% or more; real photos use colour matching)

| Model | Parameters | File | Train | Best val acc | Kaggle acc | Kaggle sens | Kaggle spec | Real sens | Real spec | Real parasitemia error | CPU per cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Simple CNN | 0.39 M | 1.6 MB | 7 min | 95.52% | 94.0% | 90.2% | 97.5% | 98.6% | **76.0%** | **29.3 pts** | 44 ms |
| MobileNetV3-Small | 1.52 M | 6.2 MB | 7 min | 97.51% | 96.2% | 93.4% | 98.7% | 95.9% | **99.4%** | **1.9 pts** | **8.5 ms** |
| EfficientNet-B0 | 4.01 M | 16.3 MB | 10 min | **98.05%** | **96.6%** | **94.6%** | 98.5% | 89.2% | 99.2% | 2.0 pts | 63 ms |
| ResNet18 (app) | 11.18 M | 44.8 MB | 12 min | 97.67% | 96.0% | 93.0% | 98.7% | 95.9% | 98.4% | 4.2 pts | 85 ms |

**What the results show**
1. **Pretraining matters most.** The simple CNN scored 94% on Kaggle but fell apart on
   real photos: it called 24% of healthy cells infected. With no ImageNet knowledge, it
   learned shortcuts tied to the Kaggle stain colours (**overfitting to one domain**).
2. **Bigger isn't better.** ResNet18 has 7× more parameters than MobileNetV3, but
   MobileNetV3 matched or beat it everywhere and is 10× faster on CPU.
3. **Best on Kaggle isn't best in the real world.** EfficientNet-B0 won on Kaggle, but
   caught the fewest real infections (89%). The test set from the *same* source as the
   training data can't tell you how a model handles new labs.
4. **On real photos, MobileNetV3 was best:** parasitemia error 1.9 points vs 4.2 for ResNet18.

**Example: the same real photos, four models** (a good live demo with the model picker)

| Real photo | Experts | MobileNetV3 | EfficientNet-B0 | ResNet18 | Simple CNN |
|---|---|---|---|---|---|
| `34f334e6…` | 6 of 38 infected (15.8%) | 18.2% | 20.5% | 15.9% | **47.7%** ❌ |
| `8874ea02…` | 5 of 49 infected (10.2%) | 10.0% | 10.0% | 10.0% | **30.0%** ❌ |

The three pretrained models land close to the experts. The simple CNN calls roughly
3 times too many cells infected.

**Be careful with these numbers**
- Each model was trained once. Another run with different randomness could shift
  scores by around half a percent, so small gaps (like 96.0% vs 96.2%) aren't meaningful.
- The 80% threshold and colour matching were tuned using ResNet18 on the same 40 real
  photos, which may slightly favour ResNet18. MobileNetV3 won anyway.
- 40 real photos contain only 74 infected cells, so a few cells change sensitivity by
  several percent.
- Speeds were measured on this laptop (CPU: one cell at a time; GPU: batches of 64).

**Decision:** the app now starts with MobileNetV3-Small. It was the best on real photos,
is the fastest on a normal CPU, and has a 6 MB file. The other models
stay available in the sidebar's model picker (Task 6).

**Python to revise:** `argparse` for command-line options, `if/elif` returning different
objects, a class inheriting from `nn.Module` with `__init__` and `forward`, `*args`
unpacking into `nn.Sequential(*blocks)`, `time.perf_counter`, `lambda` for formatting.

**Cross-questions**
- *Why did you choose your model?* We compared 4 and measured accuracy, real-world
  performance, size and speed, instead of picking one by habit.
- *What is a parameter?* A number the model learns during training. More parameters
  can learn more, but need more memory and time, and can overfit.
- *What is transfer learning, and did it help?* Starting from a model pretrained on
  ImageNet. Yes: every pretrained model beat the from-scratch CNN, and hugely on real photos.
- *Why test on real photos, not just the test set?* The test set comes from the same
  source as the training data. Real photos from another lab revealed the simple CNN's
  weakness, which the test set hid.
- *Why is MobileNet so fast?* It uses "depthwise separable convolutions", which split one
  big calculation into two much cheaper ones.

---

## Problems we faced (good interview material)

1. **Training crashed with "paging file is too small."** Each data-loading worker
   process loads its own copy of PyTorch (~1.2 GB). Eight workers overflowed 8 GB of RAM.
   Fix: 2 workers that don't stay alive between stages.
2. **Matching totals can hide mistakes.** Smear_1's infected count was correct, but
   only because two errors cancelled out. We checked each cell individually to find
   this.
3. **Data leakage risk.** A random split would put near-identical cells from the same
   photo into both train and test. We split by source photo instead.
4. **Synthetic tests aren't enough.** Everything passed on synthetic smears, but the
   first real-photo test found only half the cells. Real data exposed problems the
   synthetic data never could.
5. **Domain shift.** Pink training cells vs. grey-blue real photos: only 37% of real
   infections were caught at the 80% rule until we added colour matching.
