"""Task 6: the web app that ties everything together.

Run:  streamlit run app.py
"""
import csv
import io
import json

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src import config
from src.dataset import eval_transform
from src.gradcam import gradcam_overlay
from src.model import load_trained_model
from src.smear import analyze_smear

SAMPLE_DIR = config.PROJECT_DIR / "sample_smears"
MAX_HEATMAPS = 12           # Grad-CAM is slower than a prediction, so cap how many we draw
SMALLEST_SMEAR_SIDE = 300   # images smaller than this are probably a single cell


@st.cache_resource  # load the model once, not every time the page refreshes
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_trained_model(device), device


# Streamlit re-runs this whole script on every click. st.cache_data remembers
# results for inputs it has already seen, so repeat clicks are instant.
# Parameters starting with "_" are not used to decide whether it's a repeat.
@st.cache_data(show_spinner=False, max_entries=10)
def cached_analysis(image, _model, _device):
    return analyze_smear(image, _model, _device)


@st.cache_data(show_spinner=False, max_entries=200)
def cached_heatmap(cell, label, _model):
    return gradcam_overlay(_model, cell, label)


def read_image(source):
    """Open an uploaded file or a path as an RGB numpy array, or None if it isn't an image."""
    try:
        return np.array(Image.open(source).convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None


def predict_single_cell(cell, model, device):
    """Classify a whole image as one cell. Returns (label, infection_probability)."""
    with torch.no_grad():
        batch = eval_transform(Image.fromarray(cell)).unsqueeze(0).to(device)
        infection_probability = float(torch.softmax(model(batch), dim=1)[0, 1])
    return int(infection_probability >= config.INFECTED_THRESHOLD), infection_probability


def build_csv_report(results, summary):
    """A spreadsheet-friendly report: summary lines, then one row per cell."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for key, value in summary.items():
        writer.writerow([key, f"{value:.2%}" if key == "parasitemia" else value])
    writer.writerow([])
    writer.writerow(["cell_number", "prediction", "infection_probability", "needs_review",
                     "x", "y", "w", "h"])
    for number, r in enumerate(results, start=1):
        writer.writerow([number, r["class_name"], f"{r['infection_probability']:.2%}",
                         "yes" if r["uncertain"] else "no", *r["box"]])
    return buffer.getvalue()


def show_cell_cards(results, model):
    """Show infected and uncertain cells side by side with their heatmaps."""
    flagged = [(n, r) for n, r in enumerate(results, start=1) if r["label"] == 1 or r["uncertain"]]
    if not flagged:
        st.success("No infected or uncertain cells found.")
        return

    st.caption(f"Showing {min(len(flagged), MAX_HEATMAPS)} of {len(flagged)} flagged cells. "
               "In each heatmap, red areas influenced the decision most.")
    columns = st.columns(4)
    for i, (number, r) in enumerate(flagged[:MAX_HEATMAPS]):
        with columns[i % 4]:
            st.image([r["crop"], cached_heatmap(r["crop"], r["label"], model)], width=110)
            tag = "⚠️ review" if r["uncertain"] else ""
            st.markdown(f"**Cell {number}**: {r['class_name']}, infection probability "
                        f"{r['infection_probability']:.0%} {tag}")


def smear_tab(model, device):
    st.subheader("Analyse a blood smear")
    uploaded = st.file_uploader("Upload a smear image", type=["png", "jpg", "jpeg"], key="smear")
    samples = sorted(SAMPLE_DIR.glob("*.png"))
    choice = st.selectbox("...or pick a sample smear", ["(none)"] + [p.name for p in samples])

    if uploaded:
        image = read_image(uploaded)
    elif choice != "(none)":
        image = read_image(SAMPLE_DIR / choice)
    else:
        st.info("Upload an image or pick a sample to begin.")
        return

    if image is None:
        st.error("That file couldn't be opened as an image. Please upload a PNG or JPG.")
        return
    if max(image.shape[:2]) < SMALLEST_SMEAR_SIDE:
        st.warning("This image is very small, so it's probably a single cell. "
                   "Use the **Single cell** tab for it.")
        return

    with st.spinner("Finding and checking every cell..."):
        results, summary, annotated = cached_analysis(image, model, device)

    if summary["total_cells"] == 0:
        st.warning("No cells were detected. Try a sharper image where cells stand out "
                   "from the background.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells found", summary["total_cells"])
    c2.metric("Infected", summary["infected_cells"])
    c3.metric("Parasitemia", f"{summary['parasitemia']:.1%}")
    c4.metric("Need review", summary["uncertain_cells"])

    if summary["dense_regions_skipped"]:
        st.warning(f"{summary['dense_regions_skipped']} area(s) were packed too densely to "
                   "count cells reliably, so they were skipped (grey outline). The "
                   "parasitemia is based on the other cells.")

    st.image(annotated, caption="Green = healthy, red = infected, orange = needs review, "
                                "grey = too dense to analyse", width="stretch")

    st.subheader("Flagged cells")
    show_cell_cards(results, model)

    # on_click="ignore" stops the download from re-running the whole page
    st.download_button("Download report (CSV)", build_csv_report(results, summary),
                       file_name="smear_report.csv", mime="text/csv", on_click="ignore")


def single_cell_tab(model, device):
    st.subheader("Check a single cell image")
    uploaded = st.file_uploader("Upload one cell image", type=["png", "jpg", "jpeg"], key="cell")
    if not uploaded:
        st.info("Tip: images from the Kaggle dataset work here.")
        return

    cell = read_image(uploaded)
    if cell is None:
        st.error("That file couldn't be opened as an image. Please upload a PNG or JPG.")
        return
    if max(cell.shape[:2]) >= SMALLEST_SMEAR_SIDE:
        st.warning("This image is large for a single cell. If it's a whole smear, use the "
                   "**Blood smear** tab, because this tab treats the entire image as one cell.")

    label, infection_probability = predict_single_cell(cell, model, device)

    left, right = st.columns(2)
    left.image(cell, caption="Uploaded cell", width=250)
    right.image(cached_heatmap(cell, label, model), caption="Where the model looked", width=250)

    verdict = "🔴 Infected (Parasitized)" if label == 1 else "🟢 Healthy (Uninfected)"
    st.markdown(f"### {verdict}")
    st.progress(infection_probability,
                text=f"Infection probability: {infection_probability:.1%} "
                     f"(called infected at {config.INFECTED_THRESHOLD:.0%} or more)")
    if config.REVIEW_ABOVE < infection_probability < config.INFECTED_THRESHOLD:
        st.warning("⚠️ The model isn't sure about this cell. A human should check it.")


def sidebar():
    st.sidebar.title("🔬 Malaria Smear Analyzer")
    st.sidebar.write("Finds red blood cells in a microscope image, checks each one for the "
                     "malaria parasite, and reports the percentage infected.")
    results_file = config.OUTPUTS_DIR / "test_results.json"
    if results_file.exists():
        results = json.loads(results_file.read_text())
        st.sidebar.subheader("Model performance (test set)")
        st.sidebar.write(f"Accuracy: **{results['accuracy']:.2%}**")
        st.sidebar.write(f"Sensitivity: **{results['sensitivity']:.2%}**")
        st.sidebar.write(f"Specificity: **{results['specificity']:.2%}**")
    st.sidebar.caption("Student project for learning purposes. Not a medical device.")


def main():
    st.set_page_config(page_title="Malaria Smear Analyzer", page_icon="🔬", layout="wide")
    sidebar()
    if not config.MODEL_PATH.exists():
        st.error("No trained model found. Run `python -m src.train` first.")
        return
    model, device = get_model()
    smear, single = st.tabs(["Blood smear", "Single cell"])
    with smear:
        smear_tab(model, device)
    with single:
        single_cell_tab(model, device)


# Streamlit runs this file as "__main__"; tests can import it without starting the app
if __name__ == "__main__":
    main()
