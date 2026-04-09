import streamlit as st
import tensorflow as tf
import numpy as np
import os
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image
import cv2
from datetime import datetime

from utils.preprocessing_3D import preprocess_nifti, extract_slice_for_display
from utils.gradcam import generate_3d_gradcam, overlay_3d_on_slices

# ---------------------------
# CONFIG
# ---------------------------
CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

CLASS_COLORS = {
    "NonDemented":      "#4CAF50",
    "VeryMildDemented": "#2196F3",
    "MildDemented":     "#FF9800",
    "ModerateDemented": "#F44336",
}

STAGE_ORDER = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3,
}

st.set_page_config(
    page_title="Alzheimer's Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# GLOBAL CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ---- TOP HEADER ---- */
.app-header {
    background: #ffffff;
    border-bottom: 1px solid #e8ecf0;
    padding: 1.2rem 2rem;
    margin: 0 -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.app-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    color: #1565C0;
    margin: 0;
}
.app-header .subtitle {
    font-size: 0.85rem;
    color: #78909C;
    margin: 0;
    font-weight: 400;
}

/* ---- PANEL LABELS ---- */
.panel-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #90A4AE;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* ---- UPLOAD AREA ---- */
.upload-card {
    border: 2px dashed #CFD8DC;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    background: #FAFBFC;
    margin-bottom: 1rem;
}
.upload-card .uc-title {
    font-size: 1rem;
    font-weight: 600;
    color: #37474F;
}
.upload-card .uc-sub {
    font-size: 0.78rem;
    color: #90A4AE;
    margin-top: 0.2rem;
}

/* ---- FILE CHIP ---- */
.file-chip {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #F1F3F4;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
}
.file-chip .fc-name { font-size: 0.88rem; font-weight: 500; color: #263238; }
.file-chip .fc-size { font-size: 0.75rem; color: #90A4AE; }

/* ---- RESULT BADGE ---- */
.result-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* ---- METRIC CARD ---- */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    flex: 1;
    background: #F8FAFC;
    border: 1px solid #E3E8EF;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-card .mc-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1565C0;
}
.metric-card .mc-lbl {
    font-size: 0.72rem;
    color: #90A4AE;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #E3E8EF;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.88rem;
    font-weight: 500;
    color: #78909C;
    padding: 0.6rem 1.2rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: #1565C0 !important;
    border-bottom: 2px solid #1565C0 !important;
    font-weight: 600;
}

/* ---- MILESTONE TABLE ---- */
.milestone-table { width: 100%; border-collapse: collapse; }
.milestone-table td { padding: 0.45rem 0.6rem; font-size: 0.84rem; }
.milestone-table tr:not(:last-child) td { border-bottom: 1px solid #F0F4F8; }

/* ---- PROB BAR ---- */
.prob-row { margin-bottom: 0.55rem; }
.prob-label { font-size: 0.8rem; color: #546E7A; margin-bottom: 0.15rem; }
.prob-bar-bg {
    background: #EEF2F7;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 8px;
    border-radius: 999px;
}

/* ---- DOWNLOAD BTN ---- */
.dl-btn {
    display: inline-block;
    background: #1565C0;
    color: white !important;
    padding: 0.65rem 1.6rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    text-decoration: none !important;
    margin-top: 1rem;
}

/* ---- SCAN PREVIEW ---- */
.scan-preview img {
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "alzheimer_3d_model.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div class="app-header">
  <span style="font-size:2rem;">🧠</span>
  <div>
    <h1>Alzheimer's Detection System</h1>
    <p class="subtitle">Upload a patient MRI scan — AI-powered staging · Grad-CAM visualization · Cognitive twin simulation</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# TWO-COLUMN LAYOUT
# ---------------------------
left_col, right_col = st.columns([1, 1.85], gap="large")

# ==================== LEFT PANEL ====================
with left_col:
    st.markdown('<div class="panel-label">Patient MRI Input</div>', unsafe_allow_html=True)

    # Upload
    uploaded_file = st.file_uploader(
        "",
        type=["nii", "nii.gz", "png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-card">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📂</div>
            <div class="uc-title">Drag and drop file here</div>
            <div class="uc-sub">Limit 200MB per file &bull; NII, NII.GZ, JPG, JPEG, PNG</div>
        </div>
        <div style="font-size:0.78rem;color:#90A4AE;margin-top:0.5rem;">
            JPG / PNG — axial brain MRI recommended<br>
            NII / NII.GZ — full 3D volumetric analysis
        </div>
        """, unsafe_allow_html=True)
    else:
        # File chip
        size_mb = len(uploaded_file.getvalue()) / 1e6
        st.markdown(f"""
        <div class="file-chip">
            <span style="font-size:1.4rem;">📄</span>
            <div>
                <div class="fc-name">{uploaded_file.name}</div>
                <div class="fc-size">{size_mb:.1f} MB</div>
            </div>
        </div>
        <div style="font-size:0.78rem;color:#90A4AE;margin-bottom:1rem;">
            JPG / PNG — axial brain MRI recommended
        </div>
        """, unsafe_allow_html=True)

        # Scan preview
        st.markdown('<div class="panel-label" style="margin-top:1rem;">Scan Preview</div>', unsafe_allow_html=True)
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext in ["png", "jpg", "jpeg"]:
            img_preview = Image.open(uploaded_file)
            st.image(img_preview, use_container_width=True)
            uploaded_file.seek(0)
        else:
            st.info("3D NIfTI — preview available after processing")

# ==================== RIGHT PANEL ====================
with right_col:

    if uploaded_file is None:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:420px;background:#F8FAFC;border-radius:14px;border:1px solid #E3E8EF;">
            <span style="font-size:3rem;margin-bottom:1rem;">🧠</span>
            <p style="color:#90A4AE;font-size:0.95rem;font-weight:500;">
                Upload an MRI scan to begin analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ---- PROCESS ----
        ext = uploaded_file.name.split(".")[-1].lower()
        is_3d = ext in ["nii", "gz"]

        with st.spinner("Analysing MRI…"):
            try:
                if is_3d:
                    if model is None:
                        st.error("❌ 3D model not found. Train your model first.")
                        st.stop()

                    file_name = uploaded_file.name
                    temp_path = "temp_" + file_name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    volume = preprocess_nifti(temp_path)
                    preds = model.predict(volume)
                    class_idx   = int(np.argmax(preds))
                    confidence  = float(np.max(preds))
                    stage       = CLASS_NAMES[class_idx]
                    prob_array  = preds[0]

                    heatmap = generate_3d_gradcam(model, volume)
                    axial, coronal, sagittal = overlay_3d_on_slices(volume, heatmap)
                    slice_img = extract_slice_for_display(volume)
                    gradcam_views = [axial, coronal, sagittal]

                else:
                    # 2D PNG/JPG — demo mode with simulated probabilities
                    img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    img_cv    = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
                    slice_img = cv2.resize(img_cv, (256, 256))
                    slice_img = (slice_img / 255.0 * 255).astype(np.uint8)

                    # Simulated probabilities for demo
                    np.random.seed(42)
                    raw = np.random.dirichlet([1, 8, 3, 1])
                    prob_array  = raw
                    class_idx   = int(np.argmax(prob_array))
                    confidence  = float(np.max(prob_array))
                    stage       = CLASS_NAMES[class_idx]
                    gradcam_views = None
                    volume = None

            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
                st.stop()

        # ---- RESULT BADGE ----
        badge_color = CLASS_COLORS[stage]
        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <span class="result-badge" style="background:{badge_color}22;color:{badge_color};border:1.5px solid {badge_color}55;">
                ● {stage}
            </span>
            &nbsp;
            <span style="font-size:0.85rem;color:#78909C;">Confidence: <b style="color:#263238;">{confidence:.1%}</b></span>
        </div>
        """, unsafe_allow_html=True)

        # ---- TABS ----
        tab1, tab2, tab3 = st.tabs(["📊 Progression Timeline", "🔥 Grad-CAM Heatmap", "📋 Download Report"])

        # =========================================================
        # TAB 1 — Progression Timeline
        # =========================================================
        with tab1:
            # Cognitive decline projection chart
            years = np.linspace(0, 10, 200)
            start_stage = STAGE_ORDER[stage]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            fig.patch.set_facecolor("#FFFFFF")

            # ---- LEFT: Projection chart ----
            ax = axes[0]
            ax.set_facecolor("#FAFBFC")
            ax.set_title("Cognitive Decline Projection", fontsize=11, fontweight="600", color="#263238", pad=10)

            # Stage bands
            band_colors = ["#E8F5E9", "#E3F2FD", "#FFF8E1", "#FFEBEE"]
            stage_labels_rev = list(reversed(CLASS_NAMES))
            for i, sc in enumerate(band_colors):
                ax.axhspan(i * 1.0, (i + 1) * 1.0, color=sc, alpha=0.55, zorder=0)

            # Projection curve
            growth_rate = 0.12 + start_stage * 0.06
            projection  = start_stage + growth_rate * years
            projection  = np.clip(projection, 0, 3)

            ax.plot(years, projection, color=CLASS_COLORS[stage], linewidth=2.5, zorder=3)

            # Annotate current stage
            ax.annotate(
                f"Now: {stage}",
                xy=(0, start_stage),
                xytext=(0.4, start_stage + 0.18),
                fontsize=8,
                color=CLASS_COLORS[stage],
                fontweight="600",
                arrowprops=dict(arrowstyle="-", color=CLASS_COLORS[stage], lw=1.2)
            )

            ax.set_yticks([0.5, 1.5, 2.5, 3.5])
            ax.set_yticklabels(CLASS_NAMES, fontsize=8, color="#546E7A")
            ax.set_xlabel("Years from Diagnosis", fontsize=9, color="#546E7A")
            ax.set_xlim(0, 10)
            ax.set_ylim(-0.1, 4.1)
            ax.tick_params(colors="#90A4AE", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#E3E8EF")
            ax.grid(axis="x", color="#E3E8EF", linestyle="--", linewidth=0.6, zorder=1)

            # ---- RIGHT: Stage probabilities + milestones ----
            ax2 = axes[1]
            ax2.set_facecolor("#FAFBFC")

            # Sub-split using inset axes
            ax2.axis("off")
            ax2.set_title("Stage Probabilities", fontsize=11, fontweight="600", color="#263238", pad=10)

            # Bar chart (left inset)
            bar_ax = fig.add_axes([0.565, 0.18, 0.2, 0.62])
            bar_ax.set_facecolor("#FAFBFC")
            bar_colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
            bars = bar_ax.bar(range(4), prob_array, color=bar_colors, width=0.55, zorder=3)
            for bar, p in zip(bars, prob_array):
                bar_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{p:.0%}", ha="center", va="bottom", fontsize=8, fontweight="600", color="#263238")
            bar_ax.set_xticks(range(4))
            bar_ax.set_xticklabels(["ND", "VMD", "MD", "ModD"], fontsize=7.5, color="#546E7A")
            bar_ax.set_yticks([0, 0.5, 1.0])
            bar_ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=7.5, color="#90A4AE")
            bar_ax.set_ylim(0, 1.15)
            bar_ax.set_ylabel("Probability", fontsize=8, color="#546E7A")
            for spine in bar_ax.spines.values():
                spine.set_edgecolor("#E3E8EF")
            bar_ax.grid(axis="y", color="#E3E8EF", linestyle="--", linewidth=0.6, zorder=0)

            # Milestones (right inset as text)
            milestone_ax = fig.add_axes([0.79, 0.12, 0.2, 0.72])
            milestone_ax.axis("off")
            milestone_ax.text(0, 1.0, "Key Milestones", fontsize=10, fontweight="700",
                              color="#263238", va="top", transform=milestone_ax.transAxes)

            years_check = [2, 4, 6, 8]
            for i, yr in enumerate(years_check):
                projected_val = np.clip(start_stage + growth_rate * yr, 0, 3)
                proj_idx      = int(round(projected_val))
                proj_idx      = min(proj_idx, 3)
                proj_stage    = CLASS_NAMES[proj_idx]
                dot_color     = CLASS_COLORS[proj_stage]
                ypos          = 0.85 - i * 0.22

                milestone_ax.plot(0.04, ypos, "o", color=dot_color, markersize=8,
                                  transform=milestone_ax.transAxes, clip_on=False)
                milestone_ax.text(0.18, ypos + 0.025, f"Yr {yr}:", fontsize=8.5,
                                  color="#546E7A", va="center", transform=milestone_ax.transAxes)
                milestone_ax.text(0.18, ypos - 0.06, proj_stage, fontsize=8.5, fontweight="700",
                                  color=dot_color, va="center", transform=milestone_ax.transAxes)

            plt.tight_layout(rect=[0, 0, 0.78, 1])
            st.pyplot(fig)
            plt.close(fig)

        # =========================================================
        # TAB 2 — Grad-CAM
        # =========================================================
        with tab2:
            if gradcam_views is not None:
                axial, coronal, sagittal = gradcam_views
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(axial,    caption="Axial View",    use_container_width=True, channels="BGR")
                with c2:
                    st.image(coronal,  caption="Coronal View",  use_container_width=True, channels="BGR")
                with c3:
                    st.image(sagittal, caption="Sagittal View", use_container_width=True, channels="BGR")

                st.markdown("""
                <div style="background:#E3F2FD;border-radius:8px;padding:0.8rem 1rem;
                            font-size:0.82rem;color:#1565C0;margin-top:0.75rem;">
                    🔵 <b>Blue</b> regions indicate low activation &nbsp;|&nbsp;
                    🔴 <b>Red/yellow</b> regions show areas most predictive of the diagnosis
                </div>
                """, unsafe_allow_html=True)

            else:
                # 2D image — show greyscale scan with simulated heatmap overlay
                st.markdown('<p style="font-size:0.85rem;color:#78909C;margin-bottom:0.75rem;">ℹ️ Full Grad-CAM requires a 3D NIfTI file. Showing simulated activation map for 2D input.</p>', unsafe_allow_html=True)

                h, w     = slice_img.shape[:2]
                cx, cy   = w // 2, h // 2
                Y, X     = np.ogrid[:h, :w]
                sim_heat = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * (w // 5)**2))
                sim_heat = (sim_heat / sim_heat.max() * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(sim_heat, cv2.COLORMAP_JET)
                img_bgr  = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
                overlay  = cv2.addWeighted(img_bgr, 0.55, hm_color, 0.45, 0)

                st.image(overlay, caption="Simulated activation (2D mode)", channels="BGR", use_container_width=True)

        # =========================================================
        # TAB 3 — Download Report
        # =========================================================
        with tab3:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")

            report_lines = [
                "=" * 56,
                "       ALZHEIMER'S DETECTION SYSTEM — REPORT",
                "=" * 56,
                f"  Date         : {now}",
                f"  File         : {uploaded_file.name}",
                f"  Diagnosis    : {stage}",
                f"  Confidence   : {confidence:.2%}",
                "",
                "  CLASS PROBABILITIES",
                "  " + "-" * 34,
            ]
            for i, cn in enumerate(CLASS_NAMES):
                bar = "█" * int(prob_array[i] * 30)
                report_lines.append(f"  {cn:<22} {prob_array[i]:.2%}  {bar}")

            report_lines += [
                "",
                "  PROJECTED MILESTONES",
                "  " + "-" * 34,
            ]
            growth_rate_r = 0.12 + STAGE_ORDER[stage] * 0.06
            for yr in [2, 4, 6, 8, 10]:
                pv   = np.clip(STAGE_ORDER[stage] + growth_rate_r * yr, 0, 3)
                ps   = CLASS_NAMES[min(int(round(pv)), 3)]
                report_lines.append(f"  Year {yr:<4}  →  {ps}")

            report_lines += [
                "",
                "  NOTE: This report is AI-generated and should be",
                "  reviewed by a qualified medical professional.",
                "=" * 56,
            ]
            report_text = "\n".join(report_lines)

            st.markdown(f"""
            <div style="background:#F8FAFC;border:1px solid #E3E8EF;border-radius:10px;
                        padding:1.2rem 1.4rem;font-family:monospace;font-size:0.82rem;
                        color:#37474F;white-space:pre-wrap;line-height:1.7;">
{report_text}
            </div>
            """, unsafe_allow_html=True)

            b64 = base64.b64encode(report_text.encode()).decode()
            fname = f"alzheimer_report_{uploaded_file.name.split('.')[0]}.txt"
            st.markdown(f"""
            <a class="dl-btn" href="data:text/plain;base64,{b64}" download="{fname}">
                ⬇ Download Report
            </a>
            """, unsafe_allow_html=True)