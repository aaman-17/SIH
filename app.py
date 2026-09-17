"""
 AbyssalScan - Hydrographic Tactical Bridge (Streamlit implementation)
=======================================================================
Run with:  streamlit run app.py

Dark tactical-HUD dashboard styled after the AbyssalScan mockup, wired to
the REAL detection pipeline (preprocess -> YOLO11n -> autoencoder ->
noise filtering -> geotagging). Where the original mockup showed
placeholder/demo numbers (live FPS ticking on a video feed, fake sensor
readings like water temperature, a PDF/ROS2 export that has no backend
here), those are replaced with values this pipeline actually produces, or
left out rather than faked.
"""

import os
import sys
import time
import tempfile
import datetime

import numpy as np
import cv2
import pandas as pd
import streamlit as st
import torch
import ultralytics
import folium
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.detection_pipeline import detect
from src.geotagging import synthetic_nav_track, load_nav_track, build_report, save_report
from ultralytics import YOLO
from src.autoencoder import load_autoencoder
from src.sonar_guard import looks_like_sonar
from src.optical_detector import detect_optical
from src import aura_theme

st.set_page_config(page_title="ABYSSALSCAN", layout="wide", initial_sidebar_state="collapsed")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolo_sonar_best.pt")
SHIPWRECK_WEIGHTS = os.path.join(MODEL_DIR, "yolo_shipwreck_ai4shipwrecks.pt")
AE_WEIGHTS = os.path.join(MODEL_DIR, "autoencoder.pt")
DEMO_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "demo_sonar.png")

CLASS_COLORS = {
    "debris_net": (255, 60, 60),
    "pipe_cylinder": (60, 160, 255),
    "shipwreck_debris": (255, 200, 0),
    "shipwreck": (255, 120, 0),
    "shipwreck_candidate": (255, 120, 0),
    "unclassified_anomaly": (170, 60, 255),
}

PALETTES = {
    "DEEP CYAN": [(0, (3, 14, 34)), (0.5, (0, 90, 110)), (1.0, (0, 242, 254))],
    "SEPIA": [(0, (20, 12, 5)), (0.5, (110, 70, 30)), (1.0, (255, 222, 172))],
    "PHOSPHOR": [(0, (2, 10, 2)), (0.5, (10, 120, 40)), (1.0, (125, 255, 162))],
}


def build_lut(stops):
    """Builds a 256-entry BGR lookup table by linearly interpolating
    between color stops (used for the DEEP CYAN / SEPIA / PHOSPHOR sonar
    display palettes - purely cosmetic, applied only for display)."""
    xs = np.array([s[0] for s in stops]) * 255
    lut = np.zeros((256, 3), dtype=np.uint8)
    for ch in range(3):
        ys = np.array([s[1][ch] for s in stops])
        lut[:, ch] = np.interp(np.arange(256), xs, ys)
    return lut[:, ::-1]  # RGB stops -> BGR for OpenCV


def apply_palette(gray, palette_name):
    """Applies a cosmetic display palette (LUT) to a grayscale sonar image.
    Purely visual - detection runs on the underlying grayscale data."""
    lut = build_lut(PALETTES[palette_name])
    return lut[gray]


@st.cache_resource
def load_models():
    yolo_model = YOLO(YOLO_WEIGHTS if os.path.exists(YOLO_WEIGHTS) else "yolo11n.pt")
    shipwreck_model = YOLO(SHIPWRECK_WEIGHTS) if os.path.exists(SHIPWRECK_WEIGHTS) else None
    ae_model = load_autoencoder(AE_WEIGHTS)
    return yolo_model, shipwreck_model, ae_model


@st.cache_resource
def load_coco_yolo():
    return YOLO("yolo11n.pt")


def draw_overlay(color_img_bgr, detections):
    """color_img_bgr: HxWx3 BGR image (already palette-mapped)."""
    vis = color_img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = CLASS_COLORS.get(d["class"], (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class"]} {d["confidence_pct"]:.0f}%'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


OPTICAL_COLORS = {"possible_debris": (0, 90, 255), "plastic_bag_or_film": (255, 0, 170)}


def draw_optical_overlay(bgr_img, detections):
    vis = bgr_img.copy()
    scale = max(1, vis.shape[1] // 900)  # scale line/text with image size
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = OPTICAL_COLORS.get(d["class"], (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2 * scale)
        label = f'{d["class"]} {d["confidence_pct"]:.0f}%'
        if d.get("coco_hint"):
            label += f' (looks like: {d["coco_hint"]})'
        font_scale = 0.5 * scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, scale)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(vis, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), scale, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def hazard_badge(conf):
    if conf >= 70:
        return "HIGH HAZARD", "badge-red"
    elif conf >= 40:
        return "MED HAZARD", "badge-amber"
    return "LOW HAZARD", "badge-cyan"


def main():
    aura_theme.inject()

    mode = st.radio(
        "Input type", ["🔊 Side-scan sonar (acoustic)", "📷 Optical / camera photo"],
        horizontal=True, label_visibility="collapsed",
    )
    if mode.startswith("🔊"):
        run_sonar_mode()
    else:
        run_optical_mode()


def run_sonar_mode():
    header_ph = st.empty()
    metrics_ph = st.empty()

    col_left, col_center, col_right = st.columns([3, 6, 3])

    # ---------------- LEFT: controls ----------------
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="card-header"><span class="card-title">⚙ Inference Controls</span></div>',
                        unsafe_allow_html=True)
            conf_thresh = st.slider("YOLO Confidence Gate", 0.05, 0.95, 0.30, 0.01,
                                    help="Calibrated on the held-out eval set (reports/inference_calibration.json): 0.30 maximizes recall-weighted F2.")
            min_report_conf = st.slider("Min. Confidence to Report (%)", 0, 100, 25, 5)
            ignore_guard = st.checkbox("I know this is sonar — skip input check", value=False)
            st.selectbox("Target Weights / Backbone", [
                "YOLO11n-Maritime + AI4Shipwrecks shipwreck head",
                "YOLO11n-Maritime (debris baseline only)",
            ], help="The AI4Shipwrecks checkpoint is trained for the shipwreck class; the baseline remains active for debris classes.")

        with st.container(border=True):
            st.markdown('<div class="card-header"><span class="card-title">🧭 Nav & Geotagging</span></div>',
                        unsafe_allow_html=True)
            nav_mode = st.radio("Navigation source", ["Simulated tow-track (demo)", "Upload nav CSV"],
                                label_visibility="collapsed")
            nav_file = None
            if nav_mode == "Upload nav CSV":
                nav_file = st.file_uploader("Nav CSV (ping_index, lat, lon, heading_deg, altitude_m)", type=["csv"])
                start_lat, start_lon, heading = 17.6868, 83.2185, 45.0
            else:
                c1, c2 = st.columns(2)
                start_lat = c1.number_input("Start Lat", value=17.6868, format="%.6f")
                start_lon = c2.number_input("Start Lon", value=83.2185, format="%.6f")
                heading = st.number_input("Tow Heading (deg true)", value=45.0)
            swath_width = st.number_input("Sonar Swath Width (m)", value=50.0, min_value=5.0)

        with st.container(border=True):
            st.markdown('<div class="card-header"><span class="card-title">📡 Stream Ingest</span></div>',
                        unsafe_allow_html=True)
            uploaded = st.file_uploader("Sonar file (PNG/JPG/XTF)", type=["png", "jpg", "jpeg", "xtf"],
                                        label_visibility="collapsed")
            use_demo = False
            if uploaded is None:
                use_demo = st.checkbox("Use bundled demo sonar image", value=True)

    # ---------------- resolve input & run pipeline ----------------
    xtf_nav_df = None
    blocked_msg = None
    raw_img, source_name, clean_img, detections, records, nav_df = None, None, None, [], [], None

    if uploaded is not None and uploaded.name.lower().endswith(".xtf"):
        from src.xtf_loader import load_xtf
        with tempfile.NamedTemporaryFile(suffix=".xtf", delete=False) as tmp_xtf:
            tmp_xtf.write(uploaded.read())
            tmp_xtf_path = tmp_xtf.name
        raw_img, xtf_nav_df = load_xtf(tmp_xtf_path)
        source_name = uploaded.name
    elif uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_name = uploaded.name
    elif use_demo and os.path.exists(DEMO_IMG):
        raw_img = cv2.imread(DEMO_IMG)
        source_name = "demo_sonar.png"
    else:
        blocked_msg = ("info", "Upload a sonar image / XTF log to begin, or tick 'use bundled demo image'.")

    is_sonar, guard_diag = True, {}
    if raw_img is not None:
        is_sonar, guard_diag = looks_like_sonar(raw_img)
        if not is_sonar and not ignore_guard:
            blocked_msg = ("error", (
                "This doesn't look like acoustic side-scan sonar data — it has real color content "
                f"(channel variation ≈ {guard_diag['channel_decorrelation']:.1f}, mean saturation ≈ "
                f"{guard_diag['mean_saturation']:.1f}), which raw sonar exports don't have. Switch to "
                "**Optical / camera photo** mode above, or tick 'I know this is sonar' if your sonar "
                "software colorizes its output."
            ))

    latency_ms = 0.0
    if raw_img is not None and blocked_msg is None:
        yolo_model, shipwreck_model, ae_model = load_models()
        t0 = time.time()
        clean_img, detections = detect(
            yolo_model, ae_model, raw_img, yolo_conf=conf_thresh,
            shipwreck_model=shipwreck_model, shipwreck_conf=max(0.10, conf_thresh),
        )
        latency_ms = (time.time() - t0) * 1000
        detections = [d for d in detections if d["confidence_pct"] >= min_report_conf]

        img_h, img_w = clean_img.shape[:2]
        if xtf_nav_df is not None:
            nav_df = xtf_nav_df
        elif nav_mode == "Upload nav CSV" and nav_file is not None:
            nav_df = load_nav_track(nav_file)
        else:
            nav_df = synthetic_nav_track(n_pings=max(img_h, 10), start_lat=start_lat,
                                          start_lon=start_lon, heading_deg=heading)
        records = build_report(detections, nav_df, img_w, img_h, swath_width_m=swath_width, source_file=source_name)

    # ---------------- CENTER: sonar viewport ----------------
    with col_center:
        with st.container(border=True):
            hc1, hc2 = st.columns([3, 2])
            hc1.markdown('<div class="card-title">🌊 SONAR WATERFALL VIEWPORT</div>', unsafe_allow_html=True)
            palette = hc2.selectbox("Palette", list(PALETTES.keys()), label_visibility="collapsed")

            if blocked_msg:
                level, msg = blocked_msg
                getattr(st, level)(msg)
            else:
                color_img = apply_palette(clean_img, palette)
                overlay = draw_overlay(color_img, detections)
                st.image(overlay, width='stretch')
                legend = " &nbsp;&nbsp; ".join(
                    f'<span style="color:rgb{c}">■</span> {name}' for name, c in CLASS_COLORS.items()
                )
                st.markdown(legend, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="aura-footer" style="margin-top:6px;">'
                    f'<span>RES: <b>{clean_img.shape[1]}×{clean_img.shape[0]}px</b></span>'
                    f'<span>INFERENCE: <b>{latency_ms:.0f}ms</b></span>'
                    f'<span>SOURCE: <b>{source_name}</b></span>'
                    f'</div>', unsafe_allow_html=True,
                )

    # ---------------- RIGHT: detections feed + export ----------------
    with col_right:
        with st.container(border=True):
            n_hazard = sum(1 for d in detections if d["source"] in {"yolo", "structure_heuristic"})
            n_anomaly = sum(1 for d in detections if d["source"] == "autoencoder")
            st.markdown(
                f'<div class="card-header"><span class="card-title">Detections</span>'
                f'<span class="badge badge-cyan">{len(detections)} ACTIVE</span></div>',
                unsafe_allow_html=True,
            )
            tab_hazard, tab_anomaly = st.tabs([f"HAZARDS ({n_hazard})", f"ANOMALIES ({n_anomaly})"])

            with tab_hazard:
                hazards = [d for d in detections if d["source"] in {"yolo", "structure_heuristic"}]
                if not hazards:
                    st.caption("No classified hazards above threshold.")
                for i, d in enumerate(hazards):
                    label, badge_cls = ("SHIPWRECK CANDIDATE", "badge-red") if d["source"] == "structure_heuristic" else hazard_badge(d["confidence_pct"])
                    rec = records[detections.index(d)] if detections else None
                    geo = f'{rec["latitude"]:.4f}°N, {rec["longitude"]:.4f}°E' if rec else "—"
                    css_cls = "hazard-high" if d["confidence_pct"] >= 70 else "hazard-med" if d["confidence_pct"] >= 40 else ""
                    evidence_row = (f'<div class="det-row"><span>EVIDENCE: {d.get("evidence_count", "—")} tiled hits</span></div>'
                                    if d["source"] == "structure_heuristic" else "")
                    card_html = (
                        f'<div class="det-card {css_cls}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span class="det-title">{d["class"]}</span>'
                        f'<span class="badge {badge_cls}">{label}</span></div>'
                        f'<div class="det-row"><span>CONF: {d["confidence_pct"]:.1f}%</span>'
                        f'<span>QUALITY: {d.get("quality_tier", "—").upper()} {d.get("quality_score", 0):.0f}</span></div>'
                        f'<div class="det-row"><span>PERSISTENCE: {d.get("persistence_count", 1)} frame(s)</span>'
                        f'<span>BOX: {d["box"][2]-d["box"][0]}×{d["box"][3]-d["box"][1]}px</span></div>'
                        + evidence_row +
                        f'<div class="det-geo">{geo}</div></div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

            with tab_anomaly:
                anomalies = [d for d in detections if d["source"] == "autoencoder"]
                if not anomalies:
                    st.caption("No unclassified anomalies above threshold.")
                for d in anomalies:
                    rec = records[detections.index(d)] if detections else None
                    geo = f'{rec["latitude"]:.4f}°N, {rec["longitude"]:.4f}°E' if rec else "—"
                    st.markdown(
                        f'<div class="det-card anomaly">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span class="det-title">Unclassified Anomaly</span>'
                        f'<span class="badge badge-green">AUTOENC</span></div>'
                        f'<div class="det-row"><span>SCORE: {d["confidence_pct"]:.1f}%</span>'
                        f'<span>QUALITY: {d.get("quality_tier", "—").upper()}</span></div>'
                        f'<div class="det-geo">{geo}</div></div>',
                        unsafe_allow_html=True,
                    )

        with st.container(border=True):
            st.markdown('<div class="card-header"><span class="card-title">⬇ Mission Export</span></div>',
                        unsafe_allow_html=True)
            if records:
                df = pd.DataFrame(records).drop(columns=["bbox_px"])
                with st.expander("Preview report table"):
                    st.dataframe(df, width='stretch')
                with tempfile.TemporaryDirectory() as tmp:
                    json_path, csv_path = save_report(records, tmp)
                    with open(json_path, "rb") as f:
                        st.download_button("⬇ JSON REPORT", f, file_name="anomaly_report.json",
                                            mime="application/json", width='stretch')
                    with open(csv_path, "rb") as f:
                        st.download_button("⬇ CSV REPORT", f, file_name="anomaly_report.csv",
                                            mime="text/csv", width='stretch')
            else:
                st.caption("Nothing to export yet.")

    # ---------------- fill header + metrics now that we have data ----------------
    header_ph.markdown(
        '<div class="aura-header">'
        '<div><span class="aura-title">🛰 ABYSSALSCAN</span>'
        '<div class="aura-sub">Marine Hydrographic AI Suite — Streamlit Build</div></div>'
        '<div>'
        '<span class="pill"><span class="dot"></span>YOLO11n: <b>ACTIVE</b></span>'
        '<span class="pill">AUTOENCODER: <b>ACTIVE</b></span>'
        f'<span class="pill">LATENCY: <b>{latency_ms:.0f}ms</b></span>'
        '</div></div>', unsafe_allow_html=True,
    )

    if detections or nav_df is not None:
        max_conf = max([d["confidence_pct"] for d in detections], default=0)
        if max_conf >= 70:
            hazard_txt, hazard_cls = "HIGH RISK", "badge-red"
        elif max_conf >= 40:
            hazard_txt, hazard_cls = "MODERATE RISK", "badge-amber"
        else:
            hazard_txt, hazard_cls = "LOW RISK", "badge-green"
        first_lat = nav_df["lat"].iloc[0] if nav_df is not None else 0
        first_lon = nav_df["lon"].iloc[0] if nav_df is not None else 0
        alt = nav_df["altitude_m"].mean() if nav_df is not None and "altitude_m" in nav_df else float("nan")
        n_pings = len(nav_df) if nav_df is not None else 0

        metrics_ph.markdown(
            '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:10px;">'
            f'<div class="metric-card"><div class="metric-label"><span class="label-caps">Targets Detected</span>'
            f'<span class="badge badge-cyan">YOLO+AE</span></div>'
            f'<span class="metric-value">{len(detections)}</span>'
            f'<span class="metric-sub">{n_hazard} hazards · {n_anomaly} anomalies</span></div>'

            f'<div class="metric-card"><div class="metric-label"><span class="label-caps">Hazard Rating</span></div>'
            f'<span class="badge {hazard_cls}" style="width:fit-content;">{hazard_txt}</span>'
            f'<span class="metric-sub">Max confidence {max_conf:.0f}%</span></div>'

            f'<div class="metric-card"><div class="metric-label"><span class="label-caps">Tow-Fish Altitude</span></div>'
            f'<span class="metric-value">{alt:.1f}m</span>'
            f'<span class="metric-sub">above seabed (nav track)</span></div>'

            f'<div class="metric-card"><div class="metric-label"><span class="label-caps">Tow-Track Fix</span></div>'
            f'<span class="metric-value" style="font-size:14px;">{first_lat:.4f}°N, {first_lon:.4f}°E</span>'
            f'<span class="metric-sub">{n_pings} pings · HDG {heading:.1f}°</span></div>'

            f'<div class="metric-card"><div class="metric-label"><span class="label-caps">Neural Pipeline</span></div>'
            f'<span class="metric-value" style="font-size:14px;">YOLO11n + Autoenc</span>'
            f'<span class="metric-sub">{latency_ms:.0f}ms / frame</span></div>'
            '</div>', unsafe_allow_html=True,
        )
    else:
        metrics_ph.markdown(
            '<div class="metric-card" style="margin-bottom:10px;">'
            '<span class="label-caps">SYSTEM STATUS</span>'
            '<span class="metric-value" style="font-size:16px;">AWAITING INPUT</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="aura-footer">'
        f'<div><b>ULTRALYTICS:</b> {ultralytics.__version__} &nbsp;&nbsp; <b>TORCH:</b> {torch.__version__}</div>'
        f'<div><b>SWATH:</b> {swath_width:.0f}m &nbsp;&nbsp; <b>SYS CLOCK:</b> {datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")} UTC</div>'
        f'</div>', unsafe_allow_html=True,
    )


def run_optical_mode():
    st.caption("Optical underwater/surface photo → generic salient-object triage + plastic-film heuristic")
    st.warning(
        "**Experimental / assistive only.** No real-world annotated underwater-litter dataset "
        "was downloadable in this build environment (TACO/UAVVaste host images on Flickr, which "
        "isn't reachable here). This mode uses a pretrained general-purpose detector as a "
        "*generic foreign-object flagger* (its literal object-class guesses are unreliable "
        "underwater, so they're shown only as hints) plus a classical brightness/shape heuristic "
        "for plastic bags/film. Treat results as a first-pass triage aid, not ground truth.",
        icon="ℹ️",
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        yolo_conf = st.slider("Detector sensitivity", 0.02, 0.5, 0.05, 0.01,
                               help="Lower = more candidate boxes (more false positives too).")
        min_report_conf = st.slider("Minimum confidence to report (%)", 0, 100, 30, 5)

    with st.container(border=True):
        uploaded = st.file_uploader("Upload an underwater/surface photo", type=["png", "jpg", "jpeg"])
        use_demo = False
        if uploaded is None:
            use_demo = st.checkbox("Use bundled demo sonar image instead (will look odd here)", value=False)

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_name = uploaded.name
    elif use_demo and os.path.exists(DEMO_IMG):
        raw_img = cv2.imread(DEMO_IMG)
        source_name = "demo_sonar.png"
    else:
        st.info("Upload a photo to begin.")
        return

    coco_model = load_coco_yolo()
    with st.spinner("Running optical triage..."):
        detections = detect_optical(coco_model, raw_img, yolo_conf=yolo_conf)
        detections = [d for d in detections if d["confidence_pct"] >= min_report_conf]

    col1, col2 = st.columns([1.3, 1])
    with col1:
        with st.container(border=True):
            st.markdown('<div class="card-title">🔍 Detections Overlay</div>', unsafe_allow_html=True)
            overlay = draw_optical_overlay(raw_img, detections)
            st.image(overlay, width='stretch')
            legend = " &nbsp;&nbsp; ".join(
                f'<span style="color:rgb{c[2]},{c[1]},{c[0]}">■</span> {name}' for name, c in OPTICAL_COLORS.items()
            )
            st.markdown(legend, unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="card-title">📊 Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="metric-value">{len(detections)}</span> '
                        f'<span class="metric-sub">candidate objects flagged</span>', unsafe_allow_html=True)
            if detections:
                by_class = pd.Series([d["class"] for d in detections]).value_counts()
                st.bar_chart(by_class)

    with st.container(border=True):
        st.markdown('<div class="card-title">📄 Detections Table</div>', unsafe_allow_html=True)
        if detections:
            df = pd.DataFrame(detections)
            st.dataframe(df, width='stretch')
            with tempfile.TemporaryDirectory() as tmp:
                import json as _json
                path = os.path.join(tmp, "optical_detections.json")
                with open(path, "w") as f:
                    _json.dump(detections, f, indent=2)
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download JSON", f, file_name="optical_detections.json",
                                        mime="application/json")
        else:
            st.info("Nothing above the reporting threshold - try lowering the detector sensitivity.")


if __name__ == "__main__":
    main()
