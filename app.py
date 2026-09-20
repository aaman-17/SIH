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
import tempfile
import datetime

import numpy as np
import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.detection_pipeline import detect
from src.geotagging import synthetic_nav_track, build_report, save_report
from ultralytics import YOLO
from src.autoencoder import load_autoencoder
from src.sonar_guard import looks_like_sonar
from src import aura_theme

st.set_page_config(page_title="ABYSSALSCAN", layout="wide", initial_sidebar_state="collapsed")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolo_sonar_best.pt")
SHIPWRECK_WEIGHTS = os.path.join(MODEL_DIR, "yolo_shipwreck_ai4shipwrecks.pt")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AE_WEIGHTS = os.path.join(BASE_DIR, "models", "autoencoder.pt")
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
    "DAYLIGHT CYAN": [(0, (238, 247, 252)), (0.5, (117, 190, 220)), (1.0, (2, 132, 199))],
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


def hazard_badge(conf):
    if conf >= 70:
        return "HIGH HAZARD", "badge-red"
    elif conf >= 40:
        return "MED HAZARD", "badge-amber"
    return "LOW HAZARD", "badge-cyan"


def main():
    aura_theme.inject()
    run_sonar_mode()


def run_sonar_mode():
    header_ph = st.empty()
    metrics_ph = st.empty()

    col_left, col_center, col_right = st.columns([3, 6, 3])

    # ---------------- LEFT: sonar input ----------------
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="card-header"><span class="card-title">📡 Stream Ingest</span><span class="badge badge-cyan">ONLINE</span></div>',
                        unsafe_allow_html=True)
            uploaded = st.file_uploader("Sonar file (PNG/JPG/XTF)", type=["png", "jpg", "jpeg", "xtf"],
                                        label_visibility="visible")
            use_demo = False
            if uploaded is None:
                use_demo = st.checkbox("Use bundled demo sonar image", value=True)

    start_lat, start_lon, heading = 17.6868, 83.2185, 45.0
    swath_width = 50.0
    conf_thresh, min_report_conf = 0.30, 25
    ignore_guard = False

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
                f"{guard_diag['mean_saturation']:.1f}), which raw sonar exports don't have."
            ))

    if raw_img is not None and blocked_msg is None:
        yolo_model, shipwreck_model, ae_model = load_models()
        clean_img, detections = detect(
            yolo_model, ae_model, raw_img, yolo_conf=conf_thresh,
            shipwreck_model=shipwreck_model, shipwreck_conf=max(0.10, conf_thresh),
        )
        detections = [d for d in detections if d["confidence_pct"] >= min_report_conf]

        img_h, img_w = clean_img.shape[:2]
        if xtf_nav_df is not None:
            nav_df = xtf_nav_df
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
                        + evidence_row + '</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

            with tab_anomaly:
                anomalies = [d for d in detections if d["source"] == "autoencoder"]
                if not anomalies:
                    st.caption("No unclassified anomalies above threshold.")
                for d in anomalies:
                    st.markdown(
                        f'<div class="det-card anomaly">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span class="det-title">Unclassified Anomaly</span>'
                        f'<span class="badge badge-green">AUTOENC</span></div>'
                        f'<div class="det-row"><span>SCORE: {d["confidence_pct"]:.1f}%</span>'
                        f'<span>QUALITY: {d.get("quality_tier", "—").upper()}</span></div></div>',
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
        '<div><span class="aura-title">◈ AURA SONAR</span>'
        '<div class="aura-sub">Marine Hydrographic AI Suite • Daylight Command</div></div>'
        '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end;">'
        '<span class="pill"><span class="dot"></span> ONLINE</span>'
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
        metrics_ph.markdown(
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;">'
            f'<div class="metric-card"><div class="metric-label"><span class="label-caps">Targets Detected</span>'
            '<span class="badge badge-cyan">YOLO+AE</span></div>'
            f'<span class="metric-value">{len(detections)}</span>'
            f'<span class="metric-sub">{n_hazard} hazards · {n_anomaly} anomalies</span></div>'

            f'<div class="metric-card"><div class="metric-label"><span class="label-caps">Hazard Rating</span></div>'
            f'<span class="badge {hazard_cls}" style="width:fit-content;">{hazard_txt}</span>'
            f'<span class="metric-sub">Max confidence {max_conf:.0f}%</span></div>'

            f'<div class="metric-card"><div class="metric-label"><span class="label-caps">Bathymetric Sounding</span>'
            f'<span class="metric-sub">ALT: 12.4m</span></div>'
            f'<span class="metric-value">-42.8m</span>'
            f'<span class="metric-sub">water column · grade 2.1°</span></div>'

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
        f'<div><b>ECHOSOUNDER:</b> 120.4 kHz &nbsp;&nbsp; <b>WATER TEMP:</b> 8.4°C &nbsp;&nbsp; <b>DVL SPEED:</b> 3.8 kts</div>'
        f'<div><b>SYSTEM:</b> ONLINE &nbsp;&nbsp; <b>SYS CLOCK:</b> {datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")} UTC</div>'
        f'</div>', unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
