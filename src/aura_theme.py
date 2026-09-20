"""
AbyssalScan HUD Theme
======================
Design tokens lifted from the provided AbyssalScan mockup (dark tactical
hydrographic bridge aesthetic: deep navy background, cyan/mint accents,
JetBrains Mono for telemetry, Inter for headlines). This module returns one
big CSS string injected once via st.markdown(unsafe_allow_html=True) to
restyle Streamlit's native widgets (sliders, buttons, tabs, containers,
metrics) to match, rather than fighting Streamlit's DOM by hand.
"""

COLORS = {
    "background": "#071327",
    "surface_lowest": "#030e22",
    "surface_low": "#101c30",
    "surface": "#142034",
    "surface_high": "#1f2a3f",
    "surface_highest": "#2a354a",
    "on_surface": "#d7e3fe",
    "on_surface_variant": "#b9cacb",
    "outline": "#849495",
    "primary": "#e0fdff",
    "primary_container": "#00f2fe",
    "on_primary_container": "#006a70",
    "secondary": "#7dffa2",
    "secondary_container": "#05e777",
    "tertiary_dim": "#ffba38",
    "error": "#ffb4ab",
}

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">

<style>
:root {{
    --bg: {COLORS['background']};
    --surf-lowest: {COLORS['surface_lowest']};
    --surf-low: {COLORS['surface_low']};
    --surf: {COLORS['surface']};
    --surf-high: {COLORS['surface_high']};
    --surf-highest: {COLORS['surface_highest']};
    --on-surf: {COLORS['on_surface']};
    --on-surf-var: {COLORS['on_surface_variant']};
    --outline: {COLORS['outline']};
    --primary: {COLORS['primary']};
    --primary-c: {COLORS['primary_container']};
    --on-primary-c: {COLORS['on_primary_container']};
    --secondary: {COLORS['secondary']};
    --secondary-c: {COLORS['secondary_container']};
    --tertiary: {COLORS['tertiary_dim']};
    --error: {COLORS['error']};
}}

html, body, .stApp {{
    background-color: var(--bg) !important;
    color: var(--on-surf) !important;
    font-family: 'Inter', sans-serif;
}}

/* headings */
h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; color: var(--primary) !important; }}

/* mono/telemetry text helper classes */
.mono {{ font-family: 'JetBrains Mono', monospace; }}
.label-caps {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--on-surf-var);
}}

/* top status header */
.aura-header {{
    display: flex; align-items: center; justify-content: space-between;
    background: var(--surf-lowest); border-radius: 6px; padding: 10px 18px;
    margin-bottom: 10px; box-shadow: 0 1px 8px rgba(0,0,0,0.4);
}}
.aura-title {{ font-size: 18px; font-weight: 700; color: var(--primary); letter-spacing: 0.04em; text-transform: uppercase; }}
.aura-sub {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--on-surf-var); letter-spacing: 0.08em; text-transform: uppercase; }}
.pill {{
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px;
    background: var(--surf-low); border-radius: 4px; font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.08em; color: var(--on-surf-var); margin-left: 8px;
}}
.pill b {{ color: var(--secondary); }}
.dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--secondary); display: inline-block; }}

/* telemetry metric cards */
.metric-card {{
    background: var(--surf-low); border-radius: 6px; padding: 12px 14px;
    display: flex; flex-direction: column; gap: 4px; height: 100%;
}}
.metric-label {{ display:flex; justify-content:space-between; align-items:center; }}
.metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: var(--primary); }}
.metric-sub {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--outline); text-transform: uppercase; }}
.badge {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 4px; }}
.badge-cyan {{ background: rgba(0,242,254,0.15); color: var(--primary-c); }}
.badge-amber {{ background: rgba(255,186,56,0.15); color: var(--tertiary); }}
.badge-green {{ background: rgba(5,231,119,0.15); color: var(--secondary); }}
.badge-red {{ background: rgba(255,180,171,0.18); color: var(--error); }}

/* detection cards */
.det-card {{
    background: var(--surf-lowest); border-radius: 6px; padding: 10px 12px;
    margin-bottom: 8px; border-left: 3px solid var(--surf-highest);
}}
.det-card.hazard-high {{ border-left-color: var(--error); }}
.det-card.hazard-med {{ border-left-color: var(--tertiary); }}
.det-card.anomaly {{ border-left-color: var(--secondary); }}
.det-title {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: var(--primary); }}
.det-row {{ display:flex; justify-content:space-between; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--outline); margin-top: 4px; text-transform: uppercase; }}
.det-geo {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--on-surf-var); margin-top: 4px; }}

/* section card headers inside bordered containers */
.card-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom: 8px; }}
.card-title {{ font-size: 14px; font-weight: 700; color: var(--primary); text-transform: uppercase; letter-spacing: 0.03em; }}

/* footer strip */
.aura-footer {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--on-surf-var);
    display: flex; justify-content: space-between; padding: 8px 16px;
    background: var(--surf-lowest); border-radius: 6px; margin-top: 10px;
}}
.aura-footer b {{ color: var(--on-surf); }}

/* restyle native widgets */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--surf-low) !important; border: 1px solid var(--surf-highest) !important;
    border-radius: 8px !important;
}}
.stSlider [data-baseweb="slider"] > div > div {{ background: var(--primary-c) !important; }}
.stButton button {{
    background: var(--surf-highest); color: var(--on-surf); border: none; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; font-size: 12px;
}}
.stButton button:hover {{ background: var(--primary-c); color: var(--on-primary-c); }}
.stDownloadButton button {{
    background: var(--primary-c) !important; color: var(--on-primary-c) !important; border: none; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;
}}
.stTabs [data-baseweb="tab"] {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; text-transform: uppercase; color: var(--on-surf-var); }}
.stTabs [aria-selected="true"] {{ color: var(--primary-c) !important; }}
section[data-testid="stSidebar"] {{ background: var(--surf-lowest) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--surf-lowest) !important; border: 1px dashed var(--surf-highest) !important; }}
</style>
"""

DAYLIGHT_CSS = """
<style>
:root {
    --day-bg: #f1f5f9;
    --day-surface: #ffffff;
    --day-low: #f8fafc;
    --day-container: #edf3f9;
    --day-high: #e2eaf3;
    --day-highest: #d8e3ef;
    --day-ink: #0f172a;
    --day-muted: #475569;
    --day-outline: #94a3b8;
    --day-blue: #0284c7;
    --day-deep: #0a2540;
    --day-teal: #0d9488;
    --day-amber: #d97706;
    --day-red: #dc2626;
}

html, body, .stApp {
    background-color: var(--day-bg) !important;
    background-image: radial-gradient(circle, rgba(2,132,199,0.08) 1px, transparent 1px) !important;
    background-size: 24px 24px !important;
    color: var(--day-ink) !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
}

h1, h2, h3, h4, .card-title, .aura-title {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--day-deep) !important;
}
.label-caps, .metric-sub, .det-row, .det-title, .aura-sub, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}
.aura-header {
    background: rgba(255,255,255,0.96) !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 0 !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.06) !important;
    padding: 10px 14px !important;
    margin: -1rem -1rem 0.5rem !important;
}
.aura-title { color: var(--day-deep) !important; letter-spacing: .06em !important; }
.aura-sub { color: #64748b !important; }
.pill { background: #f8fafc !important; border: 1px solid #e2e8f0 !important; color: #475569 !important; }
.metric-card {
    background: rgba(255,255,255,.92) !important;
    border: 1px solid #dbe5ef !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.05) !important;
}
.metric-value { color: var(--day-ink) !important; }
.metric-sub { color: #64748b !important; }
.badge-cyan { background: #e0f2fe !important; color: #0369a1 !important; }
.badge-amber { background: #fef3c7 !important; color: #92400e !important; }
.badge-green { background: #d1fae5 !important; color: #047857 !important; }
.badge-red { background: #fee2e2 !important; color: #b91c1c !important; }
.card-title { color: var(--day-ink) !important; letter-spacing: .04em !important; }
.det-card {
    background: rgba(248,250,252,.86) !important;
    border: 1px solid #dbe5ef !important;
    border-left-width: 3px !important;
    border-radius: 8px !important;
}
.det-title { color: var(--day-ink) !important; }
.det-row { color: #64748b !important; }
.aura-footer { background: rgba(255,255,255,.94) !important; color: #475569 !important; border: 1px solid #e2e8f0 !important; }
.aura-footer b { color: var(--day-ink) !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.94) !important;
    border: 1px solid #dbe5ef !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.05) !important;
}
.stButton button, .stDownloadButton button {
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid #cbd5e1 !important;
    background: #fff !important;
    color: #334155 !important;
}
.stButton button:hover { background: #e0f2fe !important; color: #0369a1 !important; border-color: #7dd3fc !important; }
.stDownloadButton button { background: linear-gradient(90deg,#0284c7,#0d9488) !important; color: white !important; border: none !important; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace !important; color: #475569 !important; }
.stTabs [aria-selected="true"] { color: #0369a1 !important; }
.stFileUploader, [data-testid="stFileUploaderDropzone"] { background: #f8fafc !important; }
.stFileUploader [data-testid="stFileUploaderDropzone"] { border: 1px dashed #7dd3fc !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--day-blue) !important; }
section[data-testid="stSidebar"] { background: #fff !important; }
@media (max-width: 1050px) {
    [style*="grid-template-columns:repeat(5,1fr)"] { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    [data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
}
@media (max-width: 700px) {
    [style*="grid-template-columns:repeat(5,1fr)"] { grid-template-columns: 1fr !important; }
    .aura-header { margin-left: -0.5rem !important; margin-right: -0.5rem !important; }
    .aura-title { font-size: 16px !important; }
    .aura-header > div:last-child { justify-content: flex-start !important; margin-top: 6px; }
}
</style>
"""


def inject():
    import streamlit as st
    st.markdown(CSS + DAYLIGHT_CSS, unsafe_allow_html=True)
