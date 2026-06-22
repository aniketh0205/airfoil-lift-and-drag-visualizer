import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import io

st.set_page_config(page_title="Airfoil Lift & Drag Visualizer", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

AIR_VISCOSITY = 1.81e-5

# ---------------------------------------------------------------------------
# Aerodynamic Formulas (matches React's aerodynamics.js exactly)
# ---------------------------------------------------------------------------

def reynolds_number(velocity, chord_length, air_density):
    return (air_density * velocity * chord_length) / AIR_VISCOSITY

def calculate_aerodynamics(aoa, stall_angle, zero_lift_angle, max_camber, max_thickness, velocity, chord_length, air_density):
    Re = reynolds_number(velocity, chord_length, air_density)
    cl_alpha = 0.1

    stall_reduction = 4.0 * math.exp(-Re / 180000)
    effective_stall = max(stall_angle - stall_reduction, 6)

    re_lift_factor = 1 - 0.30 * math.exp(-Re / 70000) - 0.12 * math.exp(-Re / 250000)
    re_drag_factor = 1 + 0.45 * math.exp(-Re / 55000) + 0.18 * math.exp(-Re / 200000)

    CL = cl_alpha * (aoa - zero_lift_angle) * re_lift_factor

    stall_warning = False
    if aoa > effective_stall:
        stall_warning = True
        excess_aoa = aoa - effective_stall
        CL = cl_alpha * (effective_stall - zero_lift_angle) * re_lift_factor * math.exp(-0.1 * excess_aoa)

    cf = 0.074 / (Re ** 0.2) if Re > 0 else 0.004
    base_drag = 0.002 + cf
    induced_drag = 0.01 * CL * CL
    thickness_drag = max_thickness * 0.05
    CD = (base_drag + induced_drag + thickness_drag) * re_drag_factor

    if aoa > 12:
        CD *= (1 + (aoa - 12) * 0.05)
    if aoa > effective_stall:
        CD *= (1 + (aoa - effective_stall) * 0.15)

    lift_per_meter = 0.5 * air_density * velocity ** 2 * chord_length * CL
    drag_per_meter = 0.5 * air_density * velocity ** 2 * chord_length * CD
    ld_ratio = CL / CD if CD > 0 else 0

    return {
        "CL": round(CL, 4),
        "CD": round(CD, 4),
        "liftPerMeter": round(lift_per_meter, 2),
        "dragPerMeter": round(drag_per_meter, 2),
        "LD": round(ld_ratio, 2),
        "Re": round(Re),
        "stallWarning": stall_warning,
        "effectiveStall": round(effective_stall, 1)
    }

def generate_explanation(aoa, CL, CD, ld_ratio, stall_warning, max_camber, zero_lift_angle, effective_stall, Re):
    abs_camber = abs(max_camber)

    if Re < 50000:
        re_note = f" Reynolds number is {Re:,} — very low. Viscous forces dominate; laminar separation bubbles reduce lift and increase drag significantly."
    elif Re < 100000:
        re_note = f" Reynolds number is {Re:,} (low). Lift is reduced and drag is elevated."
    elif Re < 500000:
        re_note = f" Reynolds number is {Re:,} (moderate)."
    elif Re < 5000000:
        re_note = f" Reynolds number is {Re:,} (typical for general aviation)."
    else:
        re_note = f" Reynolds number is {Re:,} (high)."

    if stall_warning:
        return (f"At {aoa}° angle of attack, airflow starts separating from the upper surface. "
                f"Lift reduces and drag increases rapidly — this is stall. The critical stall angle "
                f"is approximately {effective_stall}° at the current Reynolds number.{re_note}")

    if aoa < 3:
        expl = f"At {aoa}° angle of attack, the airfoil produces low lift. "
    elif aoa < 8:
        expl = f"At {aoa}° angle of attack, the airfoil deflects air downward creating a pressure difference. This increases lift. "
    elif aoa < 12:
        expl = f"At {aoa}° angle of attack, lift continues to increase but drag also grows faster. "
    else:
        expl = f"At {aoa}° angle of attack, the airfoil is approaching stall. Drag increases rapidly. "

    if abs_camber > 0.005:
        expl += f"This is a cambered airfoil (zero-lift angle: {zero_lift_angle}°). "
    else:
        expl += "This is a symmetric airfoil — zero lift at 0° AoA. "

    if ld_ratio > 30:
        expl += f"L/D ratio is {ld_ratio} — excellent efficiency."
    elif ld_ratio > 15:
        expl += f"L/D ratio is {ld_ratio} — moderate efficiency."
    else:
        expl += f"L/D ratio is {ld_ratio} — drag is relatively high."

    return expl + re_note

# ---------------------------------------------------------------------------
# NACA 4-Digit Generator
# ---------------------------------------------------------------------------

def generate_naca_coords(camber_pct, pos_pct, thickness_pct, num_points=100):
    """Generate NACA 4-digit coordinates. Inputs are percentages (e.g., 2, 40, 12 for NACA 2412)."""
    m = camber_pct / 100.0
    p = pos_pct / 100.0
    t = thickness_pct / 100.0
    upper, lower = [], []
    for i in range(num_points + 1):
        x = i / num_points
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x)
        if m == 0:
            yc, dyc = 0, 0
        elif x < p:
            yc = (m / (p * p)) * (2 * p * x - x * x)
            dyc = (2 * m / (p * p)) * (p - x)
        else:
            yc = (m / ((1 - p) * (1 - p))) * (1 - 2 * p + 2 * p * x - x * x)
            dyc = (2 * m / ((1 - p) * (1 - p))) * (p - x)
        theta = math.atan(dyc)
        upper.append((round(x - yt * math.sin(theta), 4), round(yc + yt * math.cos(theta), 4)))
        lower.insert(0, (round(x + yt * math.sin(theta), 4), round(yc - yt * math.cos(theta), 4)))
    return upper + lower

def naca_name_from_digits(m, p, t):
    return f"NACA {m}{p}{str(t).zfill(2)}"

# ---------------------------------------------------------------------------
# Predefined Airfoils
# ---------------------------------------------------------------------------

def build_airfoils():
    return {
        "NACA 0012": {
            "type": "Symmetric", "maxThickness": 0.12, "maxCamber": 0.0, "zeroLiftAngle": 0, "stallAngle": 14,
            "description": "Symmetric airfoil, 12% thickness. Zero lift at 0° AoA.",
            "coords": generate_naca_coords(0, 0, 12)
        },
        "NACA 2412": {
            "type": "Mildly cambered", "maxThickness": 0.12, "maxCamber": 0.02, "zeroLiftAngle": -1.6, "stallAngle": 14,
            "description": "2% camber at 40% chord, 12% thickness. General-purpose airfoil.",
            "coords": generate_naca_coords(2, 40, 12)
        },
        "NACA 4412": {
            "type": "Moderately cambered", "maxThickness": 0.12, "maxCamber": 0.04, "zeroLiftAngle": -3.2, "stallAngle": 14,
            "description": "4% camber at 40% chord, 12% thickness. High lift at low speeds.",
            "coords": generate_naca_coords(4, 40, 12)
        },
        "Clark Y": {
            "type": "Semi-symmetric", "maxThickness": 0.117, "maxCamber": 0.034, "zeroLiftAngle": -2.7, "stallAngle": 14,
            "description": "Classic flat-bottom airfoil, 11.7% thickness. General aviation.",
            "coords": [
                (1.0, 0.0013),(0.95, 0.0074),(0.90, 0.0126),(0.80, 0.0214),(0.70, 0.0286),(0.60, 0.0347),
                (0.50, 0.0398),(0.40, 0.0433),(0.30, 0.0435),(0.25, 0.0416),(0.20, 0.0375),(0.15, 0.0308),
                (0.10, 0.0218),(0.075, 0.0163),(0.05, 0.0105),(0.025, 0.0052),(0.0125, 0.0026),(0.0, 0.0),
                (0.0125, -0.0029),(0.025, -0.0058),(0.05, -0.0113),(0.075, -0.0166),(0.10, -0.0216),
                (0.15, -0.0305),(0.20, -0.0376),(0.25, -0.0421),(0.30, -0.0440),(0.40, -0.0440),
                (0.50, -0.0401),(0.60, -0.0349),(0.70, -0.0288),(0.80, -0.0215),(0.90, -0.0127),(0.95, -0.0075),(1.0, -0.0013)
            ]
        },
        "Selig S1223": {
            "type": "High-lift", "maxThickness": 0.086, "maxCamber": 0.048, "zeroLiftAngle": -3.8, "stallAngle": 13,
            "description": "High-lift low-Re airfoil, 8.6% thickness, 4.8% camber.",
            "coords": generate_naca_coords(4, 40, 9)
        },
        "Eppler E205": {
            "type": "Laminar flow", "maxThickness": 0.11, "maxCamber": 0.035, "zeroLiftAngle": -2.8, "stallAngle": 14,
            "description": "Low-drag laminar flow airfoil, 11% thickness, 3.5% camber.",
            "coords": generate_naca_coords(3, 40, 11)
        }
    }

def get_airfoils():
    predefined = st.session_state.get("predefined_airfoils")
    if predefined is None:
        st.session_state["predefined_airfoils"] = build_airfoils()
        predefined = st.session_state["predefined_airfoils"]
    custom = st.session_state.get("custom_airfoils", {})
    all_foils = dict(predefined)
    all_foils.update(custom)
    return all_foils, predefined, custom

# ---------------------------------------------------------------------------
# Geometry extraction from coordinates
# ---------------------------------------------------------------------------

def interpolate_y(points, target_x):
    pts = sorted(points, key=lambda p: p[0])
    if len(pts) < 2:
        return 0
    if target_x <= pts[0][0]:
        return pts[0][1]
    if target_x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if x1 <= target_x <= x2:
            t = (target_x - x1) / (x2 - x1) if x2 != x1 else 0
            return y1 + t * (y2 - y1)
    return 0

def extract_geometry(coords):
    upper = [(x, y) for x, y in coords if y >= -1e-12]
    lower = [(x, y) for x, y in coords if y < -1e-12]
    upper.sort(key=lambda p: p[0])
    lower.sort(key=lambda p: p[0])

    max_thickness, max_camber = 0, 0
    for i in range(51):
        x = i / 50
        yu = interpolate_y(upper, x)
        yl = interpolate_y(lower, x)
        thickness = yu - yl
        camber = (yu + yl) / 2
        if thickness > max_thickness:
            max_thickness = thickness
        if abs(camber) > abs(max_camber):
            max_camber = camber

    abs_camber = abs(max_camber)
    if abs_camber < 0.005:
        stall = min(15, 14 + max_thickness * 100 * -0.15 + 0.5)
    elif abs_camber < 0.03:
        stall = min(16, 15 + max_thickness * 100 * -0.1 + 0.5)
    else:
        stall = 13
    stall = max(8, min(20, round(stall * 10) / 10))

    zero_lift = -max_camber * 80
    return {"maxThickness": round(max_thickness, 4), "maxCamber": round(max_camber, 4),
            "stallAngle": stall, "zeroLiftAngle": round(zero_lift, 2)}

def parse_coordinates(text):
    lines = text.strip().split("\n")
    coords = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for sep in [None, ",", "\t"]:
            try:
                parts = line.split() if sep is None else line.split(sep)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    if 0 <= x <= 1:
                        coords.append((x, y))
                        break
            except ValueError:
                continue
    if len(coords) < 5:
        return None, "Need at least 5 valid coordinate pairs."
    return coords, None

# ---------------------------------------------------------------------------
# Chart style
# ---------------------------------------------------------------------------

def style_plot(fig, ax, title, xlabel, ylabel, dark=False):
    if dark:
        bg = "#0b1225"
        text_c = "#8899b4"
        grid_c = "#1e2d48"
        face_c = "#0f1420"
        ax.set_facecolor(face_c)
        fig.patch.set_facecolor(bg)
        ax.tick_params(colors=text_c, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid_c)
        ax.spines["bottom"].set_color(grid_c)
        ax.set_title(title, fontsize=12, fontweight="600", color="#eef2f8", pad=12)
        ax.set_xlabel(xlabel, fontsize=10, color=text_c)
        ax.set_ylabel(ylabel, fontsize=10, color=text_c)
        ax.grid(True, linestyle="--", alpha=0.15, color="#3b82f6")
        ax.legend(fontsize=9, labelcolor=text_c, framealpha=0.3, loc="best")
    else:
        ax.set_facecolor("#fafbfc")
        fig.patch.set_facecolor("#ffffff")
        ax.set_title(title, fontsize=12, fontweight="600", color="#2d3748", pad=12)
        ax.set_xlabel(xlabel, fontsize=10, color="#4a5568")
        ax.set_ylabel(ylabel, fontsize=10, color="#4a5568")
        ax.grid(True, linestyle="--", alpha=0.4, color="#a0aec0")
        ax.tick_params(colors="#4a5568", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e0")
        ax.spines["bottom"].set_color("#cbd5e0")
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
<style>
    /* Dark theme base */
    .stApp { background: #060a16; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
    h1, h2, h3 { color: #eef2f8 !important; }
    h1 { background: linear-gradient(90deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 800 !important; }
    p, label, .stMarkdown, .stText, span { color: #8899b4; }
    .stSidebar { background: #0b1225 !important; }
    .stSidebar .stMarkdown, .stSidebar label, .stSidebar span { color: #cbd5e0; }
    .stRadio label { color: #cbd5e0 !important; }
    .stSelectbox > div > div { background: #111b30; border-color: #1e2d48; color: #eef2f8; border-radius: 8px; }
    .stSlider > div > div > div { background: #111b30; }
    .stButton > button { background: linear-gradient(135deg, #3b82f6, #6366f1); color: white; border: none; border-radius: 8px; padding: 0.4rem 1.5rem; font-weight: 600; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(59,130,246,0.3); }
    .stNumberInput input, .stTextInput input, .stTextArea textarea { background: #111b30; border: 1px solid #1e2d48; color: #eef2f8; border-radius: 8px; }
    .stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus { border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,0.2); }
    .stMetric { background: linear-gradient(135deg, #111b30, #0f1420); border: 1px solid #1e2d48; border-radius: 12px; padding: 1rem; }
    .stMetric label { color: #8899b4 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #eef2f8 !important; }
    hr { border-color: #1e2d48; }
    .st-expander { background: #111b30; border: 1px solid #1e2d48; border-radius: 12px; }
    .stAlert { background: #111b30; border: 1px solid #1e2d48; border-radius: 12px; color: #eef2f8; }
    .stTabs [data-baseweb="tab-list"] { background: #0b1225; border-radius: 8px; gap: 0; }
    .stTabs [data-baseweb="tab"] { color: #8899b4; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #3b82f6; }
    .stDataFrame { background: #111b30; border: 1px solid #1e2d48; border-radius: 12px; }
    /* Custom card class */
    .card { background: #111b30; border: 1px solid #1e2d48; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .card-blue { border-left: 4px solid #3b82f6; }
    .card-green { border-left: 4px solid #22c55e; }
    .card-red { border-left: 4px solid #ef4444; }
    .card-orange { border-left: 4px solid #f97316; }
    .card-purple { border-left: 4px solid #a855f7; }
    .st-badge { background: rgba(59,130,246,0.15); color: #60a5fa; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    div[data-testid="stNotification"] { background: #111b30; border: 1px solid #1e2d48; }
</style>
"""

# ---------------------------------------------------------------------------
# SVG Airfoil Visualizer (matplotlib)
# ---------------------------------------------------------------------------

def draw_airfoil(coords, aoa=0, stall=False, width=6, height=3):
    if not coords or len(coords) < 3:
        return None
    fig, ax = plt.subplots(figsize=(width, height), facecolor="#0b1225")
    ax.set_facecolor("#0f1420")

    pts = np.array(coords)
    cx, cy = 0.5, 0
    rad = math.radians(max(-30, min(30, aoa)))
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def rot(x, y):
        dx, dy = x - cx, y - cy
        return dx * cos_a - dy * sin_a + cx, dx * sin_a + dy * cos_a + cy

    rpts = np.array([rot(x, y) for x, y in coords])

    ax.plot(rpts[:, 0], rpts[:, 1], "-", color="#93c5fd", linewidth=2)
    ax.fill(rpts[:, 0], rpts[:, 1], color="#3b82f6", alpha=0.15)

    ax.axhline(0, color="#475569", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    margin = max(0.15, abs(max(rpts[:, 1])) * 1.3)
    ax.set_ylim(-margin, margin)
    ax.axis("off")
    fig.patch.set_facecolor("#0b1225")

    if stall:
        ax.text(0.5, margin * 0.7, "⚠ STALL — Flow Separated", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#ef4444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a0a0a", edgecolor="#ef4444", alpha=0.9))

    if aoa != 0:
        ax.annotate("", xy=(0.08, 0), xytext=(0.22, 0),
                    arrowprops=dict(arrowstyle="->", color="#60a5fa", lw=2))
        ax.text(0.15, -margin * 1.15, f"Wind  α={aoa}°", ha="center", fontsize=9, color="#60a5fa", fontweight="bold")

    return fig

def draw_anatomy():
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0b1225")
    ax.set_facecolor("#0f1420")

    m, p, t = 0.04, 0.4, 0.15
    xs = np.linspace(0, 1, 80)
    yt = 5 * t * (0.2969 * np.sqrt(xs) - 0.1260 * xs - 0.3516 * xs**2 + 0.2843 * xs**3 - 0.1015 * xs**4)

    camber_mask = xs < p
    yc = np.where(camber_mask,
                  (m / p**2) * (2 * p * xs - xs**2),
                  (m / (1 - p)**2) * (1 - 2 * p + 2 * p * xs - xs**2))
    dyc = np.where(camber_mask,
                   (2 * m / p**2) * (p - xs),
                   (2 * m / (1 - p)**2) * (p - xs))
    theta = np.arctan(dyc)
    yu = yc + yt * np.cos(theta)
    yl = yc - yt * np.cos(theta)

    # Upper surface
    ax.fill_between(xs, 0, yu, color="#3b82f6", alpha=0.12)
    ax.plot(xs, yu, "-", color="#60a5fa", linewidth=2, label="Upper Surface")

    # Lower surface
    ax.fill_between(xs, yl, 0, color="#ef4444", alpha=0.12)
    ax.plot(xs, yl, "-", color="#f87171", linewidth=2, label="Lower Surface")

    # Camber line
    ax.plot(xs, yc, "--", color="#4ade80", linewidth=1.5, label="Camber Line")

    # Chord line
    ax.axhline(0, color="#60a5fa", linestyle="--", linewidth=1, alpha=0.5, label="Chord Line")

    # Max thickness arrow
    max_t_idx = np.argmax(yt * 2)
    mx = xs[max_t_idx]
    ax.annotate("", xy=(mx, yu[max_t_idx]), xytext=(mx, yl[max_t_idx]),
                arrowprops=dict(arrowstyle="<->", color="#fbbf24", lw=2))
    ax.text(mx + 0.03, 0, f"Max Thickness\n{t*100:.0f}%", fontsize=8, color="#fbbf24", fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    margin = 0.22
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0b1225")
    return fig

def draw_naca_svg(camber_pct, pos_pct, thick_pct, width=6, height=3):
    coords = generate_naca_coords(camber_pct, pos_pct, thick_pct, 60)
    fig, ax = plt.subplots(figsize=(width, height), facecolor="#0b1225")
    ax.set_facecolor("#0f1420")
    pts = np.array(coords)
    ax.plot(pts[:, 0], pts[:, 1], "-", color="#1a237e", linewidth=2)
    ax.fill(pts[:, 0], pts[:, 1], color="#42a5f5", alpha=0.12)
    ax.axhline(0, color="#334155", linestyle="--", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    margin = max(0.16, abs(max(pts[:, 1])) * 1.3)
    ax.set_ylim(-margin, margin)
    ax.axis("off")
    fig.patch.set_facecolor("#0b1225")
    return fig

# ---------------------------------------------------------------------------
# Quiz questions (matches React)
# ---------------------------------------------------------------------------

QUIZ = [
    {"q": "What happens to lift when velocity increases?", "opts": ["Lift decreases", "Lift increases proportionally to V²", "Lift stays the same", "Lift increases proportionally to V"], "ans": 1},
    {"q": "Which force acts opposite to the direction of aircraft motion?", "opts": ["Lift", "Thrust", "Drag", "Weight"], "ans": 2},
    {"q": "What is angle of attack?", "opts": ["Angle between wing chord line and relative wind", "Angle of aircraft relative to ground", "Angle of tail relative to fuselage", "Angle between upper and lower wing surfaces"], "ans": 0},
    {"q": "What happens to lift after the stall angle is exceeded?", "opts": ["Lift continues to increase", "Lift remains constant", "Lift decreases rapidly", "Lift becomes zero instantly"], "ans": 2},
    {"q": "Why do cambered airfoils generate lift at low angle of attack?", "opts": ["They are lighter than symmetric airfoils", "Curved upper surface creates pressure difference even at 0° AoA", "They have lower thickness", "They create more drag which increases lift"], "ans": 1},
    {"q": "What does Reynolds number represent?", "opts": ["Ratio of lift to drag", "Ratio of inertial to viscous forces in the flow", "The speed of sound in air", "The density of the airfoil material"], "ans": 1},
    {"q": "What effect does a low Reynolds number have on an airfoil?", "opts": ["Lift increases and drag decreases", "Laminar separation bubbles reduce lift and increase drag", "The stall angle increases significantly", "The airfoil becomes more efficient"], "ans": 1},
    {"q": "What is the zero-lift angle of attack?", "opts": ["The angle at which drag is zero", "The angle at which the airfoil produces no lift", "The angle at which stall occurs", "The angle between the upper and lower camber line"], "ans": 1},
    {"q": "For a symmetric airfoil like NACA 0012, the zero-lift angle is:", "opts": ["0°", "About -2°", "About 5°", "Depends on thickness"], "ans": 0},
    {"q": "What is the chord line of an airfoil?", "opts": ["The line of maximum thickness", "Straight line connecting leading and trailing edges", "The curved upper surface", "The line along the camber at 50% chord"], "ans": 1},
    {"q": "What does the lift-to-drag ratio (L/D) measure?", "opts": ["How fast the aircraft can fly", "Aerodynamic efficiency — lift per unit drag", "The structural strength of the wing", "The stall speed of the aircraft"], "ans": 1},
    {"q": "What is the main difference between NACA 0012 and NACA 2412?", "opts": ["0012 is thicker", "2412 has 2% camber while 0012 is symmetric", "2412 has a higher stall angle", "0012 is cambered and 2412 is symmetric"], "ans": 1},
    {"q": "What is the drag polar?", "opts": ["Graph showing lift coefficient vs drag coefficient", "Distribution of pressure on the airfoil surface", "Angle at which minimum drag occurs", "Polar coordinate representation of the airfoil shape"], "ans": 0},
    {"q": "In the standard atmosphere at sea level, air density is approximately:", "opts": ["0.5 kg/m³", "1.225 kg/m³", "2.5 kg/m³", "10 kg/m³"], "ans": 1},
    {"q": "What causes induced drag?", "opts": ["Friction between air and wing surface", "Generation of lift — trailing vortices deflect airflow", "The thickness of the airfoil", "Compressibility effects at high speed"], "ans": 1}
]

THEORY_TOPICS = [
    {"title": "Airfoil", "icon": "✈️", "content": "An airfoil is the cross-sectional shape of a wing, blade, or sail designed to generate lift when moving through air.",
     "color": "blue"},
    {"title": "Lift", "icon": "⬆️", "content": "Lift is the upward force opposing weight. It is generated by the pressure difference between the upper and lower surfaces.",
     "formula": "L' = 0.5 × ρ × V² × c × CL", "color": "green"},
    {"title": "Drag", "icon": "⬅️", "content": "Drag is the force opposing forward motion, caused by air resistance and the creation of lift (induced drag).",
     "formula": "D' = 0.5 × ρ × V² × c × CD", "color": "orange"},
    {"title": "Angle of Attack", "icon": "📐", "content": "The angle between the chord line and the relative wind. Increasing AoA increases lift up to the stall point.",
     "color": "purple"},
    {"title": "Stall", "icon": "⚠️", "content": "Occurs when AoA exceeds a critical value — airflow separates from the upper surface, lift drops, drag increases.",
     "color": "red"},
    {"title": "Lift Coefficient (CL)", "icon": "📊", "content": "A dimensionless number relating lift to fluid density, velocity, and chord length.",
     "formula": "CL = 0.1 × (AoA − α₀)", "color": "blue"},
    {"title": "Drag Coefficient (CD)", "icon": "📉", "content": "A dimensionless number quantifying drag resistance, including skin friction and induced components.",
     "color": "amber"},
    {"title": "L/D Ratio", "icon": "🎯", "content": "Measure of aerodynamic efficiency. Higher L/D means more lift per unit drag. Gliders have very high L/D ratios.",
     "color": "teal"},
    {"title": "Camber", "icon": "🔄", "content": "The curvature of an airfoil. Camber helps generate lift even at zero AoA by creating a pressure difference.",
     "color": "indigo"},
    {"title": "Chord Line", "icon": "📏", "content": "Straight line connecting leading edge to trailing edge. The chord length is a reference dimension for all calculations.",
     "color": "cyan"},
    {"title": "Pressure Difference", "icon": "🔽🔼", "content": "Lift is created by the pressure difference: lower pressure above (faster flow), higher pressure below (slower flow).",
     "color": "violet"},
    {"title": "Flow Separation", "icon": "🌊", "content": "Boundary layer detaching from the surface at high AoA, leading to stall, increased drag, and decreased lift.",
     "color": "rose"}
]

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home():
    st.markdown("<h1 style='text-align:center;font-size:2.5rem;'>✈️ Interactive Airfoil Lift & Drag Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1.1rem;color:#8899b4;'>Explore how airfoil shape, velocity, and angle of attack affect lift and drag through interactive visualization.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Start Simulation", use_container_width=True):
            st.session_state["page"] = "Simulator"
            st.rerun()
    with col2:
        if st.button("📐 Custom Airfoil", use_container_width=True):
            st.session_state["page"] = "Custom Airfoil"
            st.rerun()
    with col3:
        if st.button("📊 Compare", use_container_width=True):
            st.session_state["page"] = "Compare"
            st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Understanding Aerodynamics</h2>", unsafe_allow_html=True)

    features = [
        ("⬆️", "Lift", "Upward force generated by pressure difference between upper and lower surfaces."),
        ("⬅️", "Drag", "Rearward force opposing motion, caused by air resistance and flow separation."),
        ("📐", "Angle of Attack", "Angle between wing chord line and relative wind — critical for lift generation."),
        ("🚀", "Velocity", "Speed relative to air. Lift and drag increase with the square of velocity (V²)."),
        ("✈️", "Airfoil Shape", "Cross-sectional shape determining aerodynamic characteristics. Camber and thickness are key.")
    ]
    cols = st.columns([1, 1, 1, 1, 1])
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 5]:
            st.markdown(f"<div class='card'><div style='font-size:2rem;text-align:center;'>{icon}</div><h3 style='text-align:center;color:#3b82f6;'>{title}</h3><p style='text-align:center;font-size:0.85rem;'>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h2>About This Project</h2><p>This interactive educational tool helps students understand how lift and drag are produced on different airfoils by changing input parameters and seeing real-time visual results.</p><p style='font-size:0.85rem;color:#5a6f8a;'>⚠ Uses simplified aerodynamic approximations for educational visualization. Results are not CFD-accurate.</p></div>", unsafe_allow_html=True)


def page_simulator():
    st.markdown("<h1>Aerodynamic Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p>Adjust parameters to see how lift and drag change in real time</p>", unsafe_allow_html=True)

    all_foils, _, custom = get_airfoils()
    foil_names = sorted(all_foils.keys())

    use_custom = st.checkbox("Use custom airfoil", key="sim_use_custom")
    source = custom if use_custom else all_foils
    source_names = sorted(source.keys())

    if not source_names:
        st.warning("No airfoils available. Please add custom airfoils or disable custom mode.")
        return

    sel_name = st.selectbox("Select Airfoil", source_names if source_names else foil_names, key="sim_foil")
    foil = all_foils[sel_name]

    with st.expander("Airfoil Properties", expanded=False):
        cola, colb, colc = st.columns(3)
        cola.metric("Max Thickness", f"{foil['maxThickness']*100:.1f}%")
        colb.metric("Max Camber", f"{foil['maxCamber']*100:.2f}%")
        colc.metric("Stall Angle", f"{foil['stallAngle']}°")

    c1, c2 = st.columns([1, 1])
    with c1:
        aoa = st.slider("Angle of Attack (°)", -5.0, 20.0, 5.0, 0.5)
        velocity = st.slider("Velocity (m/s)", 5.0, 100.0, 30.0, 1.0)
    with c2:
        air_density = st.number_input("Air Density (kg/m³)", 0.1, 2.0, 1.225, 0.001, format="%.3f")
        chord = st.slider("Chord Length (m)", 0.1, 2.0, 1.0, 0.05)

    result = calculate_aerodynamics(aoa, foil["stallAngle"], foil["zeroLiftAngle"],
                                     foil["maxCamber"], foil["maxThickness"],
                                     velocity, chord, air_density)

    coords = foil.get("coords")
    if coords:
        fig = draw_airfoil(coords, aoa, result["stallWarning"])
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    if result["stallWarning"]:
        st.error(f"⚠ STALL WARNING: Angle of attack exceeds the stall angle ({result['effectiveStall']}°). Lift decreases, drag increases rapidly.")

    st.markdown("### Results (per meter of wing span)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CL", result["CL"])
    col2.metric("CD", result["CD"])
    col3.metric("L/D", result["LD"])
    col4.metric("Reynolds #", f"{result['Re']:,}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Lift/m", f"{result['liftPerMeter']:.2f} N/m")
    col6.metric("Drag/m", f"{result['dragPerMeter']:.2f} N/m")

    wing_span = st.slider("Wing Span (m)", 1.0, 50.0, 10.0, 0.5, key="sim_span")
    total_lift = result["liftPerMeter"] * wing_span
    total_drag = result["dragPerMeter"] * wing_span
    col7.metric("Total Lift", f"{total_lift:.2f} N")
    col8.metric("Total Drag", f"{total_drag:.2f} N")

    # Explanation
    expl = generate_explanation(aoa, result["CL"], result["CD"], result["LD"],
                                 result["stallWarning"], foil["maxCamber"],
                                 foil["zeroLiftAngle"], result["effectiveStall"], result["Re"])
    st.info(expl)

    # Charts
    st.markdown("### Aerodynamic Characteristic Curves")
    alpha_range = np.linspace(-5, 20, 101)
    CLs = [calculate_aerodynamics(a, foil["stallAngle"], foil["zeroLiftAngle"],
                                   foil["maxCamber"], foil["maxThickness"],
                                   velocity, chord, air_density)["CL"] for a in alpha_range]
    CDs = [calculate_aerodynamics(a, foil["stallAngle"], foil["zeroLiftAngle"],
                                   foil["maxCamber"], foil["maxThickness"],
                                   velocity, chord, air_density)["CD"] for a in alpha_range]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CL vs AoA", "CD vs AoA", "Drag Polar", "L/D vs AoA", "Velocity"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(alpha_range, CLs, "-", color="#3b82f6", linewidth=2, label=sel_name)
        ax.axvline(x=foil["stallAngle"], color="#ef4444", linestyle="--", alpha=0.7, label=f"Stall ({foil['stallAngle']}°)")
        ax.axvline(x=foil["zeroLiftAngle"], color="#64748b", linestyle=":", alpha=0.5, label=f"α₀={foil['zeroLiftAngle']}°")
        ax.plot(aoa, result["CL"], "r*", markersize=15, zorder=5, label="Operating Point")
        style_plot(fig, ax, "CL vs Angle of Attack", "AoA (°)", "CL", dark=True)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(alpha_range, CDs, "-", color="#ef4444", linewidth=2, label=sel_name)
        ax.axvline(x=12, color="#a855f7", linestyle="--", alpha=0.7, label="High drag onset (12°)")
        ax.plot(aoa, result["CD"], "r*", markersize=15, zorder=5, label="Operating Point")
        style_plot(fig, ax, "CD vs Angle of Attack", "AoA (°)", "CD", dark=True)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(CDs, CLs, "-", color="#22c55e", linewidth=2, label=sel_name)
        ax.plot(result["CD"], result["CL"], "r*", markersize=15, zorder=5, label="Operating Point")
        style_plot(fig, ax, "Drag Polar (CL vs CD)", "CD", "CL", dark=True)
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        LDs = [CLs[i] / CDs[i] if CDs[i] > 0 else 0 for i in range(len(CLs))]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(alpha_range, LDs, "-", color="#a855f7", linewidth=2, label=sel_name)
        ax.plot(aoa, result["LD"], "r*", markersize=15, zorder=5, label="Operating Point")
        style_plot(fig, ax, "L/D Ratio vs Angle of Attack", "AoA (°)", "L/D", dark=True)
        st.pyplot(fig)
        plt.close(fig)

    with tab5:
        vel_range = np.arange(5, 101, 5)
        lifts_v = []
        drags_v = []
        for v in vel_range:
            r = calculate_aerodynamics(aoa, foil["stallAngle"], foil["zeroLiftAngle"],
                                        foil["maxCamber"], foil["maxThickness"],
                                        v, chord, air_density)
            lifts_v.append(r["liftPerMeter"])
            drags_v.append(r["dragPerMeter"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.plot(vel_range, lifts_v, "-", color="#22c55e", linewidth=2)
        ax1.plot(velocity, result["liftPerMeter"], "r*", markersize=12)
        style_plot(fig, ax1, "Velocity vs Lift", "Velocity (m/s)", "Lift (N/m)", dark=True)
        ax2.plot(vel_range, drags_v, "-", color="#f97316", linewidth=2)
        ax2.plot(velocity, result["dragPerMeter"], "r*", markersize=12)
        style_plot(fig, ax2, "Velocity vs Drag", "Velocity (m/s)", "Drag (N/m)", dark=True)
        st.pyplot(fig)
        plt.close(fig)


def page_custom():
    st.markdown("<h1>Custom Airfoil Upload</h1>", unsafe_allow_html=True)
    st.markdown("<p>Upload your own airfoil coordinates or paste them to analyze</p>", unsafe_allow_html=True)

    if "custom_airfoils" not in st.session_state:
        st.session_state["custom_airfoils"] = {}

    tab = st.radio("Input Method", ["Paste Coordinates", "Upload File", "My Library"], horizontal=True, label_visibility="collapsed")

    result = None
    if tab == "Paste Coordinates":
        name = st.text_input("Airfoil Name (optional)", placeholder="My Custom Airfoil")
        coord_text = st.text_area("Paste Coordinates", height=200,
            placeholder="NACA 2412\n1.0000 0.0013\n0.9500 0.0074\n0.9000 0.0126\n...")
        if st.button("Process Airfoil", type="primary"):
            if coord_text.strip():
                coords, err = parse_coordinates(coord_text)
                if err:
                    st.error(err)
                else:
                    geo = extract_geometry(coords)
                    foil_name = name.strip() or "Custom Airfoil"
                    result = {"name": foil_name, "coords": coords, "geometry": geo}
                    st.success(f"Parsed {len(coords)} coordinate points!")

    elif tab == "Upload File":
        uploaded = st.file_uploader("Choose a .dat, .txt, or .csv file", type=["dat", "txt", "csv"])
        name = st.text_input("Airfoil Name (optional)", placeholder="Airfoil name")
        if uploaded is not None:
            content = uploaded.read().decode("utf-8").strip()
            coords, err = parse_coordinates(content)
            if err:
                st.error(err)
            else:
                geo = extract_geometry(coords)
                foil_name = name.strip() or uploaded.name.rsplit(".", 1)[0]
                result = {"name": foil_name, "coords": coords, "geometry": geo}
                st.success(f"Loaded {len(coords)} coordinate points from {uploaded.name}")

    else:
        saved = st.session_state.get("custom_airfoils", {})
        if not saved:
            st.info("No saved airfoils yet. Paste or upload one first.")
        else:
            for fname, fdata in list(saved.items()):
                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 1])
                    c1.write(f"**{fname}** — {fdata.get('description', 'Custom airfoil')}")
                    if c2.button("Load", key=f"load_{fname}"):
                        result = fdata
                    if c3.button("Delete", key=f"del_{fname}"):
                        del st.session_state["custom_airfoils"][fname]
                        st.rerun()

    if result:
        geo = result["geometry"]
        st.markdown("### Geometry Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Thickness", f"{geo['maxThickness']*100:.1f}%")
        col2.metric("Max Camber", f"{geo['maxCamber']*100:.2f}%")
        col3.metric("Stall Angle (est.)", f"{geo['stallAngle']}°")
        col4.metric("Zero-Lift α", f"{geo['zeroLiftAngle']}°")

        fig = draw_airfoil(result["coords"], 0)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

        if st.button("💾 Save to Library"):
            st.session_state["custom_airfoils"][result["name"]] = {
                "type": "Custom",
                "maxThickness": geo["maxThickness"],
                "maxCamber": geo["maxCamber"],
                "zeroLiftAngle": geo["zeroLiftAngle"],
                "stallAngle": geo["stallAngle"],
                "description": "Custom uploaded airfoil",
                "coords": result["coords"]
            }
            st.success(f"✅ {result['name']} saved!")
            st.balloons()


def page_compare():
    st.markdown("<h1>Compare Airfoils</h1>", unsafe_allow_html=True)
    st.markdown("<p>Compare aerodynamic performance between two airfoils side by side</p>", unsafe_allow_html=True)

    all_foils, _, custom = get_airfoils()
    foil_names = sorted(all_foils.keys())
    custom_names = sorted(custom.keys()) if custom else []

    use_custom_a = st.checkbox("Use custom for Airfoil A", key="cmp_custom_a")
    use_custom_b = st.checkbox("Use custom for Airfoil B", key="cmp_custom_b")

    opt_a = custom_names if use_custom_a else foil_names
    opt_b = custom_names if use_custom_b else foil_names

    if not opt_a or not opt_b:
        st.warning("Not enough airfoils available for comparison.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        foil_a_name = st.selectbox("Airfoil A", opt_a, key="cmp_a")
    with c2:
        foil_b_name = st.selectbox("Airfoil B", opt_b, key="cmp_b")
    with c3:
        pass

    col1, col2, col3 = st.columns(3)
    aoa = col1.number_input("AoA (°)", -5.0, 20.0, 5.0, 0.5, key="cmp_aoa")
    vel = col2.number_input("Velocity (m/s)", 5.0, 100.0, 30.0, 1.0, key="cmp_vel")
    chord = col3.number_input("Chord (m)", 0.1, 2.0, 1.0, 0.05, key="cmp_chord")

    foil_a = all_foils[foil_a_name]
    foil_b = all_foils[foil_b_name]

    res_a = calculate_aerodynamics(aoa, foil_a["stallAngle"], foil_a["zeroLiftAngle"],
                                    foil_a["maxCamber"], foil_a["maxThickness"],
                                    vel, chord, 1.225)
    res_b = calculate_aerodynamics(aoa, foil_b["stallAngle"], foil_b["zeroLiftAngle"],
                                    foil_b["maxCamber"], foil_b["maxThickness"],
                                    vel, chord, 1.225)

    c1, c2 = st.columns(2)
    with c1:
        fig = draw_airfoil(foil_a.get("coords"), aoa, res_a["stallWarning"])
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    with c2:
        fig = draw_airfoil(foil_b.get("coords"), aoa, res_b["stallWarning"])
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("### Geometry Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{foil_a_name}**")
        st.write(f"Max Thickness: {foil_a['maxThickness']*100:.1f}%")
        st.write(f"Max Camber: {foil_a['maxCamber']*100:.2f}%")
        st.write(f"Stall Angle: {foil_a['stallAngle']}°")
    with col2:
        st.markdown(f"**{foil_b_name}**")
        st.write(f"Max Thickness: {foil_b['maxThickness']*100:.1f}%")
        st.write(f"Max Camber: {foil_b['maxCamber']*100:.2f}%")
        st.write(f"Stall Angle: {foil_b['stallAngle']}°")

    st.markdown("### Performance Comparison")
    metrics = [
        ("Lift Coefficient (CL)", "CL", res_a["CL"], res_b["CL"]),
        ("Drag Coefficient (CD)", "CD", res_a["CD"], res_b["CD"]),
        ("Lift per meter (N/m)", "liftPerMeter", res_a["liftPerMeter"], res_b["liftPerMeter"]),
        ("Drag per meter (N/m)", "dragPerMeter", res_a["dragPerMeter"], res_b["dragPerMeter"]),
        ("L/D Ratio", "LD", res_a["LD"], res_b["LD"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [m[0] for m in metrics]
    vals_a = [m[2] for m in metrics]
    vals_b = [m[3] for m in metrics]
    x = np.arange(len(labels))
    w = 0.35
    bars_a = ax.bar(x - w/2, vals_a, w, label=foil_a_name, color="#3b82f6", alpha=0.85)
    bars_b = ax.bar(x + w/2, vals_b, w, label=foil_b_name, color="#ef4444", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    style_plot(fig, ax, "Performance Comparison", "", "Value", dark=True)
    st.pyplot(fig)
    plt.close(fig)

    data = []
    for label, key, va, vb in metrics:
        diff = ((vb - va) / abs(va) * 100) if va != 0 else 0
        data.append({"Metric": label, foil_a_name: f"{va:.4f}", foil_b_name: f"{vb:.4f}", "Difference": f"{diff:+.1f}%"})
    st.table(data)


def page_learn():
    st.markdown("<h1>Learn Aerodynamics</h1>", unsafe_allow_html=True)
    st.markdown("<p>Student-friendly explanations of key aerodynamic concepts</p>", unsafe_allow_html=True)

    tab = st.radio("", ["📖 Theory", "📐 Nomenclature & Anatomy"], horizontal=True, label_visibility="collapsed")

    if tab == "📖 Theory":
        cols = st.columns(2)
        for i, topic in enumerate(THEORY_TOPICS):
            with cols[i % 2]:
                st.markdown(f"<div class='card card-{topic['color']}'><h3>{topic['icon']} {topic['title']}</h3><p>{topic['content']}</p></div>", unsafe_allow_html=True)
                if topic.get("formula"):
                    st.code(topic["formula"])

    else:
        st.markdown("### Airfoil Anatomy")
        fig = draw_anatomy()
        if fig:
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("""
        <div style='display:flex;gap:1rem;flex-wrap:wrap;justify-content:center;margin-bottom:1.5rem;'>
            <span style='color:#60a5fa;font-size:0.85rem;'><span style='color:#3b82f6;'>●</span> Upper Surface</span>
            <span style='color:#f87171;font-size:0.85rem;'><span style='color:#ef4444;'>●</span> Lower Surface</span>
            <span style='color:#4ade80;font-size:0.85rem;'>━ Camber Line</span>
            <span style='color:#60a5fa;font-size:0.85rem;'>- - Chord Line</span>
            <span style='color:#fbbf24;font-size:0.85rem;'>↕ Max Thickness</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### NACA 4-Digit Generator")
        col1, col2 = st.columns([1, 1])

        with col1:
            camber_pct = st.slider("Camber (%)", 0, 9, 2, 1)
            position_pct = st.slider("Position of max camber (%)", 10, 60, 40, 5)
            thickness_pct = st.slider("Max thickness (%)", 6, 30, 12, 1)

            m_digit = camber_pct
            p_digit = position_pct // 10
            t_str = str(thickness_pct).zfill(2)
            naca_name_str = f"NACA {m_digit}{p_digit}{t_str}"
            st.markdown(f"<h2 style='text-align:center;color:#3b82f6;font-family:monospace;'>{naca_name_str}</h2>", unsafe_allow_html=True)

            desc_parts = []
            if camber_pct == 0:
                desc_parts.append("Symmetric (no camber)")
            else:
                desc_parts.append(f"{camber_pct}% max camber at {position_pct}% chord")
            desc_parts.append(f"{thickness_pct}% max thickness")
            st.markdown(f"<p style='text-align:center;color:#5a6f8a;'>{' — '.join(desc_parts)}</p>", unsafe_allow_html=True)

            if st.button("Use in Simulator", type="primary"):
                coords = generate_naca_coords(camber_pct, position_pct, thickness_pct, 60)
                geo = extract_geometry(coords)
                st.session_state["custom_airfoils"][naca_name_str] = {
                    "type": "NACA Generated",
                    "maxThickness": geo["maxThickness"],
                    "maxCamber": geo["maxCamber"],
                    "zeroLiftAngle": geo["zeroLiftAngle"],
                    "stallAngle": geo["stallAngle"],
                    "description": f"NACA 4-digit generated: {naca_name_str}",
                    "coords": coords
                }
                st.success(f"✅ {naca_name_str} added to custom airfoils!")
                st.balloons()

        with col2:
            fig = draw_naca_svg(camber_pct, position_pct, thickness_pct)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

        with st.expander("How NACA 4-digit naming works"):
            st.markdown(f"""
            **NACA MPXX**
            - **M** (1st digit) = **Camber** (% of chord) — {m_digit} in current airfoil
            - **P** (2nd digit) = **Position** of max camber (tenths of chord) — {int(p_digit)} = {position_pct}% chord
            - **XX** (3rd & 4th digits) = **Max thickness** (% of chord) — {t_str} = {thickness_pct}%

            Examples:
            - NACA 0012 → 0% camber (symmetric), 12% thickness
            - NACA 2412 → 2% camber at 40% chord, 12% thickness
            - NACA 4412 → 4% camber at 40% chord, 12% thickness
            """)


def page_quiz():
    st.markdown("<h1>Knowledge Quiz</h1>", unsafe_allow_html=True)
    st.markdown("<p>Test your understanding of aerodynamics concepts</p>", unsafe_allow_html=True)

    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0

    if st.session_state["quiz_submitted"]:
        score = st.session_state["quiz_score"]
        total = len(QUIZ)
        if score >= 13:
            emoji, msg = "🎉", "Perfect score! Excellent understanding!"
        elif score >= 10:
            emoji, msg = "👍", "Great job!"
        elif score >= 7:
            emoji, msg = "📚", "Good effort! Review the Learn page to improve."
        else:
            emoji, msg = "📖", "Keep studying! Review the Learn page for more information."

        st.markdown(f"<div class='card' style='text-align:center;'><div style='font-size:3rem;'>{emoji}</div><h2>{score} / {total}</h2><p>{msg}</p></div>", unsafe_allow_html=True)
        if st.button("Retry Quiz"):
            st.session_state["quiz_answers"] = {}
            st.session_state["quiz_submitted"] = False
            st.rerun()

    for i, q in enumerate(QUIZ):
        st.markdown(f"<div class='card'><h3>Q{i+1}. {q['q']}</h3></div>", unsafe_allow_html=True)
        opts = q["opts"]
        for j, opt in enumerate(opts):
            label = f"{chr(65+j)}. {opt}"
            if st.session_state["quiz_submitted"]:
                correct = j == q["ans"]
                selected = st.session_state["quiz_answers"].get(i) == j
                if correct and selected:
                    st.success(f"✅ {label}")
                elif correct:
                    st.success(f"✅ {label}")
                elif selected:
                    st.error(f"❌ {label}")
                else:
                    st.markdown(f"<p style='color:#5a6f8a;'>{label}</p>", unsafe_allow_html=True)
            else:
                if st.button(label, key=f"q_{i}_{j}", use_container_width=True):
                    st.session_state["quiz_answers"][i] = j
                    st.rerun()

    if not st.session_state["quiz_submitted"]:
        answered = len(st.session_state["quiz_answers"])
        total = len(QUIZ)
        disabled = answered < total
        if st.button(f"Submit Quiz ({answered}/{total})" if disabled else "Submit Quiz", disabled=disabled, type="primary"):
            score = sum(1 for i in range(total) if st.session_state["quiz_answers"].get(i) == QUIZ[i]["ans"])
            st.session_state["quiz_score"] = score
            st.session_state["quiz_submitted"] = True
            st.rerun()


def page_about():
    st.markdown("<h1 style='text-align:center;'>✈️ Interactive Airfoil Lift & Drag Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>An interdisciplinary engineering education project</p>", unsafe_allow_html=True)

    sections = [
        ("Problem Statement", None, [
            "Students often find it difficult to understand lift and drag using only formulas and static diagrams.",
            "This project provides an interactive visual platform to understand how airfoil geometry, velocity, and angle of attack affect aerodynamic forces."
        ]),
        ("Objectives", None, [
            "To visualize lift and drag generation on airfoils",
            "To allow users to change aerodynamic parameters interactively",
            "To support predefined and custom airfoil coordinate files",
            "To calculate lift, drag, Cl, Cd, and L/D ratio",
            "To help students understand stall and pressure difference",
            "To compare different airfoil shapes"
        ]),
        ("Methodology", "This application uses simplified aerodynamic models based on thin airfoil theory and empirical approximations. Aerodynamic coefficients are calculated using educational formulas that capture the essential physics without requiring CFD-level computation.", None),
        ("Interdisciplinary Connection", "This project bridges Aeronautical Engineering and Computer Science. Aeronautical engineering provides the physics of lift, drag, and airfoil design. Computer science provides interactive visualization and real-time data processing.", None),
        ("Applications", None, ["Aerospace engineering education", "Student projects and demonstrations", "Understanding airfoil selection for aircraft design", "Preliminary aerodynamic analysis", "STEM outreach and teaching"]),
        ("Advantages", None, ["Interactive real-time visualization", "Supports both predefined and custom airfoils", "No installation required — runs in a web browser", "Educational explanations for every condition"]),
        ("Limitations", None, ["Uses simplified aerodynamic approximations, not CFD", "Not suitable for real aircraft design without validation", "Does not account for compressibility effects at high Mach numbers", "Limited to 2D airfoil analysis (no 3D wing effects)"]),
        ("Future Scope", None, ["Add real CFD visualization using panel methods", "Integrate with XFOIL for more accurate analysis", "Add 3D wing visualization and wingtip effects", "Include Reynolds number and Mach number effects"])
    ]

    for title, content, items in sections:
        st.markdown(f"<div class='card'><h3>{title}</h3>", unsafe_allow_html=True)
        if content:
            st.write(content)
        if items:
            for item in items:
                st.markdown(f"<p style='color:#3b82f6;'>▸</p><p>{item}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div style='text-align:center;padding:1rem 0;'><h1 style='font-size:2.5rem;margin:0;'>✈️</h1><h2 style='color:#eef2f8;font-size:1rem;'>Airfoil Analyzer</h2></div>", unsafe_allow_html=True)
        st.divider()

        nav_items = ["🏠 Home", "⚡ Simulator", "📐 Custom Airfoil", "📊 Compare", "📖 Learn", "❓ Quiz", "ℹ️ About"]
        page = st.session_state.get("page", "🏠 Home")
        idx = 0
        if page in nav_items:
            idx = nav_items.index(page)

        selected = st.radio("", nav_items, index=idx, label_visibility="collapsed")
        st.session_state["page"] = selected

        st.divider()
        all_foils, predefined, custom = get_airfoils()
        st.markdown(f"<div style='background:rgba(255,255,255,0.05);padding:0.75rem;border-radius:8px;'><p style='color:#5a6f8a;font-size:0.75rem;'>AIRFOILS</p><p style='color:#eef2f8;font-size:0.85rem;margin:0;'>📦 Predefined: {len(predefined)}<br>✨ Custom: {len(custom)}</p></div>", unsafe_allow_html=True)
        st.divider()
        st.caption("Built with Streamlit, NumPy & Matplotlib.\nEducational aerodynamics simulator.")

    page = st.session_state["page"]
    if "Home" in page:
        page_home()
    elif "Simulator" in page:
        page_simulator()
    elif "Custom" in page:
        page_custom()
    elif "Compare" in page:
        page_compare()
    elif "Learn" in page:
        page_learn()
    elif "Quiz" in page:
        page_quiz()
    elif "About" in page:
        page_about()

if __name__ == "__main__":
    main()
