import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title="Airfoil Lift & Drag Visualizer", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

AIR_VISCOSITY = 1.81e-5

# ---------------------------------------------------------------------------
# Aerodynamic Formulas (exact match to React's aerodynamics.js)
# ---------------------------------------------------------------------------

def calculate_aerodynamics(aoa, stall_angle, zero_lift_angle, max_camber, max_thickness, velocity, chord_length, air_density):
    Re = (air_density * velocity * chord_length) / AIR_VISCOSITY
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
    return {"CL": round(CL, 4), "CD": round(CD, 4), "liftPerMeter": round(lift_per_meter, 2),
            "dragPerMeter": round(drag_per_meter, 2), "LD": round(ld_ratio, 2),
            "Re": round(Re), "stallWarning": stall_warning, "effectiveStall": round(effective_stall, 1)}

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
        return (f"At {aoa} deg angle of attack, airflow starts separating from the upper surface. "
                f"Lift reduces and drag increases rapidly - this is stall. The critical stall angle "
                f"is approximately {effective_stall} deg at the current Reynolds number.{re_note}")
    if aoa < 3:
        expl = f"At {aoa} deg angle of attack, the airfoil produces low lift. "
    elif aoa < 8:
        expl = f"At {aoa} deg angle of attack, the airfoil deflects air downward creating a pressure difference. This increases lift. "
    elif aoa < 12:
        expl = f"At {aoa} deg angle of attack, lift continues to increase but drag also grows faster. "
    else:
        expl = f"At {aoa} deg angle of attack, the airfoil is approaching stall. Drag increases rapidly. "
    if abs_camber > 0.005:
        expl += f"This is a cambered airfoil (zero-lift angle: {zero_lift_angle} deg). "
    else:
        expl += "This is a symmetric airfoil - zero lift at 0 deg AoA. "
    if ld_ratio > 30:
        expl += f"L/D ratio is {ld_ratio} - excellent efficiency."
    elif ld_ratio > 15:
        expl += f"L/D ratio is {ld_ratio} - moderate efficiency."
    else:
        expl += f"L/D ratio is {ld_ratio} - drag is relatively high."
    return expl + re_note

# ---------------------------------------------------------------------------
# NACA 4-Digit Generator
# ---------------------------------------------------------------------------

def generate_naca_coords(camber_pct, pos_pct, thickness_pct, num_points=100):
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

# ---------------------------------------------------------------------------
# Predefined Airfoils
# ---------------------------------------------------------------------------

def build_airfoils():
    return {
        "NACA 0012": {
            "type": "Symmetric", "maxThickness": 0.12, "maxCamber": 0.0, "zeroLiftAngle": 0, "stallAngle": 14,
            "description": "Symmetric airfoil, 12% thickness. Zero lift at 0 deg AoA.",
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
            "coords": [(1.0,0.0013),(0.95,0.0074),(0.9,0.0126),(0.8,0.0214),(0.7,0.0286),(0.6,0.0347),(0.5,0.0398),(0.4,0.0433),(0.3,0.0435),(0.25,0.0416),(0.2,0.0375),(0.15,0.0308),(0.1,0.0218),(0.075,0.0163),(0.05,0.0105),(0.025,0.0052),(0.0125,0.0026),(0.0,0.0),(0.0125,-0.0029),(0.025,-0.0058),(0.05,-0.0113),(0.075,-0.0166),(0.1,-0.0216),(0.15,-0.0305),(0.2,-0.0376),(0.25,-0.0421),(0.3,-0.044),(0.4,-0.044),(0.5,-0.0401),(0.6,-0.0349),(0.7,-0.0288),(0.8,-0.0215),(0.9,-0.0127),(0.95,-0.0075),(1.0,-0.0013)]
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
    if "predefined_airfoils" not in st.session_state:
        st.session_state["predefined_airfoils"] = build_airfoils()
    predefined = st.session_state["predefined_airfoils"]
    custom = st.session_state.get("custom_airfoils", {})
    all_foils = dict(predefined)
    all_foils.update(custom)
    return all_foils, predefined, custom

# ---------------------------------------------------------------------------
# Coordinate parsing & geometry extraction
# ---------------------------------------------------------------------------

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
        if yu - yl > max_thickness:
            max_thickness = yu - yl
        if abs((yu + yl) / 2) > abs(max_camber):
            max_camber = (yu + yl) / 2
    abs_camber = abs(max_camber)
    if abs_camber < 0.005:
        stall = min(15, 14 + max_thickness * 100 * -0.15 + 0.5)
    elif abs_camber < 0.03:
        stall = min(16, 15 + max_thickness * 100 * -0.1 + 0.5)
    else:
        stall = 13
    stall = max(8, min(20, round(stall * 10) / 10))
    return {"maxThickness": round(max_thickness, 4), "maxCamber": round(max_camber, 4),
            "stallAngle": stall, "zeroLiftAngle": round(-max_camber * 80, 2)}

# ---------------------------------------------------------------------------
# Chart styling (dark theme)
# ---------------------------------------------------------------------------

def style_chart(fig, ax, title, xlabel, ylabel):
    fig.patch.set_facecolor("#0b1225")
    ax.set_facecolor("#0f1420")
    ax.set_title(title, fontsize=12, fontweight="600", color="#eef2f8", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color="#8899b4")
    ax.set_ylabel(ylabel, fontsize=10, color="#8899b4")
    ax.tick_params(colors="#5a6f8a", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e2d48")
    ax.grid(True, linestyle="--", alpha=0.12, color="#3b82f6")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, labelcolor="#8899b4", framealpha=0.2, loc="best")
    fig.tight_layout()

# ---------------------------------------------------------------------------
# Airfoil visualization (matplotlib, publication quality)
# ---------------------------------------------------------------------------

def _draw_arrow(ax, x1, y1, x2, y2, color, lw=3, head_w=0.045, head_l=0.065):
    """Draw a line with a filled triangular arrowhead from (x1,y1) to (x2,y2)."""
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle="round", zorder=10)
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-10:
        return
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    tip = (x2, y2)
    bl = (x2 - ux * head_l - px * head_w, y2 - uy * head_l - py * head_w)
    br = (x2 - ux * head_l + px * head_w, y2 - uy * head_l + py * head_w)
    ax.fill([tip[0], bl[0], br[0]], [tip[1], bl[1], br[1]], color=color, edgecolor="none", zorder=11)


def draw_airfoil(coords, aoa=0, stall=False, width=7, height=3.5, show_forces=True, CL=None, CD=None):
    if not coords or len(coords) < 3:
        fig, ax = plt.subplots(figsize=(width, height), facecolor="#0b1225")
        ax.set_facecolor("#0f1420")
        ax.text(0.5, 0.5, "Select an airfoil", ha="center", va="center", color="#5a6f8a", fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(width, height), facecolor="#0b1225")
    ax.set_facecolor("#0f1420")

    aoa = max(-30, min(30, aoa))
    rad = math.radians(aoa)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = 0.5, 0
    neg = CL is not None and CL < 0

    def rot(x, y):
        dx = (x - cx) * cos_a - (y - cy) * sin_a
        dy = (x - cx) * sin_a + (y - cy) * cos_a
        return dx + cx, dy + cy

    pts = np.array([rot(x, y) for x, y in coords])
    upper = np.array(sorted([rot(x, y) for x, y in coords if y >= -1e-12], key=lambda p: p[0]))
    lower = np.array(sorted([rot(x, y) for x, y in coords if y < -1e-12], key=lambda p: p[0]))

    # Chord line (rotated, dashed)
    ch = np.array([rot(x, 0) for x in [-0.05, 1.05]])
    ax.plot(ch[:, 0], ch[:, 1], color="#475569", linestyle="--", linewidth=1, alpha=0.4, zorder=1)

    # Airfoil fill (upper blue, lower red)
    if len(upper) > 1:
        ax.fill_between(upper[:, 0], 0, upper[:, 1], color="#60a5fa", alpha=0.08, interpolate=False, zorder=2)
    if len(lower) > 1:
        lc = "#ef4444" if not neg else "#60a5fa"
        ax.fill_between(lower[:, 0], lower[:, 1], 0, color=lc, alpha=0.08, interpolate=False, zorder=2)

    # Airfoil outline (thick, clean)
    ax.plot(pts[:, 0], pts[:, 1], color="#93c5fd", linewidth=2.5, solid_capstyle="round", zorder=3)

    # Wind arrow (horizontal from left) + AoA arc
    wx = -0.04
    ax.annotate("", xy=(wx + 0.12, 0), xytext=(wx, 0),
                arrowprops=dict(arrowstyle="->", color="#60a5fa", lw=2.5), zorder=5)

    if aoa != 0:
        arc_r = 0.22
        arc_th = np.linspace(0, rad, 40)
        ax.plot(wx + arc_r * np.cos(arc_th), arc_r * np.sin(arc_th),
                color="#60a5fa", lw=1.5, alpha=0.5, zorder=2)
        la = rad * 0.5
        ax.text(wx + (arc_r + 0.05) * math.cos(la), (arc_r + 0.05) * math.sin(la),
                f"α = {aoa}°", fontsize=9, color="#60a5fa", fontweight="bold", ha="center", zorder=5)

    # Pressure labels
    y_top = max(pts[:, 1])
    y_bot = min(pts[:, 1])
    pt = y_top + 0.1
    pb = y_bot - 0.1
    ax.text(0.5, pt, "Low Pressure" if not neg else "High Pressure",
            ha="center", fontsize=9, color="#60a5fa", fontweight="bold", zorder=5)
    ax.text(0.5, pb, "High Pressure" if not neg else "Low Pressure",
            ha="center", fontsize=9, color="#f87171", fontweight="bold", zorder=5)

    # Lift & Drag arrows at rotated quarter-chord
    if show_forces and CL is not None:
        rqx, rqy = rot(0.25, 0)
        al = 0.4
        ld = 1 if CL >= 0 else -1

        # Lift arrow (vertical, points up for +CL)
        _draw_arrow(ax, rqx, rqy, rqx, rqy + ld * al, "#22c55e")
        ax.text(rqx + 0.05, rqy + ld * al * 0.55, "Lift", fontsize=12,
                color="#22c55e", fontweight="bold", va="center", zorder=10)
        ax.text(rqx + 0.05, rqy + ld * al * 0.55 - 0.06, f"CL = {CL:.4f}", fontsize=7,
                color="#22c55e", alpha=0.7, va="center", zorder=10)

        # Drag arrow (horizontal, rearward)
        _draw_arrow(ax, rqx, rqy, rqx + 0.75 * al, rqy, "#f97316")
        ax.text(rqx + 0.35 * al, rqy - 0.1, "Drag", fontsize=12,
                color="#f97316", fontweight="bold", ha="center", zorder=10)
        ax.text(rqx + 0.35 * al, rqy - 0.16, f"CD = {CD:.4f}", fontsize=7,
                color="#f97316", alpha=0.7, ha="center", zorder=10)

    # Stall badge
    if stall:
        ax.add_patch(FancyBboxPatch((0.5 - 0.1, 0.09), 0.2, 0.055,
                                     boxstyle="round,pad=0.015",
                                     facecolor="#ef4444", alpha=0.92, zorder=15, transform=ax.transData))
        ax.text(0.5, 0.1175, "STALL", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=16)

    # Axes limits
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ym = max(abs(y_top) + 0.12, abs(y_bot) + 0.12, 0.28)
    if show_forces and CL is not None:
        ym = max(ym, 0.55)
    ax.set_ylim(-ym, ym)
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_anatomy():
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#0b1225")
    ax.set_facecolor("#0f1420")
    m, p, t = 0.04, 0.4, 0.15
    xs = np.linspace(0, 1, 120)
    yt = 5 * t * (0.2969 * np.sqrt(xs) - 0.1260 * xs - 0.3516 * xs**2 + 0.2843 * xs**3 - 0.1015 * xs**4)
    camber_mask = xs < p
    yc = np.where(camber_mask, (m / p**2) * (2 * p * xs - xs**2), (m / (1 - p)**2) * (1 - 2 * p + 2 * p * xs - xs**2))
    dyc = np.where(camber_mask, (2 * m / p**2) * (p - xs), (2 * m / (1 - p)**2) * (p - xs))
    theta = np.arctan(dyc)
    yu = yc + yt * np.cos(theta)
    yl = yc - yt * np.cos(theta)
    ax.fill_between(xs, 0, yu, color="#60a5fa", alpha=0.1)
    ax.plot(xs, yu, "-", color="#60a5fa", linewidth=2.5, label="Upper Surface")
    ax.fill_between(xs, yl, 0, color="#f87171", alpha=0.1)
    ax.plot(xs, yl, "-", color="#f87171", linewidth=2.5, label="Lower Surface")
    ax.plot(xs, yc, "--", color="#4ade80", linewidth=2, label="Camber Line")
    ax.axhline(0, color="#60a5fa", linestyle="--", linewidth=1.2, alpha=0.4, label="Chord Line")
    # LE / TE labels
    ax.annotate("Leading Edge", xy=(0, 0), xytext=(-0.03, -0.15), ha="center", fontsize=9, color="#94a3b8",
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1))
    ax.annotate("Trailing Edge", xy=(1, 0), xytext=(1.03, -0.15), ha="center", fontsize=9, color="#94a3b8",
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1))
    # Max thickness arrow
    max_t_idx = np.argmax(yt * 2)
    mx = xs[max_t_idx]
    ax.annotate("", xy=(mx, yu[max_t_idx]), xytext=(mx, yl[max_t_idx]),
                arrowprops=dict(arrowstyle="<->", color="#fbbf24", lw=2.5))
    ax.text(mx + 0.03, 0, f"Max Thickness  {t*100:.0f}%", fontsize=9, color="#fbbf24", fontweight="bold", va="center")
    ax.set_xlim(-0.08, 1.08)
    margin = 0.26
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0b1225")
    return fig


def draw_naca_preview(camber_pct, pos_pct, thick_pct):
    coords = generate_naca_coords(camber_pct, pos_pct, thick_pct, 80)
    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0b1225")
    ax.set_facecolor("#0f1420")
    pts = np.array(coords)
    ax.fill(pts[:, 0], pts[:, 1], color="#3b82f6", alpha=0.08)
    ax.plot(pts[:, 0], pts[:, 1], "-", color="#60a5fa", linewidth=2.5)
    ax.axhline(0, color="#334155", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    margin = max(0.18, abs(max(pts[:, 1])) * 1.3)
    ax.set_ylim(-margin, margin)
    ax.axis("off")
    fig.patch.set_facecolor("#0b1225")
    return fig

# ---------------------------------------------------------------------------
# Quiz questions (15 total, matches React)
# ---------------------------------------------------------------------------

QUIZ = [
    {"q": "What happens to lift when velocity increases?", "opts": ["Lift decreases", "Lift increases proportionally to V2", "Lift stays the same", "Lift increases proportionally to V"], "ans": 1},
    {"q": "Which force acts opposite to the direction of aircraft motion?", "opts": ["Lift", "Thrust", "Drag", "Weight"], "ans": 2},
    {"q": "What is angle of attack?", "opts": ["Angle between wing chord line and relative wind", "Angle of aircraft relative to ground", "Angle of tail relative to fuselage", "Angle between upper and lower wing surfaces"], "ans": 0},
    {"q": "What happens to lift after the stall angle is exceeded?", "opts": ["Lift continues to increase", "Lift remains constant", "Lift decreases rapidly", "Lift becomes zero instantly"], "ans": 2},
    {"q": "Why do cambered airfoils generate lift at low angle of attack?", "opts": ["They are lighter than symmetric airfoils", "Curved upper surface creates pressure difference even at 0 deg AoA", "They have lower thickness", "They create more drag which increases lift"], "ans": 1},
    {"q": "What does Reynolds number represent?", "opts": ["Ratio of lift to drag", "Ratio of inertial to viscous forces in the flow", "The speed of sound in air", "The density of the airfoil material"], "ans": 1},
    {"q": "What effect does a low Reynolds number have on an airfoil?", "opts": ["Lift increases and drag decreases", "Laminar separation bubbles reduce lift and increase drag", "The stall angle increases significantly", "The airfoil becomes more efficient"], "ans": 1},
    {"q": "What is the zero-lift angle of attack?", "opts": ["The angle at which drag is zero", "The angle at which the airfoil produces no lift", "The angle at which stall occurs", "The angle between the upper and lower camber line"], "ans": 1},
    {"q": "For a symmetric airfoil like NACA 0012, the zero-lift angle is:", "opts": ["0 deg", "About -2 deg", "About 5 deg", "Depends on thickness"], "ans": 0},
    {"q": "What is the chord line of an airfoil?", "opts": ["The line of maximum thickness", "Straight line connecting leading and trailing edges", "The curved upper surface", "The line along the camber at 50% chord"], "ans": 1},
    {"q": "What does the lift-to-drag ratio (L/D) measure?", "opts": ["How fast the aircraft can fly", "Aerodynamic efficiency - lift per unit drag", "The structural strength of the wing", "The stall speed of the aircraft"], "ans": 1},
    {"q": "What is the main difference between NACA 0012 and NACA 2412?", "opts": ["0012 is thicker", "2412 has 2% camber while 0012 is symmetric", "2412 has a higher stall angle", "0012 is cambered and 2412 is symmetric"], "ans": 1},
    {"q": "What is the drag polar?", "opts": ["Graph showing lift coefficient vs drag coefficient", "Distribution of pressure on the airfoil surface", "Angle at which minimum drag occurs", "Polar coordinate representation of the airfoil shape"], "ans": 0},
    {"q": "In the standard atmosphere at sea level, air density is approximately:", "opts": ["0.5 kg/m3", "1.225 kg/m3", "2.5 kg/m3", "10 kg/m3"], "ans": 1},
    {"q": "What causes induced drag?", "opts": ["Friction between air and wing surface", "Generation of lift -- trailing vortices deflect airflow", "The thickness of the airfoil", "Compressibility effects at high speed"], "ans": 1}
]

THEORY_TOPICS = [
    {"title": "Airfoil", "icon": "✈️", "content": "An airfoil is the cross-sectional shape of a wing, blade, or sail designed to generate lift when moving through air.", "color": "blue"},
    {"title": "Lift", "icon": "⬆️", "content": "Lift is the upward force opposing weight. It is generated by the pressure difference between the upper and lower surfaces.", "formula": "L = 0.5 x p x V2 x c x CL", "color": "green"},
    {"title": "Drag", "icon": "⬅️", "content": "Drag is the force opposing forward motion, caused by air resistance and the creation of lift (induced drag).", "formula": "D = 0.5 x p x V2 x c x CD", "color": "orange"},
    {"title": "Angle of Attack", "icon": "📐", "content": "The angle between the chord line and the relative wind. Increasing AoA increases lift up to the stall point.", "color": "purple"},
    {"title": "Stall", "icon": "⚠️", "content": "Occurs when AoA exceeds a critical value - airflow separates from the upper surface, lift drops, drag increases.", "color": "red"},
    {"title": "Lift Coefficient (CL)", "icon": "📊", "content": "A dimensionless number relating lift to fluid density, velocity, and chord length.", "formula": "CL = 0.1 x (AoA - a0)", "color": "blue"},
    {"title": "Drag Coefficient (CD)", "icon": "📉", "content": "A dimensionless number quantifying drag resistance, including skin friction and induced components.", "color": "amber"},
    {"title": "L/D Ratio", "icon": "🎯", "content": "Measure of aerodynamic efficiency. Higher L/D means more lift per unit drag. Gliders have very high L/D ratios.", "color": "teal"},
    {"title": "Camber", "icon": "🔄", "content": "The curvature of an airfoil. Camber helps generate lift even at zero AoA by creating a pressure difference.", "color": "indigo"},
    {"title": "Chord Line", "icon": "📏", "content": "Straight line connecting leading edge to trailing edge. The chord length is a reference dimension.", "color": "cyan"},
    {"title": "Pressure Difference", "icon": "🔽🔼", "content": "Lift is created by the pressure difference: lower pressure above (faster flow), higher pressure below (slower flow).", "color": "violet"},
    {"title": "Flow Separation", "icon": "🌊", "content": "Boundary layer detaching from the surface at high AoA, leading to stall, increased drag, and decreased lift.", "color": "rose"}
]

# ---------------------------------------------------------------------------
# CSS (dark theme, professional)
# ---------------------------------------------------------------------------

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: radial-gradient(ellipse at 50% 0%, #0a1328, #050812); }
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px; }
    h1 { background: linear-gradient(135deg, #3b82f6, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 800 !important; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
    h2 { color: #eef2f8 !important; font-weight: 700 !important; font-size: 1.25rem; letter-spacing: -0.01em; }
    h3 { color: #cbd5e0 !important; font-weight: 600 !important; font-size: 1.05rem; }
    p, li, .stMarkdown { color: #94a3b8; }
    .stSidebar { background: linear-gradient(180deg, #0b1326, #080e1e) !important; border-right: 1px solid rgba(59,130,246,0.08); }
    .stSidebar .stMarkdown, .stSidebar label, .stSidebar span, .stSidebar p { color: #cbd5e0; }
    .stSelectbox > div > div { background: rgba(17,27,48,0.8); border-color: #1e2d48; color: #eef2f8; border-radius: 10px; backdrop-filter: blur(8px); }
    .stSlider > div > div > div { background: #111b30; }
    .stButton > button { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 10px; padding: 0.4rem 1.4rem; font-weight: 600; transition: all 0.25s; box-shadow: 0 2px 12px rgba(59,130,246,0.15); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(59,130,246,0.3); }
    .stButton > button[kind="secondary"] { background: transparent; border: 1px solid rgba(59,130,246,0.4); color: #60a5fa; box-shadow: none; }
    .stButton > button[kind="secondary"]:hover { background: rgba(59,130,246,0.08); border-color: #3b82f6; }
    .stNumberInput input, .stTextInput input, .stTextArea textarea { background: rgba(17,27,48,0.8); border: 1px solid #1e2d48; color: #eef2f8; border-radius: 10px; }
    .stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus { border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,0.15); }
    .stMetric { background: rgba(17,27,48,0.6); border: 1px solid rgba(30,45,72,0.6); border-radius: 12px; padding: 0.8rem 1rem; backdrop-filter: blur(4px); }
    .stMetric label { color: #5a6f8a !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.04em; }
    .stMetric [data-testid="stMetricValue"] { color: #eef2f8 !important; font-size: 1.35rem !important; font-weight: 700; }
    hr { border-color: rgba(30,45,72,0.5) !important; margin: 0.8rem 0; }
    .stAlert { background: rgba(17,27,48,0.8); border: 1px solid #1e2d48; border-radius: 12px; color: #eef2f8; backdrop-filter: blur(4px); }
    .stTabs [data-baseweb="tab-list"] { background: rgba(11,18,37,0.6); border-radius: 10px; gap: 0; padding: 2px; border: 1px solid rgba(30,45,72,0.4); }
    .stTabs [data-baseweb="tab"] { color: #5a6f8a; font-weight: 500; font-size: 0.8rem; padding: 0.4rem 0.9rem; border-radius: 8px; transition: all 0.2s; }
    .stTabs [aria-selected="true"] { color: #eef2f8; background: rgba(59,130,246,0.15); }
    div.stTabs [data-baseweb="tab-highlight"] { display: none; }
    .st-bw { background: rgba(17,27,48,0.6); border: 1px solid #1e2d48; border-radius: 10px; }
    .stDataFrame { background: rgba(17,27,48,0.6); border: 1px solid #1e2d48; border-radius: 10px; }
    .stCheckbox label { color: #94a3b8; }
    .card { background: rgba(17,27,48,0.5); border: 1px solid rgba(30,45,72,0.5); border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; backdrop-filter: blur(8px); transition: border-color 0.2s; }
    .card:hover { border-color: rgba(59,130,246,0.25); }
    .card-center { text-align: center; }
    .glow-btn { text-decoration: none; display: inline-block; }
    .st-emotion-cache-1kyxreq { display: none; }
    div[data-testid="stExpander"] { background: rgba(17,27,48,0.6); border: 1px solid #1e2d48; border-radius: 10px; }
    details { background: rgba(17,27,48,0.6); border: 1px solid #1e2d48; border-radius: 10px; }
    .stRadio label { color: #cbd5e0 !important; }
    .stRadio > div { gap: 0.15rem; }
    .hero { text-align: center; padding: 2rem 0 1rem 0; }
    .hero-icon { font-size: 4.5rem; display: block; margin-bottom: 0.3rem; filter: drop-shadow(0 4px 20px rgba(59,130,246,0.2)); }
    .hero h1 { font-size: 2.8rem; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.1rem; color: #64748b; max-width: 650px; margin: 0 auto 1.5rem; }
    .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.8rem; margin: 1.2rem 0; }
    .feature-card { background: rgba(17,27,48,0.4); border: 1px solid rgba(30,45,72,0.4); border-radius: 14px; padding: 1.2rem 0.8rem; text-align: center; transition: all 0.25s; backdrop-filter: blur(4px); }
    .feature-card:hover { border-color: rgba(59,130,246,0.4); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.2); }
    .feature-card .icon { font-size: 2.2rem; margin-bottom: 0.4rem; }
    .feature-card h3 { margin: 0 0 0.3rem 0; font-size: 0.95rem; color: #eef2f8 !important; }
    .feature-card p { margin: 0; font-size: 0.8rem; color: #64748b; line-height: 1.4; }
    div[data-testid="stSlider"], div[data-testid="stSlider"] > div { padding-top: 0.2rem; }
    div[data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 0.8rem; }
    .st-cb, .st-cd { background: #1e2d48 !important; }
    .st-cc { background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; }
    .stCheckbox label > div[data-testid="stMarkdownContainer"] { color: #94a3b8; }
    span[data-baseweb="tag"] { background: rgba(59,130,246,0.15) !important; color: #60a5fa !important; border: 1px solid rgba(59,130,246,0.25) !important; border-radius: 6px !important; }
    div.st-ae { background: transparent !important; }
    .st-dg { background: rgba(255,255,255,0.03); }
</style>
"""

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home():
    st.markdown("<div class='hero'><span class='hero-icon'>✈️</span><h1>Interactive Airfoil Lift & Drag Visualizer</h1><p>Explore how airfoil shape, velocity, and angle of attack affect lift and drag through interactive visualization.</p></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Simulation", use_container_width=True):
            st.session_state["nav"] = "Simulator"
            st.session_state.pop("nav_radio", None)
            st.rerun()
    with col2:
        if st.button("Custom Airfoil", use_container_width=True):
            st.session_state["nav"] = "Custom Airfoil"
            st.session_state.pop("nav_radio", None)
            st.rerun()
    with col3:
        if st.button("Compare Airfoils", use_container_width=True):
            st.session_state["nav"] = "Compare"
            st.session_state.pop("nav_radio", None)
            st.rerun()

    st.markdown("<h2 style='text-align:center;margin-top:2rem;'>Understanding Aerodynamics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#5a6f8a;margin-bottom:1.5rem;'>Explore the key concepts behind lift and drag generation</p>", unsafe_allow_html=True)

    features = [
        ("⬆️", "Lift", "Upward force generated by pressure difference between upper and lower surfaces."),
        ("⬅️", "Drag", "Rearward force opposing motion, caused by air resistance and flow separation."),
        ("📐", "Angle of Attack", "Angle between wing chord line and relative wind - critical for lift generation."),
        ("🚀", "Velocity", "Speed relative to air. Lift and drag increase with the square of velocity (V2)."),
        ("✈️", "Airfoil Shape", "Cross-sectional shape determining aerodynamic characteristics. Camber and thickness are key.")
    ]
    st.markdown("<div class='feature-grid'>", unsafe_allow_html=True)
    for icon, title, desc in features:
        st.markdown(f"<div class='feature-card'><div class='icon'>{icon}</div><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h2>About This Project</h2><p>This interactive educational tool helps students understand how lift and drag are produced on different airfoils by changing input parameters and seeing real-time visual results. It combines aeronautical engineering principles with modern web technology to create an engaging learning experience.</p><p style='margin-top:0.75rem;color:#5a6f8a;font-size:0.85rem;'>Note: Uses simplified aerodynamic approximations for educational visualization. Results are not CFD-accurate and should not be used for real aircraft design without validation.</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;margin-top:1.5rem;'><span style='color:#5a6f8a;font-size:0.75rem;'>Built with Streamlit, NumPy & Matplotlib</span></div>", unsafe_allow_html=True)


def page_simulator():
    st.markdown("<h1>Aerodynamic Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom:1.5rem;'>Adjust parameters to see how lift and drag change in real time</p>", unsafe_allow_html=True)

    all_foils, _, custom = get_airfoils()
    foil_names = sorted(all_foils.keys())
    custom_names = sorted(custom.keys())

    use_custom = st.checkbox("Use custom airfoil", key="sim_custom")
    source = custom if use_custom else all_foils
    source_names = sorted(source.keys())
    if not source_names:
        st.warning("No airfoils available.")
        return

    sel_name = st.selectbox("Select Airfoil", source_names, key="sim_foil")
    foil = all_foils[sel_name]
    coords = foil.get("coords")

    st.markdown(f"<div class='card'><h3>Airfoil Properties</h3><p><b>Type:</b> {foil['type']} &nbsp;|&nbsp; <b>Thickness:</b> {foil['maxThickness']*100:.1f}% &nbsp;|&nbsp; <b>Camber:</b> {foil['maxCamber']*100:.2f}% &nbsp;|&nbsp; <b>Stall:</b> {foil['stallAngle']} deg &nbsp;|&nbsp; <b>a0:</b> {foil['zeroLiftAngle']} deg</p></div>", unsafe_allow_html=True)

    st.markdown("<h2>Flight Parameters</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        aoa = st.slider("Angle of Attack (deg)", -5, 20, 5, 1)
    with c2:
        velocity = st.slider("Velocity (m/s)", 5, 100, 30, 1)
    with c3:
        chord = st.slider("Chord Length (m)", 0.1, 2.0, 1.0, 0.05)
    with c4:
        air_density = st.number_input("Air Density (kg/m3)", 0.1, 2.0, 1.225, 0.001, format="%.3f")

    result = calculate_aerodynamics(aoa, foil["stallAngle"], foil["zeroLiftAngle"],
                                     foil["maxCamber"], foil["maxThickness"],
                                     velocity, chord, air_density)

    st.markdown("<h2>Airfoil Visualization</h2>", unsafe_allow_html=True)
    fig = draw_airfoil(coords, aoa, result["stallWarning"], show_forces=True,
                        CL=result["CL"], CD=result["CD"])
    st.pyplot(fig)
    plt.close(fig)

    if result["stallWarning"]:
        st.error(f"STALL WARNING: Angle of attack exceeds the stall angle ({result['effectiveStall']} deg). Lift decreases, drag increases rapidly.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Lift Coefficient (CL)", result["CL"])
    c2.metric("Drag Coefficient (CD)", result["CD"])
    c3.metric("L/D Ratio", result["LD"])

    wing_span = st.slider("Wing Span (m)", 1.0, 50.0, 10.0, 0.5, key="sim_span")
    st.markdown(f"<div class='card' style='text-align:center;'><h2 style='margin:0;'>"
                f"<span style='color:#22c55e;'>Lift: {result['liftPerMeter'] * wing_span:.2f} N</span>"
                f" &nbsp;|&nbsp; "
                f"<span style='color:#f97316;'>Drag: {result['dragPerMeter'] * wing_span:.2f} N</span>"
                f" &nbsp;|&nbsp; Re: {result['Re']:,}</h2></div>", unsafe_allow_html=True)

    expl = generate_explanation(aoa, result["CL"], result["CD"], result["LD"],
                                 result["stallWarning"], foil["maxCamber"],
                                 foil["zeroLiftAngle"], result["effectiveStall"], result["Re"])
    st.info(expl)

    # Charts (no operating point markers)
    st.markdown("<h2>Aerodynamic Characteristic Curves</h2>", unsafe_allow_html=True)
    alpha_range = np.linspace(-5, 20, 101)
    CLs = [calculate_aerodynamics(a, foil["stallAngle"], foil["zeroLiftAngle"],
                                   foil["maxCamber"], foil["maxThickness"],
                                   velocity, chord, air_density)["CL"] for a in alpha_range]
    CDs = [calculate_aerodynamics(a, foil["stallAngle"], foil["zeroLiftAngle"],
                                   foil["maxCamber"], foil["maxThickness"],
                                   velocity, chord, air_density)["CD"] for a in alpha_range]
    LDs = [CLs[i] / CDs[i] if CDs[i] > 0 else 0 for i in range(len(CLs))]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CL vs AoA", "CD vs AoA", "Drag Polar", "L/D vs AoA", "Velocity"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(alpha_range, CLs, "-", color="#3b82f6", linewidth=2, label=sel_name)
        ax.axvline(x=foil["stallAngle"], color="#ef4444", linestyle="--", alpha=0.7, label=f"Stall ({foil['stallAngle']} deg)")
        ax.axvline(x=foil["zeroLiftAngle"], color="#64748b", linestyle=":", alpha=0.5, label=f"a0={foil['zeroLiftAngle']} deg")
        style_chart(fig, ax, "CL vs Angle of Attack", "AoA (deg)", "CL")
        st.pyplot(fig); plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(alpha_range, CDs, "-", color="#ef4444", linewidth=2, label=sel_name)
        ax.axvline(x=12, color="#a855f7", linestyle="--", alpha=0.7, label="High drag onset (12 deg)")
        style_chart(fig, ax, "CD vs Angle of Attack", "AoA (deg)", "CD")
        st.pyplot(fig); plt.close(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(CDs, CLs, "-", color="#22c55e", linewidth=2, label=sel_name)
        style_chart(fig, ax, "Drag Polar (CL vs CD)", "CD", "CL")
        st.pyplot(fig); plt.close(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(alpha_range, LDs, "-", color="#a855f7", linewidth=2, label=sel_name)
        style_chart(fig, ax, "L/D Ratio vs Angle of Attack", "AoA (deg)", "L/D")
        st.pyplot(fig); plt.close(fig)

    with tab5:
        vel_range = np.arange(5, 101, 5)
        lifts_v = [calculate_aerodynamics(aoa, foil["stallAngle"], foil["zeroLiftAngle"], foil["maxCamber"],
                                           foil["maxThickness"], v, chord, air_density)["liftPerMeter"] for v in vel_range]
        drags_v = [calculate_aerodynamics(aoa, foil["stallAngle"], foil["zeroLiftAngle"], foil["maxCamber"],
                                           foil["maxThickness"], v, chord, air_density)["dragPerMeter"] for v in vel_range]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.plot(vel_range, lifts_v, "-", color="#22c55e", linewidth=2)
        style_chart(fig, ax1, "Velocity vs Lift", "Velocity (m/s)", "Lift (N/m)")
        ax2.plot(vel_range, drags_v, "-", color="#f97316", linewidth=2)
        style_chart(fig, ax2, "Velocity vs Drag", "Velocity (m/s)", "Drag (N/m)")
        st.pyplot(fig); plt.close(fig)


def page_custom():
    st.markdown("<h1>Custom Airfoil Upload</h1>", unsafe_allow_html=True)
    st.markdown("<p>Upload your own airfoil coordinates or paste them to analyze</p>", unsafe_allow_html=True)

    if "custom_airfoils" not in st.session_state:
        st.session_state["custom_airfoils"] = {}

    tab = st.radio("Input Method", ["Paste Coordinates", "Upload File", "My Library"], horizontal=True, label_visibility="collapsed")

    result = None
    if tab == "Paste Coordinates":
        name = st.text_input("Airfoil Name (optional)", placeholder="My Custom Airfoil")
        coord_text = st.text_area("Coordinate Data", height=200,
            placeholder="NACA 2412\n1.0000 0.0013\n0.9500 0.0074\n...")
        if st.button("Process Airfoil", type="primary"):
            if coord_text.strip():
                coords, err = parse_coordinates(coord_text)
                if err:
                    st.error(err)
                else:
                    geo = extract_geometry(coords)
                    result = {"name": name.strip() or "Custom Airfoil", "coords": coords, "geometry": geo}
                    st.success(f"Parsed {len(coords)} coordinate points!")

    elif tab == "Upload File":
        uploaded = st.file_uploader("Choose .dat, .txt, or .csv", type=["dat", "txt", "csv"])
        name = st.text_input("Name (optional)", placeholder="Airfoil name")
        if uploaded is not None:
            content = uploaded.read().decode("utf-8").strip()
            coords, err = parse_coordinates(content)
            if err:
                st.error(err)
            else:
                geo = extract_geometry(coords)
                result = {"name": name.strip() or uploaded.name.rsplit(".", 1)[0], "coords": coords, "geometry": geo}
                st.success(f"Loaded {len(coords)} points from {uploaded.name}")

    else:
        saved = st.session_state.get("custom_airfoils", {})
        if not saved:
            st.info("No saved airfoils yet.")
        else:
            for fname, fdata in list(saved.items()):
                with st.container():
                    a, b, c = st.columns([3, 1, 1])
                    a.write(f"**{fname}**")
                    if b.button("Load", key=f"ld_{fname}"):
                        result = fdata
                    if c.button("Delete", key=f"del_{fname}"):
                        del st.session_state["custom_airfoils"][fname]
                        st.rerun()

    if result:
        geo = result["geometry"]
        st.markdown("<h2>Geometry Analysis</h2>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Thickness", f"{geo['maxThickness']*100:.1f}%")
        col2.metric("Max Camber", f"{geo['maxCamber']*100:.2f}%")
        col3.metric("Stall Angle (est.)", f"{geo['stallAngle']} deg")
        col4.metric("Zero-Lift Angle", f"{geo['zeroLiftAngle']} deg")

        fig = draw_airfoil(result["coords"], 0, show_forces=False)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

        if st.button("Save to Library"):
            st.session_state["custom_airfoils"][result["name"]] = {
                "type": "Custom", "maxThickness": geo["maxThickness"], "maxCamber": geo["maxCamber"],
                "zeroLiftAngle": geo["zeroLiftAngle"], "stallAngle": geo["stallAngle"],
                "description": "Custom uploaded airfoil", "coords": result["coords"]
            }
            st.success(f"Saved!")
            st.balloons()


def page_compare():
    st.markdown("<h1>Compare Airfoils</h1>", unsafe_allow_html=True)
    st.markdown("<p>Compare aerodynamic performance between two airfoils side by side</p>", unsafe_allow_html=True)

    all_foils, _, custom = get_airfoils()
    foil_names = sorted(all_foils.keys())
    custom_names = sorted(custom.keys())

    use_a = st.checkbox("Use custom for Airfoil A", key="cmp_a")
    use_b = st.checkbox("Use custom for Airfoil B", key="cmp_b")
    opts_a = custom_names if use_a else foil_names
    opts_b = custom_names if use_b else foil_names

    if not opts_a or not opts_b:
        st.warning("Not enough airfoils.")
        return

    c1, c2 = st.columns(2)
    with c1:
        name_a = st.selectbox("Airfoil A", opts_a, key="cmp_sel_a")
    with c2:
        name_b = st.selectbox("Airfoil B", opts_b, key="cmp_sel_b")

    c1, c2, c3 = st.columns(3)
    aoa = c1.number_input("AoA (deg)", -5.0, 20.0, 5.0, 0.5, key="cmp_aoa")
    vel = c2.number_input("Velocity (m/s)", 5.0, 100.0, 30.0, 1.0, key="cmp_vel")
    chord = c3.number_input("Chord (m)", 0.1, 2.0, 1.0, 0.05, key="cmp_chord")

    fa = all_foils[name_a]
    fb = all_foils[name_b]
    ra = calculate_aerodynamics(aoa, fa["stallAngle"], fa["zeroLiftAngle"], fa["maxCamber"], fa["maxThickness"], vel, chord, 1.225)
    rb = calculate_aerodynamics(aoa, fb["stallAngle"], fb["zeroLiftAngle"], fb["maxCamber"], fb["maxThickness"], vel, chord, 1.225)

    c1, c2 = st.columns(2)
    with c1:
        fig = draw_airfoil(fa.get("coords"), aoa, ra["stallWarning"], show_forces=False)
        st.pyplot(fig); plt.close(fig)
    with c2:
        fig = draw_airfoil(fb.get("coords"), aoa, rb["stallWarning"], show_forces=False)
        st.pyplot(fig); plt.close(fig)

    st.markdown("<h2>Comparison</h2>", unsafe_allow_html=True)
    metrics = [("Lift Coefficient (CL)", ra["CL"], rb["CL"]), ("Drag Coefficient (CD)", ra["CD"], rb["CD"]),
               ("Lift/m (N/m)", ra["liftPerMeter"], rb["liftPerMeter"]), ("Drag/m (N/m)", ra["dragPerMeter"], rb["dragPerMeter"]),
               ("L/D Ratio", ra["LD"], rb["LD"])]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, [m[1] for m in metrics], w, label=name_a, color="#3b82f6", alpha=0.85)
    ax.bar(x + w/2, [m[2] for m in metrics], w, label=name_b, color="#ef4444", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
    ax.legend()
    style_chart(fig, ax, "Performance Comparison", "", "Value")
    st.pyplot(fig); plt.close(fig)

    data = [{"Metric": m[0], name_a: f"{m[1]:.4f}", name_b: f"{m[2]:.4f}",
             "Diff": f"{((m[2]-m[1])/abs(m[1])*100 if m[1]!=0 else 0):+.1f}%"} for m in metrics]
    st.table(data)


def page_learn():
    st.markdown("<h1>Learn Aerodynamics</h1>", unsafe_allow_html=True)
    st.markdown("<p>Student-friendly explanations of key aerodynamic concepts</p>", unsafe_allow_html=True)

    tab = st.radio("", ["Theory", "Nomenclature & Anatomy"], horizontal=True, label_visibility="collapsed")

    if tab == "Theory":
        for i, topic in enumerate(THEORY_TOPICS):
            st.markdown(f"<div class='card'><h3>{topic['icon']} {topic['title']}</h3><p>{topic['content']}</p></div>", unsafe_allow_html=True)
            if topic.get("formula"):
                st.code(topic["formula"])
    else:
        st.markdown("<h2>Airfoil Anatomy</h2>", unsafe_allow_html=True)
        fig = draw_anatomy()
        st.pyplot(fig); plt.close(fig)

        st.markdown("<div style='display:flex;gap:1.5rem;flex-wrap:wrap;justify-content:center;margin:1rem 0;font-size:0.85rem;'>"
                    "<span style='color:#60a5fa;'>Upper Surface</span>"
                    "<span style='color:#f87171;'>Lower Surface</span>"
                    "<span style='color:#4ade80;'>Camber Line</span>"
                    "<span style='color:#60a5fa;'>Chord Line</span>"
                    "<span style='color:#fbbf24;'>Max Thickness</span></div>", unsafe_allow_html=True)

        st.markdown("<h2>NACA 4-Digit Generator</h2>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            camber_pct = st.slider("Camber (%)", 0, 9, 2, 1)
            pos_pct = st.slider("Position of max camber (%)", 10, 60, 40, 5)
            thick_pct = st.slider("Max thickness (%)", 6, 30, 12, 1)
            m_d = camber_pct
            p_d = pos_pct // 10
            naca_name = f"NACA {m_d}{p_d}{str(thick_pct).zfill(2)}"
            st.markdown(f"<h2 style='text-align:center;font-family:monospace;color:#3b82f6;'>{naca_name}</h2>", unsafe_allow_html=True)
            desc = "Symmetric" if camber_pct == 0 else f"{camber_pct}% camber at {pos_pct}% chord"
            st.markdown(f"<p style='text-align:center;color:#5a6f8a;'>{desc} - {thick_pct}% thickness</p>", unsafe_allow_html=True)
            if st.button("Use in Simulator", type="primary"):
                coords = generate_naca_coords(camber_pct, pos_pct, thick_pct, 60)
                geo = extract_geometry(coords)
                st.session_state["custom_airfoils"][naca_name] = {
                    "type": "NACA Generated", "maxThickness": geo["maxThickness"],
                    "maxCamber": geo["maxCamber"], "zeroLiftAngle": geo["zeroLiftAngle"],
                    "stallAngle": geo["stallAngle"], "description": f"NACA 4-digit: {naca_name}",
                    "coords": coords
                }
                st.success(f"Saved! Go to Simulator and enable custom airfoil.")
                st.balloons()
        with c2:
            fig = draw_naca_preview(camber_pct, pos_pct, thick_pct)
            st.pyplot(fig); plt.close(fig)

        with st.expander("How NACA 4-digit naming works"):
            st.markdown(f"**NACA MPXX** - M=Camber%, P=Position/10, XX=Thickness%\nCurrent: {naca_name}")


def page_quiz():
    st.markdown("<h1>Knowledge Quiz</h1>", unsafe_allow_html=True)
    st.markdown("<p>Test your understanding of aerodynamics concepts</p>", unsafe_allow_html=True)

    if "qa" not in st.session_state:
        st.session_state["qa"] = {}
    if "qs" not in st.session_state:
        st.session_state["qs"] = False
    if "qscore" not in st.session_state:
        st.session_state["qscore"] = 0

    if st.session_state["qs"]:
        s = st.session_state["qscore"]
        t = len(QUIZ)
        emoji = "🎉" if s >= 13 else "👍" if s >= 10 else "📚" if s >= 7 else "📖"
        msg = "Excellent!" if s >= 13 else "Great job!" if s >= 10 else "Good effort!" if s >= 7 else "Keep studying!"
        st.markdown(f"<div class='card card-center'><div style='font-size:3rem'>{emoji}</div><h2>{s}/{t}</h2><p>{msg}</p></div>", unsafe_allow_html=True)
        if st.button("Retry"):
            st.session_state["qa"] = {}
            st.session_state["qs"] = False
            st.rerun()

    for i, q in enumerate(QUIZ):
        st.markdown(f"<div class='card'><h3>Q{i+1}. {q['q']}</h3></div>", unsafe_allow_html=True)
        for j, opt in enumerate(q["opts"]):
            label = f"{chr(65+j)}. {opt}"
            if st.session_state["qs"]:
                is_correct = j == q["ans"]
                is_selected = st.session_state["qa"].get(i) == j
                if is_correct:
                    st.success(label)
                elif is_selected:
                    st.error(label)
                else:
                    st.markdown(f"<p style='color:#5a6f8a;'>{label}</p>", unsafe_allow_html=True)
            else:
                if st.button(label, key=f"q_{i}_{j}", use_container_width=True):
                    st.session_state["qa"][i] = j
                    st.rerun()

    if not st.session_state["qs"]:
        answered = len(st.session_state["qa"])
        disabled = answered < len(QUIZ)
        if st.button(f"Submit ({answered}/{len(QUIZ)})" if disabled else "Submit Quiz", disabled=disabled, type="primary"):
            st.session_state["qscore"] = sum(1 for i in range(len(QUIZ)) if st.session_state["qa"].get(i) == QUIZ[i]["ans"])
            st.session_state["qs"] = True
            st.rerun()


def page_about():
    st.markdown("<h1 style='text-align:center;'>Interactive Airfoil Lift & Drag Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>An interdisciplinary engineering education project</p>", unsafe_allow_html=True)

    sections = [
        {"title": "Problem Statement", "content": "Students often find it difficult to understand lift and drag using only formulas and static diagrams. This project provides an interactive visual platform to understand how airfoil geometry, velocity, and angle of attack affect aerodynamic forces."},
        {"title": "Objectives", "items": [
            "To visualize lift and drag generation on airfoils",
            "To allow users to change aerodynamic parameters interactively",
            "To support predefined and custom airfoil coordinate files",
            "To calculate lift, drag, Cl, Cd, and L/D ratio",
            "To help students understand stall and pressure difference",
            "To compare different airfoil shapes"
        ]},
        {"title": "Methodology", "content": "This application uses simplified aerodynamic models based on thin airfoil theory and empirical approximations. Predefined airfoil data is stored in JSON format. Custom airfoil coordinates are parsed, normalized, and analyzed to estimate geometric properties. Aerodynamic coefficients are calculated using educational formulas that capture the essential physics without requiring CFD-level computation."},
        {"title": "Interdisciplinary Connection", "content": "This project bridges Aeronautical Engineering and Computer Science. Aeronautical engineering provides the physics of lift, drag, and airfoil design. Computer science provides interactive visualization, web development, API design, and real-time data processing. The combination creates an engaging educational tool that makes complex concepts accessible."},
        {"title": "Applications", "items": [
            "Aerospace engineering education",
            "Student projects and demonstrations",
            "Understanding airfoil selection for aircraft design",
            "Preliminary aerodynamic analysis",
            "STEM outreach and teaching"
        ]},
        {"title": "Advantages", "items": [
            "Interactive real-time visualization",
            "Supports both predefined and custom airfoils",
            "No installation required - runs in a web browser",
            "Clean, intuitive user interface",
            "Educational explanations for every condition",
            "Modular code structure for easy extension"
        ]},
        {"title": "Limitations", "items": [
            "Uses simplified aerodynamic approximations, not CFD",
            "Not suitable for real aircraft design without validation",
            "Does not account for compressibility effects at high Mach numbers",
            "Limited to 2D airfoil analysis (no 3D wing effects)",
            "No boundary layer or transition modeling",
            "No wind tunnel data correlation"
        ]},
        {"title": "Future Scope", "items": [
            "Add real CFD visualization using panel methods",
            "Integrate with XFOIL for more accurate analysis",
            "Add database for saving simulations (MongoDB/PostgreSQL)",
            "Add user login and authentication",
            "Export results as PDF reports",
            "Add wind tunnel data comparison",
            "Add 3D wing visualization and wingtip effects",
            "Include Reynolds number and Mach number effects",
            "Add multi-element airfoil support (flaps, slats)"
        ]}
    ]
    for s in sections:
        html = f"<div class='card'><h3>{s['title']}</h3>"
        if "content" in s:
            html += f"<p>{s['content']}</p>"
        if "items" in s:
            html += "<ul style='margin:0.5rem 0 0 0;padding-left:1.2rem;'>"
            for item in s["items"]:
                html += f"<li style='margin-bottom:0.3rem;'>{item}</li>"
            html += "</ul>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div class='card card-center'><p style='color:#5a6f8a;font-size:0.85rem;'>Built with Streamlit, NumPy & Matplotlib. Uses simplified aerodynamic approximations for educational visualization.</p></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    nav_items = ["Home", "Simulator", "Custom Airfoil", "Compare", "Learn", "Quiz", "About"]
    nav_icons = ["🏠", "⚡", "📐", "📊", "📖", "❓", "ℹ️"]
    nav_display = [f"{icon} {name}" for icon, name in zip(nav_icons, nav_items)]
    nav_lookup = dict(zip(nav_display, nav_items))

    if "nav" not in st.session_state:
        st.session_state["nav"] = "Home"

    with st.sidebar:
        st.markdown("<div style='text-align:center;padding:0.5rem 0;'><span style='font-size:2.5rem'>✈️</span><p style='color:#eef2f8;font-weight:700;margin:0;'>Airfoil Analyzer</p></div>", unsafe_allow_html=True)
        st.divider()

        # Build index based on current state
        current = st.session_state["nav"]
        default_idx = 0
        if current in nav_items:
            default_idx = nav_items.index(current)

        selected = st.radio("Navigate", nav_display, index=default_idx, key="nav_radio", label_visibility="collapsed")
        # Map from display name back to short name
        st.session_state["nav"] = nav_lookup.get(selected, "Home")

        st.divider()
        all_foils, pre, cust = get_airfoils()
        st.markdown(f"<div style='background:rgba(255,255,255,0.03);padding:0.75rem;border-radius:8px;'>"
                    f"<p style='color:#5a6f8a;font-size:0.7rem;margin:0 0 0.25rem;'>AIRFOILS</p>"
                    f"<p style='color:#eef2f8;font-size:0.85rem;margin:0;'>Predefined: {len(pre)} | Custom: {len(cust)}</p></div>",
                    unsafe_allow_html=True)

    page = st.session_state["nav"]
    if page == "Home":
        page_home()
    elif page == "Simulator":
        page_simulator()
    elif page == "Custom Airfoil":
        page_custom()
    elif page == "Compare":
        page_compare()
    elif page == "Learn":
        page_learn()
    elif page == "Quiz":
        page_quiz()
    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()
