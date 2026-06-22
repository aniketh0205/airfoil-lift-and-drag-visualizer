"""
Interactive Airfoil Lift and Drag Visualizer
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Custom CSS for improved UI (Dark themed)
# ---------------------------------------------------------------------------

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
        }
        
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        .stApp {
            background: #0f1117;
        }
        
        h1 {
            color: #f0f0f0 !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        
        h2, h3 {
            color: #e0e0e0 !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stSidebar"] {
            background: #161822;
            border-right: 1px solid #252836;
        }
        
        [data-testid="stSidebar"] .stMarkdown, 
        [data-testid="stSidebar"] .stRadio label {
            color: #b0b3c0 !important;
        }
        
        .stSelectbox > div > div {
            background: #1e2030 !important;
            border: 1px solid #2d3148 !important;
            color: #e0e0e0 !important;
            border-radius: 8px;
        }
        
        .stSelectbox [data-testid="stMarkdownContainer"] {
            color: #e0e0e0 !important;
        }
        
        .stSlider > div > div > div {
            color: #6c7bff !important;
        }
        
        .stSlider [data-baseweb="slider"] div {
            background: #6c7bff !important;
        }
        
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background: #1e2030 !important;
            border: 1px solid #2d3148 !important;
            color: #e0e0e0 !important;
            border-radius: 8px;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: #6c7bff !important;
            box-shadow: 0 0 0 2px rgba(108, 123, 255, 0.2) !important;
        }
        
        .stButton > button {
            background: #6c7bff !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.2s !important;
        }
        
        .stButton > button:hover {
            background: #5a6ae0 !important;
            box-shadow: 0 4px 15px rgba(108, 123, 255, 0.3) !important;
        }
        
        .stAlert {
            border-radius: 8px !important;
            border: none !important;
        }
        
        hr {
            border: none;
            height: 1px;
            background: #252836;
            margin: 1.5rem 0;
        }
        
        .info-card {
            background: #1a1c2a;
            border: 1px solid #252836;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            border-left: 3px solid #6c7bff;
        }
        
        .info-card p {
            color: #b0b3c0;
            margin: 0;
            font-size: 0.9rem;
        }
        
        .metric-card {
            padding: 1.25rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #252836;
        }
        
        .metric-card .label {
            color: #888;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0 0 0.5rem 0;
        }
        
        .metric-card .value {
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
        }
        
        .stall-warning-modern {
            background: #2a1515;
            border: 1px solid #e53e3e;
            border-radius: 10px;
            padding: 0.75rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .stall-warning-modern .icon {
            color: #fc8181;
            font-size: 1.5rem;
        }
        
        .stall-warning-modern .text {
            color: #fc8181;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .stall-warning-modern .sub {
            color: #a0a0a0;
            font-size: 0.8rem;
        }
        
        .theory-card {
            background: #1a1c2a;
            border: 1px solid #252836;
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s;
        }
        
        .theory-card:hover {
            border-color: #6c7bff;
        }
        
        .theory-card h4 {
            color: #6c7bff;
            margin: 0 0 0.4rem 0;
            font-size: 1rem;
        }
        
        .theory-card p {
            color: #9ca3af;
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        .param-label {
            color: #888;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #888 !important;
            font-size: 0.75rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        
        .st-bd, .st-be, .st-bf, .st-bg {
            border-color: #2d3148 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_airfoil_coordinates(text: str):
    """
    Parse airfoil coordinate data from text.
    Supports common formats with x y coordinates.
    Returns (x_coords, y_coords, error_string | None)
    """
    lines = text.strip().split('\n')
    x_coords = []
    y_coords = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to parse as coordinate pair
        for delimiter in [None, ',', '\t']:
            try:
                if delimiter is None:
                    parts = line.split()
                else:
                    parts = line.split(delimiter)
                parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    x_coords.append(x)
                    y_coords.append(y)
                    break
            except ValueError:
                continue
    
    if len(x_coords) < 4:
        return None, None, "Need at least 4 coordinate points to define an airfoil."
    
    return np.array(x_coords), np.array(y_coords), None


def split_upper_lower(x_norm, y_norm):
    """Split coordinates into upper (y>=0) and lower (y<0) surfaces, like React."""
    upper = y_norm >= -1e-12
    x_upper = x_norm[upper]
    y_upper = y_norm[upper]
    x_lower = x_norm[~upper]
    y_lower = y_norm[~upper]

    sort_upper = np.argsort(x_upper)
    sort_lower = np.argsort(x_lower)

    return x_upper[sort_upper], y_upper[sort_upper], x_lower[sort_lower], y_lower[sort_lower]


def extract_airfoil_geometry(x_coords, y_coords):
    """
    Extract max camber and max thickness from airfoil coordinates.
    Matches React's calculateGeometry() logic.
    """
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    chord = x_max - x_min
    if chord <= 0:
        chord = 1.0

    x_norm = (x_coords - x_min) / chord
    y_norm = y_coords / chord

    # Split by y-sign like React's splitUpperLower
    x_upper, y_upper, x_lower, y_lower = split_upper_lower(x_norm, y_norm)

    # Interpolate to 50 common stations (same as React)
    x_stations = np.linspace(0, 1, 50)

    y_upper_interp = np.interp(x_stations, x_upper, y_upper)
    y_lower_interp = np.interp(x_stations, x_lower, y_lower)

    camber = (y_upper_interp + y_lower_interp) / 2
    thickness = y_upper_interp - y_lower_interp

    max_camber = float(np.max(np.abs(camber)))
    max_thickness = float(np.max(thickness))

    # React's calculateGeometry stall angle formula
    abs_camber = abs(max_camber)
    if abs_camber < 0.005:
        stall_angle = min(15, 14 + max_thickness * 100 * -0.15 + 0.5)
    elif abs_camber < 0.03:
        stall_angle = min(16, 15 + max_thickness * 100 * -0.10 + 0.5)
    else:
        stall_angle = 13
    stall_angle = max(8, min(20, round(stall_angle * 10) / 10))

    return {
        "maxCamber": max_camber,
        "maxThickness": max_thickness,
        "stallAngle": stall_angle
    }


AIR_VISCOSITY = 1.81e-5


def compute_coefficients(angle_of_attack, max_camber, max_thickness, stall_angle, velocity, air_density, chord_length):
    """
    Compute CL and CD using React's calculateAerodynamics() logic.
    Returns (CL, CD, is_stalled)
    """
    zero_lift_angle = -max_camber * 80
    cl_alpha = 0.1

    # Reynolds number (React: reynoldsNumber function)
    Re = (air_density * velocity * chord_length) / AIR_VISCOSITY

    # Stall reduction at low Re (React: stallReduction = 4.0 * exp(-Re/180000))
    stall_reduction = 4.0 * np.exp(-Re / 180000)
    effective_stall = stall_angle - stall_reduction
    effective_stall = max(effective_stall, 6)

    # Re lift factor (React: reLiftFactor = 1 - 0.30*exp(-Re/70000) - 0.12*exp(-Re/250000))
    re_lift_factor = 1 - 0.30 * np.exp(-Re / 70000) - 0.12 * np.exp(-Re / 250000)

    # Re drag factor (React: reDragFactor = 1 + 0.45*exp(-Re/55000) + 0.18*exp(-Re/200000))
    re_drag_factor = 1 + 0.45 * np.exp(-Re / 55000) + 0.18 * np.exp(-Re / 200000)

    CL = cl_alpha * (angle_of_attack - zero_lift_angle) * re_lift_factor

    is_stalled = angle_of_attack > effective_stall

    if is_stalled:
        # Exponential CL decay after stall (React: CL * exp(-0.1 * excess))
        excess_aoa = angle_of_attack - effective_stall
        CL_at_stall = cl_alpha * (effective_stall - zero_lift_angle) * re_lift_factor
        CL = CL_at_stall * np.exp(-0.1 * excess_aoa)

    # Drag build-up (React: Prandtl-Schlichting Cf + base + induced + thickness)
    cf = 0.074 / (Re ** 0.2) if Re > 0 else 0.004
    base_drag = 0.002 + cf
    induced_drag = 0.01 * CL * CL
    thickness_drag = max_thickness * 0.05
    CD = (base_drag + induced_drag + thickness_drag) * re_drag_factor

    # AoA-dependent drag rise past 12° (React: CD * (1 + (AoA-12)*0.05))
    if angle_of_attack > 12:
        CD = CD * (1 + (angle_of_attack - 12) * 0.05)

    # Post-stall drag rise (React: CD * (1 + excess*0.15))
    if is_stalled:
        CD = CD * (1 + (angle_of_attack - effective_stall) * 0.15)

    return CL, CD, is_stalled


def compute_forces(rho, V, chord, CL, CD):
    """
    Compute Lift and Drag per meter span (matches React).
    L = 0.5 × ρ × V² × chord × CL
    D = 0.5 × ρ × V² × chord × CD
    """
    q = 0.5 * rho * V ** 2
    L = q * chord * CL
    D = q * chord * CD
    return L, D


# ---------------------------------------------------------------------------
# Predefined airfoil database with properties
# ---------------------------------------------------------------------------

def get_predefined_airfoils() -> dict:
    """Return a dict of predefined airfoil data sets with properties."""
    
    airfoils = {}
    
    # NACA 0012 (symmetric)
    airfoils["NACA 0012"] = {
        "maxCamber": 0.0,
        "maxThickness": 0.12,
        "stallAngle": 14,
        "description": "Symmetric airfoil, zero camber"
    }
    
    # NACA 2412 (cambered)
    airfoils["NACA 2412"] = {
        "maxCamber": 0.02,
        "maxThickness": 0.12,
        "stallAngle": 14,
        "description": "2% camber at 40% chord, 12% thickness"
    }
    
    # NACA 4412 (high camber)
    airfoils["NACA 4412"] = {
        "maxCamber": 0.04,
        "maxThickness": 0.12,
        "stallAngle": 14,
        "description": "4% camber at 40% chord, 12% thickness"
    }
    
    # Clark Y
    airfoils["Clark Y"] = {
        "maxCamber": 0.034,
        "maxThickness": 0.117,
        "stallAngle": 14,
        "description": "Classic general aviation airfoil"
    }
    
    # Selig S1223 (high-lift)
    airfoils["Selig S1223"] = {
        "maxCamber": 0.048,
        "maxThickness": 0.086,
        "stallAngle": 13,
        "description": "High-lift, low Reynolds number airfoil"
    }
    
    # Eppler E205 (low-drag)
    airfoils["Eppler E205"] = {
        "maxCamber": 0.035,
        "maxThickness": 0.11,
        "stallAngle": 14,
        "description": "Low-drag laminar flow airfoil"
    }
    
    return airfoils


def get_all_airfoils() -> dict:
    """Merge predefined airfoils with any custom airfoils stored in session."""
    airfoils = get_predefined_airfoils()
    if "custom_airfoils" in st.session_state:
        airfoils.update(st.session_state["custom_airfoils"])
    return airfoils


# ---------------------------------------------------------------------------
# Chart styling helper
# ---------------------------------------------------------------------------

def style_chart(fig, ax, title, xlabel, ylabel):
    """Apply dark theme styling to matplotlib charts."""
    ax.set_xlabel(xlabel, fontsize=10, color='#888')
    ax.set_ylabel(ylabel, fontsize=10, color='#888')
    ax.set_title(title, fontsize=11, fontweight='600', color='#ccc', pad=12)
    ax.grid(True, linestyle='--', alpha=0.15, color='#555')
    ax.set_facecolor('#11131f')
    fig.patch.set_facecolor('#0f1117')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2d3148')
    ax.spines['bottom'].set_color('#2d3148')
    ax.tick_params(colors='#888', labelsize=8)
    ax.legend(fontsize=8, framealpha=0.3, loc='best', labelcolor='#aaa')
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Page: Lift and Drag Visualizer
# ---------------------------------------------------------------------------

def card_html(label, value, color):
    return f"""
    <div class="metric-card" style="border-top: 3px solid {color};">
        <p class="label">{label}</p>
        <p class="value" style="color: {color};">{value}</p>
    </div>
    """

def page_visualizer():
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.header("Lift & Drag Simulator")
    with col_status:
        airfoils_all = get_all_airfoils()
        st.markdown(f'<p style="color:#888;font-size:0.8rem;text-align:right;">{len(airfoils_all)} airfoils loaded</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card"><p>Select an airfoil and adjust flight parameters to see real-time aerodynamic calculations.</p></div>', unsafe_allow_html=True)

    airfoils = get_all_airfoils()
    airfoil_names = list(airfoils.keys())

    col_sel, col_info = st.columns([2, 1])
    with col_sel:
        selected = st.selectbox("Airfoil", airfoil_names, label_visibility="collapsed")
    
    foil = airfoils[selected]
    
    with col_info:
        st.markdown(f'<div style="background:#1a1c2a;border:1px solid #252836;border-radius:8px;padding:0.6rem 1rem;text-align:center;"><span style="color:#888;font-size:0.75rem;">{foil["description"]}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<p class="param-label">Flight Parameters</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        aoa = st.slider("Angle of Attack (°)", min_value=-5.0, max_value=20.0, value=5.0, step=0.5)
        velocity = st.slider("Velocity (m/s)", min_value=5.0, max_value=100.0, value=30.0, step=1.0)

    with col2:
        rho = st.number_input("Air Density (kg/m³)", min_value=0.1, max_value=2.0, value=1.225, step=0.001, format="%.3f")
        chord = st.slider("Chord Length (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.05)

    with col3:
        span = st.slider("Wing Span (m)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)

    S = chord * span

    st.markdown("---")

    max_camber = foil["maxCamber"]
    max_thickness = foil["maxThickness"]
    stall_angle = foil["stallAngle"]
    
    CL, CD, is_stalled = compute_coefficients(aoa, max_camber, max_thickness, stall_angle, velocity, rho, chord)
    L, D = compute_forces(rho, velocity, chord, CL, CD)
    
    LD_ratio = CL / CD if CD > 0 else 0

    if is_stalled:
        st.markdown(f"""
        <div class="stall-warning-modern">
            <span class="icon">⚠️</span>
            <div>
                <div class="text">STALL — AoA exceeds effective stall angle</div>
                <div class="sub">Lift reduced, drag increased significantly</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<p class="param-label">Aerodynamic Coefficients</p>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CL", f"{CL:.4f}")
    c2.metric("CD", f"{CD:.5f}")
    c3.metric("L/D", f"{LD_ratio:.1f}")
    c4.metric("Reynolds #", f"{(rho * velocity * chord / AIR_VISCOSITY):,.0f}")

    st.markdown('<p class="param-label" style="margin-top:1rem;">Forces (per meter span)</p>', unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    f1.metric("Lift", f"{L:,.1f} N/m")
    f2.metric("Drag", f"{D:,.1f} N/m")
    f3.metric("Wing Area", f"{S:.2f} m²")

    st.markdown("---")

    st.markdown('<p class="param-label">Characteristic Curves</p>', unsafe_allow_html=True)
    
    alpha_range = np.linspace(-5, 20, 100)
    CL_curve = []
    CD_curve = []
    
    for a in alpha_range:
        cl, cd, _ = compute_coefficients(a, max_camber, max_thickness, stall_angle, velocity, rho, chord)
        CL_curve.append(cl)
        CD_curve.append(cd)
    
    CL_curve = np.array(CL_curve)
    CD_curve = np.array(CD_curve)
    
    chart1, chart2, chart3 = st.columns(3)
    
    with chart1:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.plot(alpha_range, CL_curve, '-', color='#6c7bff', linewidth=2, label=selected)
        ax1.axvline(x=stall_angle, color='#e53e3e', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Stall ({stall_angle}°)')
        ax1.plot(aoa, CL, 'o', color='#fc8181', markersize=8, zorder=5, label='Operating Point')
        ax1.axhline(y=0, color='#444', linestyle='-', linewidth=0.5)
        style_chart(fig1, ax1, "CL vs Angle of Attack", "Angle of Attack α (°)", "CL")
        st.pyplot(fig1)
        plt.close(fig1)
    
    with chart2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.plot(alpha_range, CD_curve, '-', color='#ed8a3e', linewidth=2, label=selected)
        ax2.axvline(x=12, color='#a06cd5', linestyle='--', linewidth=1.5, alpha=0.7, label='Drag onset (12°)')
        ax2.plot(aoa, CD, 'o', color='#fc8181', markersize=8, zorder=5, label='Operating Point')
        ax2.axhline(y=0, color='#444', linestyle='-', linewidth=0.5)
        style_chart(fig2, ax2, "CD vs Angle of Attack", "Angle of Attack α (°)", "CD")
        st.pyplot(fig2)
        plt.close(fig2)
    
    with chart3:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.plot(CD_curve, CL_curve, '-', color='#48bb78', linewidth=2, label=selected)
        ax3.plot(CD, CL, 'o', color='#fc8181', markersize=8, zorder=5, label='Operating Point')
        ax3.axhline(y=0, color='#444', linestyle='-', linewidth=0.5)
        style_chart(fig3, ax3, "Drag Polar (CL vs CD)", "Drag Coefficient CD", "Lift Coefficient CL")
        st.pyplot(fig3)
        plt.close(fig3)

  

# ---------------------------------------------------------------------------
# Page: Custom Airfoil Upload
# ---------------------------------------------------------------------------

def page_custom_airfoil():
    st.header("Custom Airfoil Upload")
    
    st.markdown('<div class="info-card"><p>Paste airfoil coordinate data to extract geometry and use it in the simulator.</p></div>', unsafe_allow_html=True)

    if "custom_airfoils" not in st.session_state:
        st.session_state["custom_airfoils"] = {}

    col_input, col_preview = st.columns([1, 1])
    
    with col_input:
        name = st.text_input("Airfoil Name", placeholder="e.g., My Custom Airfoil")
        
        st.markdown("""
        <div style="background:#1a1c2a;border:1px solid #252836;border-radius:8px;padding:0.75rem 1rem;margin:0.75rem 0;">
            <p style="margin:0;color:#888;font-size:0.8rem;">
                Format: <code style="color:#6c7bff;">x y</code> per line. Space, tab, or comma delimiters work.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        coord_text = st.text_area(
            "Paste Coordinates",
            height=280,
            placeholder="1.0000  0.0013\n0.9500  0.0074\n0.9000  0.0126\n..."
        )
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            preview_clicked = st.button("Preview", use_container_width=True)
        with col_b2:
            save_clicked = st.button("Save Airfoil", type="primary", use_container_width=True)
        
        if preview_clicked:
            if coord_text.strip():
                x, y, err = parse_airfoil_coordinates(coord_text)
                if err:
                    st.error(err)
                else:
                    st.session_state["preview_coords"] = (x, y)
                    st.session_state["preview_name"] = name.strip() if name.strip() else "Unnamed"
                    st.success(f"Parsed {len(x)} points")
            else:
                st.warning("Paste coordinate data first.")
        
        if save_clicked:
            if not name or not name.strip():
                st.error("Name is required.")
            elif not coord_text.strip():
                st.error("Coordinate data is required.")
            else:
                x, y, err = parse_airfoil_coordinates(coord_text)
                if err:
                    st.error(err)
                else:
                    try:
                        geometry = extract_airfoil_geometry(x, y)
                        st.session_state["custom_airfoils"][name.strip()] = {
                            "maxCamber": geometry["maxCamber"],
                            "maxThickness": geometry["maxThickness"],
                            "stallAngle": geometry["stallAngle"],
                            "coordinates": {"x": x.tolist(), "y": y.tolist()},
                            "description": "Custom uploaded airfoil"
                        }
                        st.success(f"Saved \"{name.strip()}\"!")
                        if "preview_coords" in st.session_state:
                            del st.session_state["preview_coords"]
                    except Exception as ex:
                        st.error(f"Analysis error: {ex}")
    
    with col_preview:
        if "preview_coords" in st.session_state:
            x, y = st.session_state["preview_coords"]
            preview_name = st.session_state.get("preview_name", "Preview")
            
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(x, y, '-', color='#6c7bff', linewidth=1.5, marker='.', markersize=2)
            ax.fill(x, y, alpha=0.2, color='#6c7bff')
            ax.set_aspect('equal')
            ax.set_xlabel('x/c', fontsize=9, color='#888')
            ax.set_ylabel('y/c', fontsize=9, color='#888')
            ax.set_title(preview_name, fontsize=11, fontweight='600', color='#ccc')
            ax.grid(True, linestyle='--', alpha=0.15, color='#555')
            ax.axhline(y=0, color='#444', linestyle='-', linewidth=0.5)
            ax.set_facecolor('#11131f')
            fig.patch.set_facecolor('#0f1117')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2d3148')
            ax.spines['bottom'].set_color('#2d3148')
            ax.tick_params(colors='#888', labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            try:
                geometry = extract_airfoil_geometry(x, y)
                gp1, gp2, gp3, gp4 = st.columns(4)
                gp1.metric("Max Thickness", f"{geometry['maxThickness']*100:.1f}%")
                gp2.metric("Max Camber", f"{geometry['maxCamber']*100:.2f}%")
                gp3.metric("Est. Stall", f"{geometry['stallAngle']:.1f}°")
                zero_lift = -geometry['maxCamber'] * 80
                gp4.metric("Zero-Lift α", f"{zero_lift:.2f}°")
            except Exception as ex:
                st.warning(f"Geometry extraction failed: {ex}")
        else:
            st.markdown("""
            <div style="background:#1a1c2a;border:1px dashed #2d3148;border-radius:10px;padding:3rem 1.5rem;text-align:center;">
                <p style="color:#555;font-size:2.5rem;margin:0;">📐</p>
                <p style="color:#666;margin:0.75rem 0 0 0;">Click <b style="color:#6c7bff;">Preview</b> to see the airfoil shape</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.get("custom_airfoils"):
        st.markdown('<p class="param-label">Saved Custom Airfoils</p>', unsafe_allow_html=True)
        
        airfoil_items = list(st.session_state["custom_airfoils"].items())
        cols = st.columns(min(3, len(airfoil_items)))
        
        for idx, (foil_name, data) in enumerate(airfoil_items):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style="background:#1a1c2a;border:1px solid #252836;border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.75rem;">
                    <p style="margin:0 0 0.25rem 0;color:#6c7bff;font-weight:600;">{foil_name}</p>
                    <p style="margin:0;color:#666;font-size:0.8rem;">
                        Thickness: {data['maxThickness']*100:.1f}% &nbsp;|&nbsp; Camber: {data['maxCamber']*100:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Delete", key=f"del_{foil_name}"):
                    del st.session_state["custom_airfoils"][foil_name]
                    st.rerun()
    else:
        st.markdown('<div class="info-card"><p>No custom airfoils saved yet.</p></div>', unsafe_allow_html=True)

 
# Page: Theory
# ---------------------------------------------------------------------------

def page_theory():
    st.header("Theory")
    
    st.markdown('<div class="info-card"><p>Fundamental aerodynamics concepts for understanding airfoil behavior.</p></div>', unsafe_allow_html=True)
    
    topics = [
        ("Airfoil", "The cross-sectional shape of a wing designed to produce lift. The curved upper surface and flatter lower surface create a pressure difference that generates upward force."),
        ("Lift", "The upward force that keeps an aircraft in the air. Air moves faster over the curved top surface, creating lower pressure above and higher pressure below (Bernoulli's principle)."),
        ("Drag", "The backward force opposing motion through air. Drag increases with velocity and angle of attack. Minimizing drag is key to efficient flight."),
        ("Angle of Attack (α)", "The angle between the wing's chord line and the oncoming airflow. Increasing α increases lift up to the stall point, beyond which lift drops sharply."),
        ("Stall", "Flow separation from the upper wing surface at excessive angle of attack. Lift drops suddenly and drag increases dramatically. A critical safety concept in aviation."),
        ("Lift Coefficient (CL)", "A dimensionless number representing lifting efficiency. Higher CL means more lift per unit speed and area. Depends on airfoil shape and angle of attack."),
        ("Drag Coefficient (CD)", "A dimensionless number measuring aerodynamic resistance. Lower CD means less drag. Increases at high angles of attack and after stall."),
        ("L/D Ratio", "Lift-to-drag ratio measures aerodynamic efficiency. Gliders achieve 20-60, commercial aircraft 15-20, fighters 5-10. Higher is better for range and fuel economy."),
        ("Chord Line", "The straight line from leading edge to trailing edge. Chord length is the reference dimension for all airfoil measurements and calculations."),
        ("Pressure Difference", "The fundamental mechanism of lift — low pressure above the wing and high pressure below creates a net upward force described by Bernoulli's equation."),
        ("Flow Separation", "When airflow detaches from the wing surface, creating turbulence in the wake. This is the physical cause of stall, reducing lift and increasing drag."),
    ]
    
    for title, desc in topics:
        st.markdown(f"""
        <div class="theory-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Compare
# ---------------------------------------------------------------------------

def page_compare():
    st.header("Compare Airfoils")
    st.markdown('<div class="info-card"><p>Compare aerodynamic performance of two airfoils side by side.</p></div>', unsafe_allow_html=True)
    
    airfoils = get_all_airfoils()
    names = list(airfoils.keys())
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="param-label">Airfoil A</p>', unsafe_allow_html=True)
        sel_a = st.selectbox("Airfoil A", names, key="comp_a", label_visibility="collapsed")
    with col_b:
        st.markdown('<p class="param-label">Airfoil B</p>', unsafe_allow_html=True)
        sel_b = st.selectbox("Airfoil B", names, key="comp_b", index=min(1, len(names)-1), label_visibility="collapsed")
    
    st.markdown("---")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        comp_aoa = st.slider("AoA (°)", -5.0, 20.0, 5.0, 0.5)
    with col_p2:
        comp_vel = st.slider("Velocity (m/s)", 5.0, 100.0, 30.0, 1.0)
    with col_p3:
        comp_chord = st.slider("Chord (m)", 0.1, 2.0, 0.5, 0.05)
    
    foil_a = airfoils[sel_a]
    foil_b = airfoils[sel_b]
    
    CL_a, CD_a, stall_a = compute_coefficients(comp_aoa, foil_a["maxCamber"], foil_a["maxThickness"], foil_a["stallAngle"], comp_vel, 1.225, comp_chord)
    CL_b, CD_b, stall_b = compute_coefficients(comp_aoa, foil_b["maxCamber"], foil_b["maxThickness"], foil_b["stallAngle"], comp_vel, 1.225, comp_chord)
    LD_a = CL_a / CD_a if CD_a > 0 else 0
    LD_b = CL_b / CD_b if CD_b > 0 else 0
    
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"CL - {sel_a}", f"{CL_a:.4f}", f"{CL_b - CL_a:+.4f}")
    c2.metric(f"CD - {sel_a}", f"{CD_a:.5f}", f"{CD_b - CD_a:+.5f}")
    c3.metric(f"L/D - {sel_a}", f"{LD_a:.1f}", f"{LD_b - LD_a:+.1f}")
    c4.metric("Stall", f"{'⚠️' if stall_a else '✓'}", f"{'⚠️' if stall_b else '✓'}")
    
    st.markdown("---")
    comp_range = np.linspace(-5, 20, 100)
    CL_a_curve = np.array([compute_coefficients(a, foil_a["maxCamber"], foil_a["maxThickness"], foil_a["stallAngle"], comp_vel, 1.225, comp_chord)[0] for a in comp_range])
    CL_b_curve = np.array([compute_coefficients(a, foil_b["maxCamber"], foil_b["maxThickness"], foil_b["stallAngle"], comp_vel, 1.225, comp_chord)[0] for a in comp_range])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(comp_range, CL_a_curve, '-', color='#6c7bff', linewidth=2, label=sel_a)
    ax.plot(comp_range, CL_b_curve, '-', color='#ed8a3e', linewidth=2, label=sel_b)
    ax.axhline(y=0, color='#444', linestyle='-', linewidth=0.5)
    ax.set_facecolor('#11131f')
    fig.patch.set_facecolor('#0f1117')
    ax.grid(True, linestyle='--', alpha=0.15, color='#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2d3148')
    ax.spines['bottom'].set_color('#2d3148')
    ax.tick_params(colors='#888', labelsize=9)
    ax.set_xlabel("Angle of Attack (°)", color='#888', fontsize=10)
    ax.set_ylabel("CL", color='#888', fontsize=10)
    ax.set_title("CL Comparison", color='#ccc', fontsize=11, fontweight='600')
    ax.legend(fontsize=9, labelcolor='#aaa')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: Quiz
# ---------------------------------------------------------------------------

QUIZ_DATA = [
    {"q": "What is the primary force that opposes an aircraft's motion through air?", "o": ["Lift", "Drag", "Thrust", "Weight"], "a": 1},
    {"q": "Stall occurs when:", "o": ["Engine fails", "Angle of attack exceeds critical value", "Speed is too high", "Wing breaks"], "a": 1},
    {"q": "The lift coefficient CL typically increases with:", "o": ["Decreasing airspeed", "Increasing angle of attack", "Decreasing wing area", "Increasing drag"], "a": 1},
    {"q": "A cambered airfoil produces:", "o": ["Zero lift at 0° AoA", "Some lift even at 0° AoA", "More drag than a symmetric one", "No stall"], "a": 1},
    {"q": "What does L/D ratio measure?", "o": ["Speed", "Weight", "Aerodynamic efficiency", "Engine power"], "a": 2},
    {"q": "Reynolds number represents the ratio of:", "o": ["Inertial to viscous forces", "Lift to drag", "Pressure to density", "Speed to chord"], "a": 0},
    {"q": "The zero-lift angle for a symmetric airfoil is:", "o": ["-5°", "0°", "5°", "Depends on speed"], "a": 1},
    {"q": "At low Reynolds numbers, an airfoil typically:", "o": ["Performs better", "Has lower CL and higher CD", "Has no stall", "Generates more lift"], "a": 1},
]

def page_quiz():
    st.header("Quiz")
    st.markdown('<div class="info-card"><p>Test your knowledge of aerodynamics fundamentals.</p></div>', unsafe_allow_html=True)
    
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    
    for i, item in enumerate(QUIZ_DATA):
        st.markdown(f"**{i+1}. {item['q']}**")
        ans = st.radio("", item["o"], key=f"quiz_{i}", index=None, label_visibility="collapsed")
        if ans is not None:
            st.session_state.quiz_answers[i] = item["o"].index(ans)
    
    if st.button("Submit Answers", type="primary", use_container_width=False):
        st.session_state.quiz_submitted = True
        correct = 0
        for i, item in enumerate(QUIZ_DATA):
            if st.session_state.quiz_answers.get(i) == item["a"]:
                correct += 1
        pct = correct / len(QUIZ_DATA) * 100
        st.markdown(f"<div style='background:#1a1c2a;border:1px solid #252836;border-radius:10px;padding:1.25rem;text-align:center;margin-top:1rem;'><p style='color:#6c7bff;font-size:2rem;font-weight:700;margin:0;'>{correct}/{len(QUIZ_DATA)}</p><p style='color:#888;margin:0;'>{pct:.0f}% — {'Excellent!' if pct >= 80 else 'Good job!' if pct >= 60 else 'Keep studying!'}</p></div>", unsafe_allow_html=True)

    if st.session_state.quiz_submitted:
        for i, item in enumerate(QUIZ_DATA):
            user_ans = st.session_state.quiz_answers.get(i)
            is_correct = user_ans == item["a"]
            icon = "✅" if is_correct else "❌"
            correct_text = f" → Correct: {item['o'][item['a']]}" if not is_correct else ""
            st.markdown(f"<p style='color:#ccc;font-size:0.85rem;margin:0.2rem 0;'>{icon} {i+1}. {item['q']}{correct_text}</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about():
    st.header("About")
    
    st.markdown("""
    <div style="background:#1a1c2a;border:1px solid #252836;border-radius:10px;padding:1.5rem;margin-bottom:1rem;">
        <h4 style="color:#6c7bff;margin:0 0 0.5rem 0;">Interactive Airfoil Lift & Drag Visualizer</h4>
        <p style="color:#9ca3af;margin:0;font-size:0.9rem;line-height:1.7;">
            An interdisciplinary educational tool combining Aeronautical Engineering and Computer Science 
            to help students understand how lift and drag are produced on different airfoil shapes through 
            interactive visualization.
        </p>
    </div>
    <div style="background:#1a1c2a;border:1px solid #252836;border-radius:10px;padding:1.5rem;margin-bottom:1rem;">
        <h4 style="color:#6c7bff;margin:0 0 0.5rem 0;">How It Works</h4>
        <p style="color:#9ca3af;margin:0;font-size:0.9rem;line-height:1.7;">
            The simulator uses simplified aerodynamic models based on thin airfoil theory. 
            Lift coefficient is calculated from angle of attack with Reynolds number corrections. 
            Drag is modeled as a combination of skin friction (Prandtl-Schlichting), induced drag, 
            and thickness effects. Stall is simulated with angle reduction at low Reynolds numbers.
        </p>
    </div>
    <div style="background:#1a1c2a;border:1px solid #252836;border-radius:10px;padding:1.5rem;margin-bottom:1rem;">
        <h4 style="color:#6c7bff;margin:0 0 0.5rem 0;">Limitations</h4>
        <p style="color:#9ca3af;margin:0;font-size:0.9rem;line-height:1.7;">
            This is an educational visualization, not a CFD tool. Results are not suitable for 
            real aircraft design. The aerodynamic models use empirical curve-fits and simplified 
            equations — for accurate analysis, use XFOIL, panel methods, or full CFD.
        </p>
    </div>
    <div style="background:#1a1c2a;border:1px solid #252836;border-radius:10px;padding:1.5rem;">
        <h4 style="color:#6c7bff;margin:0 0 0.5rem 0;">Tech Stack</h4>
        <p style="color:#9ca3af;margin:0;font-size:0.9rem;line-height:1.7;">
            Python • Streamlit • NumPy • Matplotlib<br>
            Aerodynamic models ported from the React/Node.js version.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------

def page_home():
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0;">
        <div style="font-size:3rem;">✈️</div>
        <h2 style="color:#f0f0f0;margin:0.5rem 0 0.25rem 0;">Airfoil Lift & Drag Visualizer</h2>
        <p style="color:#888;margin:0;font-size:0.95rem;">Explore aerodynamics through interactive simulation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    features = [
        ("Simulator", "Adjust angle of attack, velocity, and airfoil shape to see real-time CL, CD, and force calculations with interactive charts."),
        ("Custom Airfoil", "Upload your own airfoil coordinates to analyze geometry — thickness, camber, stall angle, and zero-lift angle."),
        ("Compare", "Compare two airfoils side-by-side with CL curves and performance metrics."),
        ("Learn", "Study fundamental aerodynamics concepts — lift, drag, stall, pressure difference, and more."),
        ("Quiz", "Test your knowledge with multiple-choice questions on aerodynamics."),
    ]
    
    cols = st.columns(3)
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#1a1c2a;border:1px solid #252836;border-radius:10px;padding:1.25rem;margin-bottom:1rem;height:180px;">
                <h4 style="color:#6c7bff;margin:0 0 0.5rem 0;">{title}</h4>
                <p style="color:#9ca3af;margin:0;font-size:0.85rem;line-height:1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Airfoil Lift & Drag Visualizer",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()

    st.sidebar.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.5rem 0;">
        <div style="font-size:2rem;margin:0;">✈️</div>
        <div style="color:#e0e0e0;font-weight:700;font-size:1rem;margin:0.25rem 0 0 0;">Airfoil Visualizer</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Simulator", "Custom Airfoil", "Compare", "Learn", "Quiz", "About"],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    
    num_predefined = len(get_predefined_airfoils())
    num_custom = len(st.session_state.get("custom_airfoils", {}))
    
    st.sidebar.markdown(f"""
    <div style="background:#1e2030;padding:0.75rem 1rem;border-radius:8px;border:1px solid #252836;">
        <p style="color:#888;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 0.35rem 0;">Airfoils</p>
        <p style="color:#b0b3c0;font-size:0.85rem;margin:0;">
            Predefined: <b style="color:#6c7bff;">{num_predefined}</b> &nbsp;|&nbsp; Custom: <b style="color:#6c7bff;">{num_custom}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    st.sidebar.caption("Educational aerodynamics simulator • Streamlit")

    st.markdown(f"""
    <div style="text-align:center;margin-bottom:1.5rem;">
        <h1 style="font-size:1.6rem;margin:0;color:#f0f0f0;">Airfoil Lift & Drag Visualizer</h1>
        <p style="color:#666;font-size:0.85rem;margin:0.35rem 0 0 0;">Interactive aerodynamic simulation for engineering education</p>
    </div>
    """, unsafe_allow_html=True)

    if page == "Home":
        page_home()
    elif page == "Simulator":
        page_visualizer()
    elif page == "Custom Airfoil":
        page_custom_airfoil()
    elif page == "Compare":
        page_compare()
    elif page == "Learn":
        page_theory()
    elif page == "Quiz":
        page_quiz()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
