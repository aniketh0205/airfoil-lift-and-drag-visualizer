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
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Card-like sections */
        .stMetric {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 12px;
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stMetric label {
            color: rgba(255,255,255,0.85) !important;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: white !important;
            font-weight: 700;
        }
        
        /* Headers styling */
        h1 {
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800 !important;
        }
        
        h2, h3 {
            color: #4a5568;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0;
        }
        
        [data-testid="stSidebar"] .stRadio label {
            color: #e2e8f0 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        
        /* Selectbox and sliders */
        .stSelectbox > div > div {
            border-radius: 8px;
        }
        
        /* Info boxes - Dark themed */
        .stAlert {
            border-radius: 12px;
            border: none;
        }
        
        /* Expander styling - Dark themed */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            border-radius: 8px;
            color: #e2e8f0 !important;
        }
        
        /* Text area - Dark themed */
        .stTextArea textarea {
            border-radius: 8px;
            border: 2px solid #4a5568;
            background-color: #1a202c;
            color: #e2e8f0;
            font-family: 'Courier New', monospace;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
        }
        
        .stTextArea textarea::placeholder {
            color: #718096;
        }
        
        /* Text input - Dark themed */
        .stTextInput input {
            border-radius: 8px;
            border: 2px solid #4a5568;
            background-color: #1a202c;
            color: #e2e8f0;
        }
        
        .stTextInput input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
        }
        
        .stTextInput input::placeholder {
            color: #718096;
        }
        
        /* Number input - Dark themed */
        .stNumberInput input {
            border-radius: 8px;
            border: 2px solid #4a5568;
            background-color: #1a202c;
            color: #e2e8f0;
        }
        
        .stNumberInput input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
        }
        
        /* Divider */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 2rem 0;
        }
        
        /* Dark info boxes */
        .dark-info-box {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            color: #e2e8f0;
            margin-bottom: 1.5rem;
        }
        
        .dark-info-box p {
            color: #e2e8f0;
            margin: 0;
        }
        
        /* Theory card styling */
        .theory-card {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        
        .theory-card h4 {
            color: #a78bfa;
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
        }
        
        .theory-card p {
            color: #cbd5e0;
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* Stall warning */
        .stall-warning {
            background: linear-gradient(135deg, #c53030 0%, #9b2c2c 100%);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
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
    """Apply consistent professional styling to matplotlib charts."""
    ax.set_xlabel(xlabel, fontsize=11, fontweight='500', color='#4a5568')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='500', color='#4a5568')
    ax.set_title(title, fontsize=13, fontweight='600', color='#2d3748', pad=15)
    ax.grid(True, linestyle='--', alpha=0.4, color='#a0aec0')
    ax.set_facecolor('#fafbfc')
    fig.patch.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e0')
    ax.spines['bottom'].set_color('#cbd5e0')
    ax.tick_params(colors='#4a5568', labelsize=9)
    ax.legend(fontsize=9, framealpha=0.95, loc='best')
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Page: Lift and Drag Visualizer
# ---------------------------------------------------------------------------

def page_visualizer():
    st.header("✈️ Lift & Drag Simulator")
    
    st.markdown("""
    <div class="dark-info-box">
        <p>🎯 <b>Interactive Flight Simulator</b> — Select an airfoil and adjust parameters to calculate 
        lift and drag forces using simplified aerodynamic equations.</p>
    </div>
    """, unsafe_allow_html=True)

    airfoils = get_all_airfoils()
    airfoil_names = list(airfoils.keys())

    # --- Airfoil selector ---
    col_select, col_info = st.columns([2, 1])
    with col_select:
        selected = st.selectbox("🔧 Select Airfoil", airfoil_names, 
                               help="Choose from predefined airfoils or your custom uploads")
    
    foil = airfoils[selected]
    
    with col_info:
        st.markdown(f"""
        <div style="background: #2d3748; padding: 0.75rem; border-radius: 8px; margin-top: 1.5rem;">
            <small style="color: #a0aec0;">
                Camber: <b style="color:#e2e8f0;">{foil['maxCamber']*100:.1f}%</b> | 
                Thickness: <b style="color:#e2e8f0;">{foil['maxThickness']*100:.1f}%</b>
            </small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Input controls ---
    st.subheader("⚙️ Flight Parameters")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📐 Angle of Attack**")
        aoa = st.slider("Angle of Attack (°)", min_value=-5.0, max_value=20.0, value=5.0, step=0.5,
                       help="Wing angle relative to airflow")
        
        st.markdown("**🌬️ Velocity**")
        velocity = st.slider("Velocity (m/s)", min_value=5.0, max_value=100.0, value=30.0, step=1.0,
                            help="Airspeed in meters per second")

    with col2:
        st.markdown("**🌡️ Air Density**")
        rho = st.number_input("Air Density (kg/m³)", min_value=0.1, max_value=2.0, 
                              value=1.225, step=0.001, format="%.3f",
                              help="Sea level standard: 1.225 kg/m³")
        
        st.markdown("**📏 Chord Length**")
        chord = st.slider("Chord Length (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.05,
                         help="Distance from leading to trailing edge")

    with col3:
        st.markdown("**📐 Wing Span**")
        span = st.slider("Wing Span (m)", min_value=0.5, max_value=10.0, value=3.0, step=0.1,
                        help="Tip-to-tip wing length")

    # Wing area (display only, forces use per-meter like React)
    S = chord * span

    st.markdown("---")

    # --- Compute coefficients and forces ---
    max_camber = foil["maxCamber"]
    max_thickness = foil["maxThickness"]
    stall_angle = foil["stallAngle"]
    
    CL, CD, is_stalled = compute_coefficients(aoa, max_camber, max_thickness, stall_angle, velocity, rho, chord)
    L, D = compute_forces(rho, velocity, chord, CL, CD)
    
    # L/D ratio
    LD_ratio = CL / CD if CD > 0 else 0

    # --- Stall Warning ---
    if is_stalled:
        st.markdown("""
        <div class="stall-warning">
            <h3 style="margin: 0;">⚠️ STALL WARNING ⚠️</h3>
            <p style="margin: 0.5rem 0 0 0;">Angle of attack exceeds stall angle! Lift is reduced and drag is increased.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # --- Coefficients Display ---
    st.subheader("📊 Aerodynamic Coefficients")
    
    coef1, coef2, coef3 = st.columns(3)
    
    coef1.markdown(f"""
    <div style="background: linear-gradient(135deg, #4c51bf 0%, #6b46c1 100%); 
                padding: 1.5rem; border-radius: 12px; text-align: center;
                box-shadow: 0 4px 15px rgba(76, 81, 191, 0.3);">
        <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin: 0;">LIFT COEFFICIENT</p>
        <p style="color: white; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">
            C<sub>L</sub> = {CL:.4f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    coef2.markdown(f"""
    <div style="background: linear-gradient(135deg, #d69e2e 0%, #b7791f 100%); 
                padding: 1.5rem; border-radius: 12px; text-align: center;
                box-shadow: 0 4px 15px rgba(214, 158, 46, 0.3);">
        <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin: 0;">DRAG COEFFICIENT</p>
        <p style="color: white; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">
            C<sub>D</sub> = {CD:.5f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    coef3.markdown(f"""
    <div style="background: linear-gradient(135deg, #38a169 0%, #2f855a 100%); 
                padding: 1.5rem; border-radius: 12px; text-align: center;
                box-shadow: 0 4px 15px rgba(56, 161, 105, 0.3);">
        <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin: 0;">L/D RATIO</p>
        <p style="color: white; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">
            {LD_ratio:.1f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Forces Display ---
    st.subheader("🎯 Aerodynamic Forces")
    
    force1, force2, force3 = st.columns(3)
    
    force1.markdown(f"""
    <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                padding: 1.5rem; border-radius: 12px; text-align: center;
                box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);">
        <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin: 0;">LIFT FORCE</p>
        <p style="color: white; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">
            {L:,.2f} N
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    force2.markdown(f"""
    <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); 
                padding: 1.5rem; border-radius: 12px; text-align: center;
                box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);">
        <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin: 0;">DRAG FORCE</p>
        <p style="color: white; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">
            {D:,.2f} N
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    force3.markdown(f"""
    <div style="background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%); 
                padding: 1.5rem; border-radius: 12px; text-align: center;
                border: 2px solid #4a5568;">
        <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin: 0;">WING AREA</p>
        <p style="color: white; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">
            {S:.2f} m²
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Charts ---
    st.subheader("📈 Aerodynamic Characteristic Curves")
    
    # Generate data for charts
    alpha_range = np.linspace(-5, 20, 100)
    CL_curve = []
    CD_curve = []
    
    for a in alpha_range:
        cl, cd, _ = compute_coefficients(a, max_camber, max_thickness, stall_angle, velocity, rho, chord)
        CL_curve.append(cl)
        CD_curve.append(cd)
    
    CL_curve = np.array(CL_curve)
    CD_curve = np.array(CD_curve)
    
    # Three charts
    chart1, chart2, chart3 = st.columns(3)
    
    # Chart 1: CL vs Alpha
    with chart1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.plot(alpha_range, CL_curve, '-', color='#667eea', linewidth=2, label=f'{selected}')
        ax1.axvline(x=stall_angle, color='#e53e3e', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Stall ({stall_angle}°)')
        ax1.plot(aoa, CL, 'r*', markersize=18, zorder=5, label=f'Operating Point')
        ax1.axhline(y=0, color='#a0aec0', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=0, color='#a0aec0', linestyle='-', linewidth=0.5, alpha=0.5)
        style_chart(fig1, ax1, "CL vs Angle of Attack", "Angle of Attack α (°)", "CL")
        st.pyplot(fig1)
        plt.close(fig1)
    
    # Chart 2: CD vs Alpha
    with chart2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(alpha_range, CD_curve, '-', color='#ed8936', linewidth=2, label=f'{selected}')
        ax2.axvline(x=12, color='#805ad5', linestyle='--', linewidth=1.5, alpha=0.7, label='High drag onset (12°)')
        ax2.plot(aoa, CD, 'r*', markersize=18, zorder=5, label=f'Operating Point')
        ax2.axvline(x=0, color='#a0aec0', linestyle='-', linewidth=0.5, alpha=0.5)
        style_chart(fig2, ax2, "CD vs Angle of Attack", "Angle of Attack α (°)", "CD")
        st.pyplot(fig2)
        plt.close(fig2)
    
    # Chart 3: CL vs CD (Drag Polar)
    with chart3:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.plot(CD_curve, CL_curve, '-', color='#48bb78', linewidth=2, label=f'{selected}')
        ax3.plot(CD, CL, 'r*', markersize=18, zorder=5, label=f'Operating Point')
        ax3.axhline(y=0, color='#a0aec0', linestyle='-', linewidth=0.5, alpha=0.5)
        style_chart(fig3, ax3, "Drag Polar (CL vs CD)", "Drag Coefficient CD", "Lift Coefficient CL")
        st.pyplot(fig3)
        plt.close(fig3)

  

# ---------------------------------------------------------------------------
# Page: Custom Airfoil Upload
# ---------------------------------------------------------------------------

def page_custom_airfoil():
    st.header("📝 Custom Airfoil Upload")
    
    st.markdown("""
    <div class="dark-info-box">
        <p>🔬 <b>Add any airfoil</b> by pasting its coordinate data. The app will automatically 
        extract geometry properties (camber, thickness) and estimate aerodynamic behavior.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session storage
    if "custom_airfoils" not in st.session_state:
        st.session_state["custom_airfoils"] = {}

    # Two columns layout
    col_input, col_preview = st.columns([1, 1])
    
    with col_input:
        st.subheader("📋 Input Airfoil Data")
        
        name = st.text_input("✏️ Airfoil Name", placeholder="e.g., My Custom Airfoil",
                            help="Give your airfoil a unique name")
        
        st.markdown("""
        <div style="background: #2d3748; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="margin: 0 0 0.5rem 0; color: #a78bfa; font-weight: 600;">
                📄 Coordinate Format
            </p>
            <p style="margin: 0; color: #cbd5e0; font-size: 0.85rem;">
                Each line: <code style="color:#faf089;">x-coordinate  y-coordinate</code><br>
                Supports space, tab, or comma delimiters.<br>
                Non-numeric header lines are automatically ignored.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        coord_text = st.text_area(
            "📐 Paste Airfoil Coordinates",
            height=300,
            placeholder="""NACA 2412
1.0000  0.0013
0.9500  0.0074
0.9000  0.0126
0.8000  0.0214
0.7000  0.0286
0.6000  0.0347
0.5000  0.0398
0.4000  0.0433
0.3000  0.0435
0.2000  0.0375
0.1000  0.0256
0.0000  0.0000
0.1000 -0.0121
0.2000 -0.0180
...""",
            help="Paste coordinate data from airfoil databases like UIUC, Airfoil Tools, etc."
        )
        
        # Parse and preview button
        if st.button("🔍 Preview & Analyze", type="secondary", use_container_width=True):
            if coord_text.strip():
                x, y, err = parse_airfoil_coordinates(coord_text)
                if err:
                    st.error(f"❌ {err}")
                else:
                    st.session_state["preview_coords"] = (x, y)
                    st.session_state["preview_name"] = name.strip() if name.strip() else "Unnamed"
                    st.success(f"✅ Parsed {len(x)} coordinate points successfully!")
            else:
                st.warning("⚠️ Please paste coordinate data first.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Save button
        if st.button("💾 Save Airfoil", type="primary", use_container_width=True):
            errors = []
            
            if not name or not name.strip():
                errors.append("Airfoil name is required.")
            
            if not coord_text.strip():
                errors.append("Coordinate data is required.")
            
            if errors:
                for e in errors:
                    st.error(f"❌ {e}")
            else:
                x, y, err = parse_airfoil_coordinates(coord_text)
                if err:
                    st.error(f"❌ {err}")
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
                        st.success(f"✅ **{name.strip()}** saved successfully!")
                        st.balloons()
                        if "preview_coords" in st.session_state:
                            del st.session_state["preview_coords"]
                    except Exception as ex:
                        st.error(f"❌ Error analyzing airfoil: {str(ex)}")
    
    with col_preview:
        st.subheader("👁️ Preview")
        
        if "preview_coords" in st.session_state:
            x, y = st.session_state["preview_coords"]
            preview_name = st.session_state.get("preview_name", "Preview")
            
            # Plot airfoil shape
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, 'b-', linewidth=1.5, marker='.', markersize=3)
            ax.fill(x, y, alpha=0.3, color='#667eea')
            ax.set_aspect('equal')
            ax.set_xlabel('x/c', fontsize=10)
            ax.set_ylabel('y/c', fontsize=10)
            ax.set_title(f'Airfoil Shape: {preview_name}', fontsize=12, fontweight='600')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Show extracted properties
            try:
                geometry = extract_airfoil_geometry(x, y)
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%); 
                            padding: 1rem; border-radius: 12px; margin-top: 1rem;
                            border-left: 4px solid #48bb78;">
                    <p style="margin: 0 0 0.5rem 0; color: #48bb78; font-weight: 600;">
                        📊 Extracted Geometry
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                p1, p2 = st.columns(2)
                p1.metric("Max Thickness", f"{geometry['maxThickness']*100:.1f}%")
                p2.metric("Max Camber", f"{geometry['maxCamber']*100:.2f}%")
                
                p3, p4 = st.columns(2)
                p3.metric("Est. Stall Angle", f"{geometry['stallAngle']:.1f}°")
                zero_lift = -geometry['maxCamber'] * 80
                p4.metric("Zero-Lift α", f"{zero_lift:.2f}°")
                
            except Exception as ex:
                st.warning(f"Could not extract geometry: {str(ex)}")
        else:
            st.markdown("""
            <div style="background: #2d3748; padding: 3rem 1.5rem; border-radius: 12px; 
                        text-align: center; border: 2px dashed #4a5568;">
                <p style="color: #667eea; font-size: 3rem; margin: 0;">📊</p>
                <p style="color: #a0aec0; margin: 1rem 0 0 0;">
                    Paste coordinates and click <b>Preview & Analyze</b><br>
                    to see the airfoil shape and properties.
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Show existing custom airfoils
    if st.session_state.get("custom_airfoils"):
        st.subheader("📚 Your Custom Airfoils")
        
        airfoil_items = list(st.session_state["custom_airfoils"].items())
        cols = st.columns(min(3, len(airfoil_items)))
        
        for idx, (foil_name, data) in enumerate(airfoil_items):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%); 
                            padding: 1rem; border-radius: 12px; margin-bottom: 1rem;
                            border: 1px solid #4a5568;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #a78bfa;">🔹 {foil_name}</h4>
                    <p style="color: #cbd5e0; font-size: 0.85rem; margin: 0;">
                        Thickness: {data['maxThickness']*100:.1f}% | 
                        Camber: {data['maxCamber']*100:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🗑️ Delete", key=f"del_{foil_name}"):
                    del st.session_state["custom_airfoils"][foil_name]
                    st.rerun()
    else:
        st.markdown("""
        <div class="dark-info-box">
            <p>💡 No custom airfoils yet. Add one above to get started!</p>
        </div>
        """, unsafe_allow_html=True)

 
# Page: Theory
# ---------------------------------------------------------------------------

def page_theory():
    st.header("📖 Theory")
    
    st.markdown("""
    <div class="dark-info-box">
        <p>📚 Learn the fundamentals of aerodynamics with these simple, student-friendly explanations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Theory cards
    st.markdown("""
    <div class="theory-card">
        <h4>✈️ Airfoil</h4>
        <p>An airfoil is the cross-sectional shape of a wing. It is designed to produce lift when air flows over it. 
        The curved top and flatter bottom create a pressure difference that lifts the aircraft.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>⬆️ Lift</h4>
        <p>Lift is the upward force that keeps an aircraft in the air. It is created when air moves faster over 
        the top of the wing than the bottom, causing lower pressure above and higher pressure below.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>➡️ Drag</h4>
        <p>Drag is the force that opposes motion through the air. It acts backward, slowing the aircraft down. 
        Drag increases with speed and angle of attack.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>📐 Angle of Attack (α)</h4>
        <p>The angle of attack is the angle between the wing's chord line and the oncoming airflow. 
        Increasing this angle increases lift — but only up to a point. Too much angle causes stall.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>⚠️ Stall</h4>
        <p>Stall occurs when the angle of attack becomes too high. The airflow separates from the wing surface, 
        lift drops suddenly, and drag increases. This is dangerous and must be avoided.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>📊 Lift Coefficient (CL)</h4>
        <p>The lift coefficient is a number that represents how effectively a wing produces lift. 
        Higher CL means more lift for the same speed and wing size. It depends on shape and angle of attack.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>📉 Drag Coefficient (CD)</h4>
        <p>The drag coefficient measures how much drag a wing creates. Lower CD means less resistance 
        and better efficiency. It increases at high angles of attack.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>⚖️ Lift-to-Drag Ratio (L/D)</h4>
        <p>This ratio tells you how efficient a wing is. A higher L/D ratio means the wing produces 
        more lift for less drag. Gliders have very high L/D ratios (20-60), while fighters are lower (5-10).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>📏 Chord Line</h4>
        <p>The chord line is a straight line from the leading edge (front) to the trailing edge (back) of the wing. 
        The chord length is the distance of this line.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>🔄 Pressure Difference</h4>
        <p>Air moving over a curved wing surface speeds up and creates low pressure on top. 
        The higher pressure below pushes the wing up. This pressure difference is what creates lift.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-card">
        <h4>💨 Flow Separation</h4>
        <p>Flow separation happens when air can no longer follow the wing's curved surface and breaks away. 
        This creates turbulence, reduces lift, and increases drag. It's the main cause of stall.</p>
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

    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2.5rem; margin: 0;">✈️</h1>
        <h2 style="color: #e2e8f0; font-size: 1.1rem; margin: 0.5rem 0;">
            Airfoil Analyzer
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "📍 Navigation",
        ["🔬 Lift & Drag Simulator", "📝 Custom Airfoil Upload", "📖 Theory"],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    
    # Airfoil count indicator
    num_predefined = len(get_predefined_airfoils())
    num_custom = len(st.session_state.get("custom_airfoils", {}))
    
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;">
        <p style="color: #a0aec0; font-size: 0.8rem; margin: 0 0 0.5rem 0;">AIRFOILS AVAILABLE</p>
        <p style="color: #e2e8f0; font-size: 0.95rem; margin: 0;">
            📦 Predefined: <b>{num_predefined}</b><br>
            ✨ Custom: <b>{num_custom}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    st.sidebar.caption(
        "Built with Streamlit, NumPy & Matplotlib.\n\n"
        "Educational aerodynamics simulator."
    )

    # Main title
    st.markdown("""
    <h1 style="text-align: center; font-size: 2.2rem; margin-bottom: 0.5rem;">
        ✈️ Interactive Airfoil Lift & Drag Visualizer
    </h1>
    <p style="text-align: center; color: #718096; margin-bottom: 2rem;">
        Explore aerodynamic principles with interactive simulations
    </p>
    """, unsafe_allow_html=True)

    # Page routing
    if "Lift & Drag Simulator" in page:
        page_visualizer()
    elif "Custom Airfoil Upload" in page:
        page_custom_airfoil()
    else:
        page_theory()


if __name__ == "__main__":
    main()
