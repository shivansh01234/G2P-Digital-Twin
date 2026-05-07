import streamlit as st
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from backend import G2PDigitalTwin

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(page_title="G2P Digital Twin", page_icon="🧬", layout="wide")

# Consulting-Grade Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* KPI Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 15px;
        margin: 20px 0;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 15px;
        padding: 25px;
        flex: 1;
        text-align: center;
        transition: 0.3s;
    }
    .metric-card:hover { border-color: #DEFF9A; transform: translateY(-5px); }
    .metric-label { color: #9CA3AF; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #DEFF9A; font-size: 1.8rem; font-weight: 700; margin-top: 5px; }

    /* Architecture Section */
    .arch-card {
        background: #1F2937;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #DEFF9A;
        margin-bottom: 20px;
    }
    
    /* Sidebar Engineering Badge */
    .sidebar-badge {
        background: linear-gradient(135deg, #DEFF9A, #A3E635);
        color: #064E3B;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE & ENGINE LOADING
# ==========================================
@st.cache_resource(show_spinner="Training Stacking Ensemble on eSOL Data...")
def load_engine():
    return G2PDigitalTwin()

twin = load_engine()

if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'sim_graphs' not in st.session_state: st.session_state.sim_graphs = None
if 'active_graph' not in st.session_state: st.session_state.active_graph = "Product"

# ==========================================
# 3. SIDEBAR (LEAD ENGINEER BADGE & CONFIG)
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-badge">👨‍🔬 Engineer: Shivansh Sahu</div>', unsafe_allow_html=True)
    st.title("G2P Navigation")
    page = st.radio("Navigation Menu:", [
        "🚀 Mission Control", 
        "📈 Factory Telemetry", 
        "🎛️ Control Trajectory",
        "🏗️ Model Architecture"
    ])
    st.markdown("---")
    st.markdown("### Model Config")
    microbe_choice = st.selectbox("Select Target Host", ("E. coli", "Bacillus subtilis", "Saccharomyces cerevisiae", "Pichia pastoris", "Aspergillus niger"))
    hours = st.slider("Duration (h)", 12, 120, 72)
    volume = st.number_input("Tank Volume (L)", value=50000)
    
    # The new Optimizer Logic
    run_opt = st.toggle("AI Optimizer", value=True)
    if run_opt:
        opt_method_ui = st.selectbox("Solver Engine", ["Differential Evolution (Accurate/Slow)", "L-BFGS-B (Fast/Draft)"])
    else:
        opt_method_ui = "None"

# ==========================================
# PAGE 1: MISSION CONTROL
# ==========================================
if page == "🚀 Mission Control":
    st.title("🧬 Gene-to-Plant (G2P) Digital Twin")
    st.markdown("#### Real-world industrial optimization via deep bioinformatics and mechanistic physics.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Payload Input")
        seq_input = st.text_area("Amino Acid Sequence", value="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", height=250)
        st.session_state.seq_input = seq_input
        
        if st.button("🚀 RUN FACTORY PIPELINE", use_container_width=True, type="primary"):
            with st.status("Simulating Industrial Pipeline...", expanded=True) as status:
                st.write("Extracting 14 Biophysical Features...")
                st.write("Applying Host-Specific AI Modifiers...")
                
                # Map the UI choice to the backend string
                if run_opt:
                    backend_method = 'differential_evolution' if 'Accurate' in opt_method_ui else 'L-BFGS-B'
                    st.write(f"Running ODE-Physics Solver with {backend_method}...")
                else:
                    backend_method = 'none'
                    st.write("Running static baseline ODE-Physics Solver...")

                # Pass everything to the backend
                st.session_state.sim_results = twin.run_pipeline(
                    sequence=seq_input, 
                    hours=hours, 
                    volume=volume, 
                    optimize=run_opt, 
                    microbe_name=microbe_choice,
                    opt_method=backend_method  # Injecting the solver choice!
                )
                st.session_state.sim_graphs = twin.plot_results(st.session_state.sim_results)
                status.update(label="Simulation Successful", state="complete")

    with col2:
        st.subheader("Executive KPI Summary")
        if st.session_state.sim_results:
            res = st.session_state.sim_results
            yield_kg = (res['optimized_simulation' if run_opt else 'baseline_simulation']['product'][-1] * res['tank_volume']) / 1000
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Bio-Yield</div>
                <div class="metric-value">{yield_kg:.2f} kg</div>
            </div>
            <div class="metric-card" style="margin-top:10px;">
                <div class="metric-label">Metabolic Burden</div>
                <div class="metric-value" style="color:#DEFF9A;">{res['burden_analysis']['burden_class']}</div>
            </div>
            <div class="metric-card" style="margin-top:10px;">
                <div class="metric-label">Target Organism</div>
                <div class="metric-value" style="color:#A3E635;">{res['microbe']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Input a sequence and run simulation to see factory metrics.")

# ==========================================
# PAGE 2: TELEMETRY (Interactive Buttons)
# ==========================================
elif page == "📈 Factory Telemetry":
    st.title("Telemetry Charts")
    if not st.session_state.sim_graphs:
        st.warning("Please run simulation first.")
    else:
        st.markdown("### Interactive Metric Selection")
        # Multi-row button layout
        c1, c2, c3, c4, c5 = st.columns(5)
        if c1.button("🟢 Biomass"): st.session_state.active_graph = "Biomass"
        if c2.button("🔵 Product"): st.session_state.active_graph = "Product"
        if c3.button("🟡 Substrate"): st.session_state.active_graph = "Substrate"
        if c4.button("💨 Oxygen"): st.session_state.active_graph = "Dissolved Oxygen"
        if c5.button("🔥 Heat"): st.session_state.active_graph = "Heat Generation"
        
        c6, c7, c8, c9, c10 = st.columns(5)
        if c6.button("⚙️ Agitation"): st.session_state.active_graph = "Agitation"
        if c7.button("🌬️ Aeration"): st.session_state.active_graph = "Aeration"
        if c8.button("💧 Feeding"): st.session_state.active_graph = "Feeding"
        if c9.button("🧬 Features"): st.session_state.active_graph = "Features"
        if c10.button("⚖️ Burden"): st.session_state.active_graph = "Burden"
        
        st.divider()
        try:
            st.pyplot(st.session_state.sim_graphs[st.session_state.active_graph])
        except KeyError:
            st.info("⚠️ Optimizer disabled during simulation: Optimization charts are hidden.")

# ==========================================
# PAGE 3: CONTROL TRAJECTORY
# ==========================================
elif page == "🎛️ Control Trajectory":
    st.title("Factory Floor Controls")
    if not st.session_state.sim_results or 'optimization' not in st.session_state.sim_results:
        st.warning("No optimized data available. Please run the simulation with the AI Optimizer enabled.")
    else:
        df = twin.generate_control_report(st.session_state.sim_results)
        st.dataframe(df.style.background_gradient(cmap='Greens', subset=['RPM', 'Product (g/L)']), use_container_width=True, height=600)

# ==========================================
# PAGE 4: ARCHITECTURE (Knowledge Slide)
# ==========================================
elif page == "🏗️ Model Architecture":
    st.title("🏗️ Engineering Blueprints: G2P System")
    
    st.markdown("""
    ### The 4-Pillar Architecture
    Designed by **Shivansh Sahu**, this digital twin integrates four distinct layers of science and computation.
    """)

    colL, colR = st.columns(2)
    with colL:
        st.markdown("""
        <div class="arch-card">
            <h4>1. Bio-AI Layer (Pillar 1)</h4>
            <p>Uses a <b>Triple-Branch Stacking Ensemble</b> (Random Forest, XGBoost, Gradient Boosting) trained on 15,000 lab records.
            Extracts 14 biophysical features including GRAVY, Instability Index, and rare codon bias.</p>
        </div>
        <div class="arch-card">
            <h4>2. Mechanistic Solver (Pillar 2)</h4>
            <p>Converts AI Burden predictions into physical kinetic parameters. Uses Monod-growth equations to link metabolic strain 
            to growth rate (μ) and death rate (kd).</p>
        </div>
        """, unsafe_allow_html=True)
    
    with colR:
        st.markdown("""
        <div class="arch-card">
            <h4>3. Transport Physics (Pillar 3)</h4>
            <p>Simulates industrial mass transfer (kLa) and heat generation. It calculates oxygen supply capacity vs demand
            based on reactor geometry (Diameter, Impeller type).</p>
        </div>
        <div class="arch-card">
            <h4>4. Dynamic Optimizer (Pillar 4)</h4>
            <p>Uses <b>Differential Evolution</b> and <b>L-BFGS-B</b> optimization to find the perfect trajectory for RPM, Airflow, and 
            Feeding rates over 72+ hours.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Core Mechanistic ODE System (Fed-Batch Kinetics)")
    
    st.markdown("The physics engine solves these coupled differential equations over thousands of time steps to simulate the bioreactor trajectory:")
    
    col_math1, col_math2 = st.columns(2)
    
    with col_math1:
        st.markdown("**1. Biomass Growth Rate ($X$)**")
        st.latex(r" \frac{dX}{dt} = \mu \cdot X - k_d \cdot X - \frac{F}{V} \cdot X ")
        
        st.markdown("**2. Product Formation Rate ($P$)**")
        st.latex(r" \frac{dP}{dt} = (\alpha \cdot \mu + \beta) \cdot X - \frac{F}{V} \cdot P ")

    with col_math2:
        st.markdown("**3. Substrate Consumption Rate ($S$)**")
        st.latex(r" \frac{dS}{dt} = -\left( \frac{\mu}{Y_{xs}} + m_s \right) \cdot X + \frac{F}{V} \cdot (S_{feed} - S) ")
        
        st.markdown("**4. Specific Growth Rate ($\mu$)**")
        st.latex(r" \mu = \mu_{max} \left( \frac{S}{K_s + S} \right) \left( \frac{O_2}{K_{o2} + O_2} \right) ")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Engineer Contact: shivanshsahu01234@gmail.com | Model Version: G2P-Production-v1.0")