# app/streamlit_ui.py
import sys
from pathlib import Path

# --- Ingest absolute path parameters into system memory before libraries process ---
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: 
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from src.utils import setup_logger

ui_logger = setup_logger("streamlit_client_app")

st.set_page_config(
    page_title="ICU Mortality Risk Decision Support Portal", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PHASE 1: ACTIVE INFRASTRUCTURE CONNECTION HANDSHAKE ---
if 'assets_verified' not in st.session_state:
    with st.spinner("🔄 Handshaking with local production inference nodes..."):
        try:
            # Probe backend documentation route to confirm port availability
            check = requests.get("http://127.0.0.1:8000/docs", timeout=2.0)
            st.session_state['assets_verified'] = True
            st.toast("✅ Inference Server Sync Operational!", icon="🏥")
            ui_logger.info("Successfully established connection handshake with core FastAPI engine.")
        except Exception as e:
            st.session_state['assets_verified'] = False
            ui_logger.error(f"Handshake failed. Backend serving layer port offline: {str(e)}")

st.title("🏥 ICU Mortality Risk Hybrid Decision Support Portal")
st.markdown(
    "Exposes ensembled **LightGBM Platt-Scaled Estimators** combined with a "
    "**PyTorch Recurrent BiLSTM Layer** optimized over 8,000 unified patient profiles."
)
st.write("---")

# If connection handshakes are actively failing, notify the client and pause viewport initialization
if not st.session_state.get('assets_verified', False):
    st.error("### 🔴 System Connection Blocked\nThe local FastAPI inference service running on port `:8000` is currently offline or unreachable. Please launch your backend server using `uvicorn api.main:app --port 8000` to synchronize this client panel dashboard.")
    st.stop()

# --- PHASE 2: SIDEBAR COMPONENT CONTAINER SEPARATIONS WITH INTERNAL CONTEXT FIXES ---
# FIXED: Stripped explicit 'st.sidebar' definitions from the nested widgets to allow successful rendering inside expander scopes
with st.sidebar.expander("👤 Patient Baseline Profile", expanded=True):
    age_input = st.slider("Chronological Age (Years)", min_value=15, max_value=110, value=65)
    gender_label = st.radio("Biological Gender Profile", options=["Female", "Male"])
    gender_encoded = 1 if gender_label == "Male" else 0

with st.sidebar.expander("🫁 Real-Time High-Frequency Vitals", expanded=True):
    st.caption("Observed clinical means compiled over active frames:")
    hr_input = st.number_input("Heart Rate Average (bpm)", min_value=30.0, max_value=220.0, value=88.0)
    gcs_input = st.number_input("Glasgow Coma Scale (GCS Units)", min_value=3.0, max_value=15.0, value=12.0)
    sysbp_input = st.number_input("Systolic Blood Pressure (mmHg)", min_value=40.0, max_value=250.0, value=115.0)
    temp_input = st.number_input("Core Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.2)

with st.sidebar.expander("🧪 Advanced Optional Laboratory Panels", expanded=False):
    st.caption("Toggle specific markers to increment threshold boundaries:")
    bun_active = st.checkbox("Include Blood Urea Nitrogen (BUN)")
    bun_val = st.number_input("BUN Value (mg/dL)", min_value=1.0, max_value=200.0, value=20.0, disabled=not bun_active)
    
    creat_active = st.checkbox("Include Serum Creatinine")
    creat_val = st.number_input("Creatinine Value (mg/dL)", min_value=0.1, max_value=15.0, value=1.0, step=0.1, disabled=not creat_active)
    
    plat_active = st.checkbox("Include Platelet Count")
    plat_val = st.number_input("Platelets (*10^3/µL)", min_value=5.0, max_value=1000.0, value=250.0, disabled=not plat_active)
    
    wbc_active = st.checkbox("Include White Blood Cell Count (WBC)")
    wbc_val = st.number_input("WBC (*10^3/µL)", min_value=0.1, max_value=100.0, value=7.5, step=0.1, disabled=not wbc_active)
    
    gluc_active = st.checkbox("Include Serum Glucose")
    gluc_val = st.number_input("Glucose Value (mg/dL)", min_value=10.0, max_value=800.0, value=120.0, disabled=not gluc_active)
    
    fio2_active = st.checkbox("Include Fractional Inspired Oxygen (FiO2)")
    fio2_val = st.slider("FiO2 Parameter Allocation", min_value=0.21, max_value=1.00, value=0.40, step=0.01, disabled=not fio2_active)

# --- PHASE 3: PACK DATA CONTRACT FOR REST TRANSMISSION ---
compiled_observations = [
    {"Parameter": "HR", "Value": float(hr_input)}, 
    {"Parameter": "GCS", "Value": float(gcs_input)},
    {"Parameter": "SysBP", "Value": float(sysbp_input)}, 
    {"Parameter": "Temp", "Value": float(temp_input)}
]
if bun_active: compiled_observations.append({"Parameter": "BUN", "Value": float(bun_val)})
if creat_active: compiled_observations.append({"Parameter": "Creatinine", "Value": float(creat_val)})
if plat_active: compiled_observations.append({"Parameter": "Platelets", "Value": float(plat_val)})
if wbc_active: compiled_observations.append({"Parameter": "WBC", "Value": float(wbc_val)})
if gluc_active: compiled_observations.append({"Parameter": "Glucose", "Value": float(gluc_val)})
if fio2_active: compiled_observations.append({"Parameter": "FiO2", "Value": float(fio2_val)})

api_payload = {
    "Age": float(age_input), 
    "Gender": int(gender_encoded), 
    "Observations": compiled_observations
}

# --- PHASE 4: VIEWPORT GRID ROW LAYOUT ALLOCATIONS ---
left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    st.subheader("📊 Calibrated Clinical Risk Assessment")
    if st.button("🚀 Run Analytical Core Evaluation Pass", use_container_width=True):
        ui_logger.info(f"Inference run triggered. Requesting calculation for packet size: {len(compiled_observations)}")
        
        with st.spinner("Processing network matrix calculations..."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=api_payload)
                if response.status_code == 200:
                    result = response.json()
                    prob = result["Mortality_Risk_Probability"]
                    flag = result["Clinical_Status_Flag"]
                    drivers = result["Primary_Risk_Drivers"]
                    
                    st.write(" ")
                    met_col, prog_col = st.columns([1, 2])
                    with met_col:
                        st.metric(
                            label="Calibrated Mortality Score", 
                            value=f"{prob * 100:.1f} %", 
                            delta=flag, 
                            delta_color="normal" if prob < 0.35 else "inverse"
                        )
                    with prog_col:
                        st.write(" ")
                        st.write(" ")
                        st.progress(float(prob))
                        st.caption(f"Risk Envelope Threshold Boundary (Classification Assignment: {flag})")
                    
                    st.write("---")
                    st.subheader("🔬 Local AI Explainability Analysis (TreeSHAP)")
                    st.caption("Quantifies the exact mathematical push-and-pull contributions of patient feature drivers.")
                    
                    features_list, impacts_list, color_list = [], [], []
                    
                    for item in drivers["escalating"]:
                        features_list.append(item["feature"])
                        impacts_list.append(item["impact"])
                        color_list.append("#ef4444") 
                        
                    for item in drivers["mitigating"]:
                        features_list.append(item["feature"])
                        impacts_list.append(item["impact"])
                        color_list.append("#10b981") 
                        
                    if features_list:
                        # Render Plotly Horizontal Bar Engine context
                        fig = go.Figure(go.Bar(
                            x=impacts_list[::-1],
                            y=features_list[::-1],
                            orientation='h',
                            marker_color=color_list[::-1],
                            text=[f"{v:+.3f}" for v in impacts_list[::-1]],
                            textposition='outside',
                            hovertemplate="<b>Driver:</b> %{y}<br><b>SHAP Weight Contribution:</b> %{x:+.4f}<extra></extra>"
                        ))
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Relative Game-Theoretic Attribution Metric Score",
                            height=380,
                            margin=dict(l=10, r=45, t=10, b=10),
                            xaxis=dict(showgrid=True, gridcolor='#f3f4f6', zeroline=True, zerolinecolor='#4b5563')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("The calculated attribution matrix baseline is currently flat.")
                else:
                    st.error(f"Inference server rejected request packet. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"Handshake Fault: Failed to complete pipeline execution loop: {str(e)}")

with right_col:
    st.subheader("📋 Active Dynamic Observation Registry")
    st.caption("Monitors and captures variable objects passed through the backend Pydantic validator gate:")
    
    registry_log = [
        {"Clinical Parameter ID": "Patient Admittance Age", "Current Input Value": f"{age_input} Years"},
        {"Clinical Parameter ID": "Biological Gender Token", "Current Input Value": gender_label},
        {"Clinical Parameter ID": "Heart Rate Tracking (HR)", "Current Input Value": f"{hr_input} bpm"},
        {"Clinical Parameter ID": "Glasgow Coma Index Scale (GCS)", "Current Input Value": f"{gcs_input} Points"},
        {"Clinical Parameter ID": "Systolic Vascular Pressure (SysBP)", "Current Input Value": f"{sysbp_input} mmHg"},
        {"Clinical Parameter ID": "Core Body Temperature Profile", "Current Input Value": f"{temp_input} °C"}
    ]
    if bun_active: registry_log.append({"Clinical Parameter ID": "Blood Urea Nitrogen (BUN)", "Current Input Value": f"{bun_val} mg/dL"})
    if creat_active: registry_log.append({"Clinical Parameter ID": "Serum Creatinine Levels", "Current Input Value": f"{creat_val} mg/dL"})
    if plat_active: registry_log.append({"Clinical Parameter ID": "Platelets Concentration", "Current Input Value": f"{plat_val} *10^3/µL"})
    if wbc_active: registry_log.append({"Clinical Parameter ID": "White Blood Cell Count (WBC)", "Current Input Value": f"{wbc_val} *10^3/µL"})
    if gluc_active: registry_log.append({"Clinical Parameter ID": "Serum Glucose Baseline", "Current Input Value": f"{gluc_val} mg/dL"})
    if fio2_active: registry_log.append({"Clinical Parameter ID": "Inspired Oxygen Fraction (FiO2)", "Current Input Value": f"{fio2_val * 100:.0f}% O2"})
    
    st.dataframe(pd.DataFrame(registry_log), use_container_width=True, hide_index=True)