#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ================= Page Configuration =================
st.set_page_config(page_title="CN Film Predictor", layout="wide")

# ================= CSS Styling (Enhanced Layout) =================
st.markdown("""
<style>
    /* Hide default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global font */
    .stApp { font-family: "Segoe UI", Arial, sans-serif; }
    
    /* Bold labels for inputs */
    .stNumberInput label, .stSelectbox label { font-weight: 600; color: #2c3e50; }
    
    /* Center alignment for middle column */
    .middle-column { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; }
    
    /* Predict button styling */
    .stButton > button {
        background-color: #2c3e50; 
        color: white; 
        border: none; 
        padding: 15px 40px; 
        font-size: 18px; 
        font-weight: bold; 
        border-radius: 8px; 
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #34495e; 
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Arrow icon styling */
    .arrow-icon { font-size: 60px; color: #7f8c8d; margin: 10px 0; text-align: center; }
    
    /* Right result box styling */
    .result-container {
        background-color: #f8f9fa;
        border: 2px solid #2c3e50;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .result-label { font-size: 18px; color: #7f8c8d; margin-bottom: 10px; }
    .result-value { font-size: 48px; font-weight: bold; color: #2c3e50; }
    .result-unit { font-size: 16px; color: #95a5a6; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 CN Thin Film N/C Ratio Prediction System")
st.markdown("---")

# ================= Load Model (with Error Handling) =================
@st.cache_resource
def load_model():
    possible_files = ["xgboost_standalone.pkl", "xgboost_clean.pkl", "model.pkl", "best_model.pkl"]
    model_path = None
    
    for f in possible_files:
        if os.path.exists(f):
            model_path = f
            break
    
    if not model_path:
        return None, "Model file (.pkl) not found"
    
    try:
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
        
        # Handle dictionary-wrapped models
        if isinstance(obj, dict):
            if 'model' in obj:
                return obj['model'], f"Loaded (extracted from dict): {model_path}"
            elif 'xgb' in obj:
                return obj['xgb'], f"Loaded (extracted from dict): {model_path}"
            else:
                return list(obj.values())[0], f"Loaded (inferred): {model_path}"
        else:
            return obj, f"Loaded: {model_path}"
            
    except Exception as e:
        return None, f"Load error: {str(e)}"

model, status_msg = load_model()

# ================= Main Layout (3 Columns) =================
# Column ratios: Input : Action : Output = 4 : 1 : 3
col_input, col_action, col_output = st.columns([4, 1, 3])

# --- Left: Input Section ---
with col_input:
    st.subheader("📝 Experimental Parameters")
    st.markdown("Enter thin film deposition conditions:")
    
    c1, c2 = st.columns(2)
    with c1:
        power = st.number_input("Power (W)", min_value=0.0, value=100.0, step=1.0)
        c_content = st.number_input("C Content (at%)", min_value=0.0, value=14.12, step=0.01)
        n_content = st.number_input("N Content (at%)", min_value=0.0, value=31.92, step=0.01)
        h_content = st.number_input("H Content (at%)", min_value=0.0, value=52.29, step=0.01)
        de_val = st.number_input("DE of Carbon (eV)", min_value=0.0, value=4.55, step=0.01)
    
    with c2:
        input_nc = st.number_input("Input N/C Ratio", min_value=0.0, value=3.52, step=0.01)
        pressure = st.number_input("Pressure (Pa)", min_value=0.0, value=91.19, step=0.01)
        time_min = st.number_input("Time (min)", min_value=0.0, value=100.0, step=1.0)
        
        gas_map = {"NH3": 0, "N2": 1, "Ar": 2, "Mixed": 3}
        gas_type = st.selectbox("Gas Type", options=list(gas_map.keys()), index=1)
        gas_encoded = gas_map[gas_type]

# --- Middle: Arrow & Button ---
with col_action:
    st.markdown("<div class='middle-column'>", unsafe_allow_html=True)
    st.markdown("<div class='arrow-icon'>➡️</div>", unsafe_allow_html=True)
    predict_clicked = st.button("PREDICT")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Right: Output Section ---
with col_output:
    st.subheader("📊 Prediction Result")
    
    result_placeholder = st.empty()
    
    if 'prediction_result' not in st.session_state:
        result_placeholder.markdown("""
        <div class='result-container'>
            <div class='result-label'>Awaiting Prediction...</div>
            <div style='font-size: 60px; color: #bdc3c7;'>❓</div>
            <div class='result-unit'>Enter parameters on the left and click Predict</div>
        </div>
        """, unsafe_allow_html=True)

# ================= Prediction Logic =================
if predict_clicked:
    if model is None:
        result_placeholder.error(f"❌ Model loading failed:\n{status_msg}")
    else:
        try:
            input_array = np.array([[
                power, c_content, n_content, h_content, de_val,
                input_nc, pressure, time_min, gas_encoded
            ]])
            
            pred_value = model.predict(input_array)[0]
            
            result_placeholder.markdown(f"""
            <div class='result-container'>
                <div class='result-label'>Predicted N/C Ratio</div>
                <div class='result-value'>{pred_value:.4f}</div>
                <div class='result-unit'>Based on XGBoost Model</div>
                <div style='margin-top:15px; font-size:12px; color:#27ae60; font-weight:bold;'>✅ Prediction Successful</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state['prediction_result'] = pred_value
            
        except Exception as e:
            result_placeholder.error(f"❌ Prediction error:\n{str(e)}")
            st.exception(e)

# Footer Status Bar
st.divider()
st.caption(f"System Status: {'🟢 Model Ready' if model else '🔴 Model Missing'} | {status_msg}")

