"""
Elderly Burn Wound Infection Risk Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
import shap
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Burn Wound Infection Prediction",
    page_icon="🏥",
    layout="wide"
)

# Load Model and Files
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('catboost_model.cbm')
    return model

@st.cache_resource
def load_feature_names():
    with open('feature_names.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_ranges():
    with open('feature_ranges.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    feature_names = load_feature_names()
    feature_ranges = load_feature_ranges()
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# Feature Display Names and Input Types
FEATURE_INFO = {
    'Classes of antibiotics ': {'name': 'Classes of Antibiotics', 'type': 'number', 'unit': ''},
    'BMI': {'name': 'BMI', 'type': 'number', 'unit': 'kg/m²'},
    'Serum Albumin': {'name': 'Serum Albumin', 'type': 'number', 'unit': 'g/L'},
    'surgery': {'name': 'Surgery', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'TBSA': {'name': 'TBSA', 'type': 'number', 'unit': '%'},
    'with Full-thickness burn': {'name': 'Full-thickness Burn', 'type': 'number', 'unit': '%'},
    'Numbers of Indwelling Tubes': {'name': 'Indwelling Tubes', 'type': 'number', 'unit': ''},
    'sex': {'name': 'Sex', 'type': 'select', 'options': {0: 'Female', 1: 'Male'}},
    'Lower extremity burn': {'name': 'Lower Extremity Burn', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'Season of Admission': {'name': 'Season of Admission', 'type': 'select', 'options': {1: 'Spring', 10: 'Summer', 100: 'Autumn', 1000: 'Winter'}},
    'Tracheostomy tube': {'name': 'Tracheostomy Tube', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'CVC': {'name': 'CVC', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'with  inhalation injury': {'name': 'Inhalation Injury', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'Multimorbidity': {'name': 'Multimorbidity', 'type': 'number', 'unit': ''},
    'EN': {'name': 'Enteral Nutrition', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'age': {'name': 'Age Group', 'type': 'select', 'options': {1: '60-69', 2: '70-79', 3: '≥80'}},
    'Using advanced wound dressings': {'name': 'Advanced Wound Dressings', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
}

# Sidebar - Feature Description Only
with st.sidebar:
    st.markdown("## 📊 Feature Description")
    for f in feature_names:
        info = FEATURE_INFO.get(f, {'name': f})
        if f in feature_ranges:
            r = feature_ranges[f]
            with st.expander(info['name']):
                st.write(f"Range: {r['min']:.1f} - {r['max']:.1f}")
                st.write(f"Mean: {r['mean']:.2f}")

# Main Title
st.title("🏥 Elderly Burn Wound Infection Risk Prediction")
st.markdown("Please input children's clinical indicators:")
st.markdown("---")

# Input Form
col1, col2, col3 = st.columns(3)
input_values = {}
cols = [col1, col2, col3]

for idx, feature in enumerate(feature_names):
    info = FEATURE_INFO.get(feature, {'name': feature, 'type': 'number', 'unit': ''})
    ranges = feature_ranges.get(feature, {'min': 0, 'max': 100, 'median': 50})
    col = cols[idx % 3]
    
    with col:
        label = info['name']
        if info.get('unit'):
            label += f" ({info['unit']})"
        
        if info['type'] == 'select':
            options = info.get('options', {0: 'No', 1: 'Yes'})
            keys = list(options.keys())
            labels = list(options.values())
            # Find default index
            med = int(ranges['median'])
            default_idx = keys.index(med) if med in keys else 0
            selected = st.selectbox(label, labels, index=default_idx, key=feature)
            input_values[feature] = keys[labels.index(selected)]
        else:
            input_values[feature] = st.number_input(
                label,
                min_value=float(ranges['min']),
                max_value=float(ranges['max']) * 1.5,
                value=float(ranges['median']),
                key=feature
            )

# Predict Button
st.markdown("---")
if st.button("Predict", type="primary", use_container_width=True):
    
    input_df = pd.DataFrame([input_values])
    
    # Prediction
    pred_proba = model.predict_proba(input_df)[0]
    risk = pred_proba[1] * 100
    
    # Results
    st.markdown("---")
    st.markdown("## Prediction Results")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("### Risk of Wound Infection")
        st.markdown(f"""
        <div style="text-align:center; padding:2rem; background:#f0f2f6; border-radius:10px; margin:1rem 0;">
            <h1 style="font-size:4rem; color:#1f4e79; margin:0;">{risk:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
        if risk < 30:
            level, color, advice = "Low Risk", "#00cc66", "Low infection risk. Routine care recommended."
        elif risk < 60:
            level, color, advice = "Medium Risk", "#ffa500", "Moderate risk. Enhanced monitoring recommended."
        else:
            level, color, advice = "High Risk", "#ff4b4b", "High risk. Aggressive prevention recommended."
        
        st.markdown(f"""
        <div style="text-align:center; padding:1rem; background:{color}20; border-left:5px solid {color}; border-radius:5px;">
            <h3 style="color:{color}; margin:0;">{level}</h3>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"💡 {advice}")
    
    with col_r2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={'suffix': '%', 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': '#d4edda'},
                    {'range': [30, 60], 'color': '#fff3cd'},
                    {'range': [60, 100], 'color': '#f8d7da'}
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Analysis
    st.markdown("---")
    st.markdown("## Model Interpretation")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
        
        shap_df = pd.DataFrame({
            'Feature': [FEATURE_INFO.get(f, {'name': f})['name'] for f in feature_names],
            'Value': [input_values[f] for f in feature_names],
            'SHAP': sv
        }).sort_values('SHAP', key=abs, ascending=True)
        
        colors = ['#ff4b4b' if x > 0 else '#0068c9' for x in shap_df['SHAP']]
        
        fig_shap = go.Figure(go.Bar(
            y=[f"{r['Feature']} = {r['Value']}" for _, r in shap_df.iterrows()],
            x=shap_df['SHAP'],
            orientation='h',
            marker_color=colors,
            text=[f"{x:.3f}" for x in shap_df['SHAP']],
            textposition='outside'
        ))
        fig_shap.add_vline(x=0, line_color="gray")
        fig_shap.update_layout(
            title="Feature Contribution Analysis",
            height=450,
            xaxis_title="SHAP Value (Positive = Increases Risk, Negative = Decreases Risk)",
            margin=dict(l=200)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Legend
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("""
            <div style="background:#fff3cd; padding:1rem; border-radius:5px; border-left:5px solid #ff4b4b;">
                <b style="color:#ff4b4b;">🔴 Positive Contribution</b><br>
                Increases infection risk
            </div>
            """, unsafe_allow_html=True)
        with col_e2:
            st.markdown("""
            <div style="background:#d4edda; padding:1rem; border-radius:5px; border-left:5px solid #0068c9;">
                <b style="color:#0068c9;">🔵 Negative Contribution</b><br>
                Decreases infection risk
            </div>
            """, unsafe_allow_html=True)
        
        # Table
        st.markdown("### Feature Contribution Analysis")
        table_df = shap_df.sort_values('SHAP', key=abs, ascending=False).copy()
        table_df['No.'] = range(1, len(table_df)+1)
        table_df['Direction'] = table_df['SHAP'].apply(lambda x: '↑ Increases Risk' if x > 0 else '↓ Decreases Risk')
        table_df['SHAP'] = table_df['SHAP'].round(4)
        st.dataframe(
            table_df[['No.', 'Feature', 'Value', 'SHAP', 'Direction']],
            use_container_width=True, 
            hide_index=True
        )
        
    except Exception as e:
        st.warning(f"SHAP analysis unavailable: {e}")

# Instructions
st.markdown("---")
with st.expander("Instructions"):
    st.markdown("""
    **How to Use:**
    1. Enter the patient's clinical indicators in the form above
    2. Click "Predict" button
    3. View the infection risk probability and risk level
    4. Review the SHAP analysis to understand which factors contribute to the prediction
    
    **Risk Levels:**
    - 🟢 Low Risk (0-30%): Routine care and monitoring
    - 🟡 Medium Risk (30-60%): Enhanced monitoring and preventive measures
    - 🔴 High Risk (60-100%): Aggressive prevention and treatment
    
    **Disclaimer:** This system is for clinical reference only.
    """)