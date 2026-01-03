"""
Elderly Burn Wound Infection Risk Prediction System
Using CatBoost Model
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
    page_title="Elderly Burn Wound Infection Prediction",
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

# Feature Info
FEATURE_INFO = {
    'age': {'name': 'Age Group', 'type': 'select', 'options': {1: '60-69 yrs', 2: '70-79 yrs', 3: '≥80 yrs'}},
    'sex': {'name': 'Sex', 'type': 'select', 'options': {0: 'Female', 1: 'Male'}},
    'TBSA': {'name': 'TBSA (%)', 'type': 'number'},
    'with Full-thickness burn': {'name': 'Full-thickness Burn (%)', 'type': 'number'},
    'with  inhalation injury': {'name': 'Inhalation Injury', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'with shock': {'name': 'Shock', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'Multimorbidity': {'name': 'Comorbidities', 'type': 'number'},
    'ICU admission': {'name': 'ICU Admission', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'Numbers of Indwelling Tubes': {'name': 'Indwelling Tubes', 'type': 'number'},
    'surgery': {'name': 'Surgery', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'Classes of antibiotics ': {'name': 'Antibiotic Classes', 'type': 'number'},
    'LOS': {'name': 'Length of Stay (days)', 'type': 'number'},
    'Serum Albumin': {'name': 'Serum Albumin (g/L)', 'type': 'number'},
    'BMI': {'name': 'BMI (kg/m²)', 'type': 'number'},
    'Comorbid diabetes': {'name': 'Diabetes', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'Nutritional Support': {'name': 'Nutritional Support', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}},
    'Using advanced wound dressings': {'name': 'Advanced Dressings', 'type': 'select', 'options': {0: 'No', 1: 'Yes'}}
}

# Sidebar
with st.sidebar:
    st.markdown("## Model Information")
    st.markdown(f"""
    - **Model**: CatBoost Classifier
    - **Target**: Wound Infection
    - **Features**: {len(feature_names)}
    """)
    st.markdown("---")
    st.markdown("## Feature Ranges")
    for f in feature_names:
        info = FEATURE_INFO.get(f, {'name': f})
        if f in feature_ranges:
            r = feature_ranges[f]
            st.markdown(f"**{info['name']}**: {r['min']:.1f} - {r['max']:.1f}")

# Main Title
st.title("🏥 Elderly Burn Wound Infection Risk Prediction")
st.markdown("---")

# Input Form
st.subheader("📝 Enter Patient Information")
col1, col2, col3 = st.columns(3)

input_values = {}
cols = [col1, col2, col3]

for idx, feature in enumerate(feature_names):
    info = FEATURE_INFO.get(feature, {'name': feature, 'type': 'number'})
    ranges = feature_ranges.get(feature, {'min': 0, 'max': 100, 'median': 50})
    col = cols[idx % 3]
    
    with col:
        if info['type'] == 'select':
            options = info.get('options', {0: 'No', 1: 'Yes'})
            keys = list(options.keys())
            labels = list(options.values())
            default_idx = 0
            selected = st.selectbox(info['name'], labels, index=default_idx, key=feature)
            input_values[feature] = keys[labels.index(selected)]
        else:
            input_values[feature] = st.number_input(
                info['name'],
                min_value=float(ranges['min']),
                max_value=float(ranges['max']) * 1.5,
                value=float(ranges['median']),
                key=feature
            )

# Predict Button
st.markdown("---")
if st.button("🔮 Predict", type="primary", use_container_width=True):
    
    input_df = pd.DataFrame([input_values])
    
    # Prediction
    pred_proba = model.predict_proba(input_df)[0]
    risk = pred_proba[1] * 100
    
    # Results
    st.markdown("## 📊 Prediction Results")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown(f"""
        <div style="text-align:center; padding:2rem; background:#f0f2f6; border-radius:10px;">
            <h1 style="font-size:4rem; color:#1f4e79;">{risk:.1f}%</h1>
            <p>Wound Infection Risk</p>
        </div>
        """, unsafe_allow_html=True)
        
        if risk < 30:
            level, color = "🟢 Low Risk", "#00cc66"
        elif risk < 60:
            level, color = "🟡 Medium Risk", "#ffa500"
        else:
            level, color = "🔴 High Risk", "#ff4b4b"
        
        st.markdown(f"<h3 style='text-align:center; color:{color};'>{level}</h3>", unsafe_allow_html=True)
    
    with col_r2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={'suffix': '%'},
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
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Analysis
    st.markdown("---")
    st.markdown("## 🔍 Feature Contribution (SHAP)")
    
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
            y=[f"{r['Feature']} = {r['Value']:.1f}" for _, r in shap_df.iterrows()],
            x=shap_df['SHAP'],
            orientation='h',
            marker_color=colors
        ))
        fig_shap.add_vline(x=0, line_color="gray")
        fig_shap.update_layout(
            title="SHAP Values (Red = Increases Risk, Blue = Decreases Risk)",
            height=450,
            xaxis_title="SHAP Value"
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Table
        st.markdown("### Feature Contribution Table")
        table_df = shap_df.sort_values('SHAP', key=abs, ascending=False).copy()
        table_df['Direction'] = table_df['SHAP'].apply(lambda x: '↑ Risk' if x > 0 else '↓ Risk')
        table_df['SHAP'] = table_df['SHAP'].round(4)
        st.dataframe(table_df[['Feature', 'Value', 'SHAP', 'Direction']], use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.warning(f"SHAP analysis unavailable: {e}")

# Footer
st.markdown("---")
st.markdown("<center>Elderly Burn Wound Infection Prediction | Powered by CatBoost & SHAP</center>", unsafe_allow_html=True)