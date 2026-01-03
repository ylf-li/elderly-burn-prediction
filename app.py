"""
Elderly Burn Wound Infection Risk Prediction System
Streamlit Web Application
=====================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="Elderly Burn Wound Infection Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Custom CSS Styles
# ================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# Load Model and Related Files
# ================================
@st.cache_resource
def load_model():
    """Load the Stacking ensemble model"""
    return joblib.load('stacking_model.pkl')

@st.cache_resource
def load_feature_names():
    """Load feature names"""
    return joblib.load('feature_names.pkl')

@st.cache_resource
def load_feature_ranges():
    """Load feature ranges"""
    return joblib.load('feature_ranges.pkl')

@st.cache_resource
def load_base_models():
    """Load base models for SHAP explanation"""
    return joblib.load('base_models.pkl')

# Try to load all resources
try:
    model = load_model()
    feature_names = load_feature_names()
    feature_ranges = load_feature_ranges()
    base_models = load_base_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# ================================
# Feature Descriptions
# ================================
FEATURE_INFO = {
    'age': {
        'name': 'Age Group',
        'unit': '',
        'type': 'select',
        'options': {1: '60-69 years', 2: '70-79 years', 3: '≥80 years'},
        'description': 'Patient age group'
    },
    'sex': {
        'name': 'Sex',
        'unit': '',
        'type': 'select',
        'options': {0: 'Female', 1: 'Male'},
        'description': 'Patient sex'
    },
    'TBSA': {
        'name': 'Total Burn Surface Area (TBSA)',
        'unit': '%',
        'type': 'number',
        'description': 'Percentage of total body surface area burned'
    },
    'with Full-thickness burn': {
        'name': 'Full-thickness Burn Area',
        'unit': '%',
        'type': 'number',
        'description': 'Percentage of full-thickness (third-degree) burn'
    },
    'with  inhalation injury': {
        'name': 'Inhalation Injury',
        'unit': '',
        'type': 'select',
        'options': {0: 'No', 1: 'Yes'},
        'description': 'Presence of inhalation injury'
    },
    'with shock': {
        'name': 'Shock',
        'unit': '',
        'type': 'select',
        'options': {0: 'No', 1: 'Yes'},
        'description': 'Presence of shock'
    },
    'Multimorbidity': {
        'name': 'Number of Comorbidities',
        'unit': '',
        'type': 'number',
        'description': 'Number of comorbid conditions'
    },
    'ICU admission': {
        'name': 'ICU Admission',
        'unit': '',
        'type': 'select',
        'options': {0: 'No', 1: 'Yes'},
        'description': 'Whether admitted to ICU'
    },
    'Numbers of Indwelling Tubes': {
        'name': 'Number of Indwelling Tubes',
        'unit': '',
        'type': 'number',
        'description': 'Number of indwelling tubes'
    },
    'surgery': {
        'name': 'Surgery',
        'unit': '',
        'type': 'select',
        'options': {0: 'No', 1: 'Yes'},
        'description': 'Whether surgery was performed'
    },
    'Classes of antibiotics ': {
        'name': 'Classes of Antibiotics',
        'unit': '',
        'type': 'number',
        'description': 'Number of antibiotic classes used'
    },
    'LOS': {
        'name': 'Length of Stay (LOS)',
        'unit': 'days',
        'type': 'number',
        'description': 'Hospital length of stay in days'
    },
    'Serum Albumin': {
        'name': 'Serum Albumin',
        'unit': 'g/L',
        'type': 'number',
        'description': 'Serum albumin level'
    },
    'BMI': {
        'name': 'Body Mass Index (BMI)',
        'unit': 'kg/m²',
        'type': 'number',
        'description': 'Body mass index'
    },
    'Comorbid diabetes': {
        'name': 'Comorbid Diabetes',
        'unit': '',
        'type': 'select',
        'options': {0: 'No', 1: 'Yes'},
        'description': 'Presence of diabetes'
    },
    'Nutritional Support': {
        'name': 'Nutritional Support',
        'unit': '',
        'type': 'select',
        'options': {0: 'No', 1: 'Yes'},
        'description': 'Whether nutritional support was provided'
    },
    'Using advanced wound dressings': {
        'name': 'Advanced Wound Dressings',
        'unit': '',
        'type': 'select',
        'options': {0: 'No', 1: 'Yes'},
        'description': 'Whether advanced wound dressings were used'
    }
}

# ================================
# Sidebar - Model Information
# ================================
with st.sidebar:
    st.markdown("## 📋 Model Information")
    
    st.markdown(f"""
    - **Model Type**: Stacking Ensemble
    - **Base Models**: 
      - Logistic Regression
      - Random Forest
      - XGBoost
      - LightGBM
      - CatBoost
    - **Meta Learner**: Logistic Regression
    - **Target Variable**: Wound Infection
    - **Number of Features**: {len(feature_names)}
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Feature Description")
    
    for feature in feature_names:
        info = FEATURE_INFO.get(feature, {'name': feature, 'unit': ''})
        if feature in feature_ranges:
            ranges = feature_ranges[feature]
            with st.expander(f"📌 {info['name']}"):
                if info.get('unit'):
                    st.markdown(f"**Unit**: {info['unit']}")
                st.markdown(f"**Range**: {ranges['min']:.1f} - {ranges['max']:.1f}")
                st.markdown(f"**Mean**: {ranges['mean']:.2f}")
                if 'description' in info:
                    st.markdown(f"**Description**: {info['description']}")

# ================================
# Main Page Title
# ================================
st.markdown('<p class="main-header">🏥 Elderly Burn Wound Infection Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Based on Stacking Ensemble Machine Learning Model</p>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📝 Please Enter Patient Clinical Indicators")

# ================================
# Input Form
# ================================
col1, col2, col3 = st.columns(3)

input_values = {}

# Distribute features to three columns
features_list = list(feature_names)
n_features = len(features_list)
features_per_col = (n_features + 2) // 3

for idx, feature in enumerate(features_list):
    info = FEATURE_INFO.get(feature, {'name': feature, 'unit': '', 'type': 'number'})
    ranges = feature_ranges.get(feature, {'min': 0, 'max': 100, 'median': 50, 'mean': 50})
    
    # Determine which column
    if idx < features_per_col:
        col = col1
    elif idx < features_per_col * 2:
        col = col2
    else:
        col = col3
    
    with col:
        # Build label
        if info.get('unit'):
            label = f"{info['name']} ({info['unit']})"
        else:
            label = info['name']
        
        if info['type'] == 'select':
            options = info.get('options', {0: 'No', 1: 'Yes'})
            option_keys = list(options.keys())
            option_labels = list(options.values())
            
            # Find default value index
            default_val = int(ranges.get('median', 0))
            if default_val in option_keys:
                default_idx = option_keys.index(default_val)
            else:
                default_idx = 0
            
            selected_label = st.selectbox(
                label,
                options=option_labels,
                index=default_idx,
                key=feature,
                help=info.get('description', '')
            )
            # Convert label back to numeric value
            input_values[feature] = option_keys[option_labels.index(selected_label)]
        else:
            # Numeric input
            min_val = float(ranges['min'])
            max_val = float(ranges['max'])
            default_val = float(ranges['median'])
            
            # Determine step based on range
            if max_val - min_val > 100:
                step = 1.0
            elif max_val - min_val > 10:
                step = 0.5
            else:
                step = 0.1
            
            input_values[feature] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val * 1.5,
                value=default_val,
                step=step,
                key=feature,
                help=info.get('description', '')
            )

# ================================
# Predict Button
# ================================
st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

# ================================
# Prediction Results
# ================================
if predict_button:
    # Prepare input data
    input_df = pd.DataFrame([input_values])
    
    # Make prediction
    prediction_proba = model.predict_proba(input_df)[0]
    risk_probability = prediction_proba[1] * 100
    
    # ================================
    # Display Prediction Results
    # ================================
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    col_result1, col_result2 = st.columns([1, 1])
    
    with col_result1:
        st.markdown("### Risk of Wound Infection")
        
        # Display probability in large font
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
            <h1 style="font-size: 4rem; margin: 0; color: #1f4e79;">{risk_probability:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk level determination
        if risk_probability < 30:
            risk_level = "Low Risk"
            risk_color = "#00cc66"
            risk_emoji = "🟢"
            risk_advice = "Low infection risk. Routine care and monitoring recommended."
        elif risk_probability < 60:
            risk_level = "Medium Risk"
            risk_color = "#ffa500"
            risk_emoji = "🟡"
            risk_advice = "Moderate infection risk. Enhanced monitoring and preventive measures recommended."
        else:
            risk_level = "High Risk"
            risk_color = "#ff4b4b"
            risk_emoji = "🔴"
            risk_advice = "High infection risk. Aggressive prevention and treatment measures recommended."
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: {risk_color}20; 
                    border-left: 5px solid {risk_color}; border-radius: 5px; margin: 1rem 0;">
            <h3 style="color: {risk_color}; margin: 0;">{risk_emoji} {risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **Recommendation**: {risk_advice}")
    
    with col_result2:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_probability,
            number={'suffix': '%', 'font': {'size': 40}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Infection Risk Probability", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#d4edda'},
                    {'range': [30, 60], 'color': '#fff3cd'},
                    {'range': [60, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_probability
                }
            }
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "darkblue", 'family': "Arial"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ================================
    # SHAP Analysis
    # ================================
    st.markdown("---")
    st.markdown("## 🔍 Model Interpretation")
    
    # Use Random Forest for SHAP explanation (more stable)
    try:
        rf_model = base_models['Random Forest']
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_df)
        
        # Handle SHAP values format
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            shap_vals = shap_values[0, :, 1]
        else:
            shap_vals = shap_values[0]
        
        # Create SHAP dataframe
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Feature_Name': [FEATURE_INFO.get(f, {'name': f})['name'] for f in feature_names],
            'Value': [input_values[f] for f in feature_names],
            'SHAP_Value': shap_vals
        }).sort_values('SHAP_Value', key=abs, ascending=True)
        
        # Create horizontal bar chart
        colors = ['#ff4b4b' if x > 0 else '#0068c9' for x in shap_df['SHAP_Value']]
        
        fig_shap = go.Figure()
        
        fig_shap.add_trace(go.Bar(
            y=[f"{row['Feature_Name']}<br>= {row['Value']:.1f}" for _, row in shap_df.iterrows()],
            x=shap_df['SHAP_Value'],
            orientation='h',
            marker_color=colors,
            text=[f"{x:.3f}" for x in shap_df['SHAP_Value']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
        ))
        
        fig_shap.add_vline(x=0, line_width=2, line_dash="solid", line_color="gray")
        
        fig_shap.update_layout(
            title={
                'text': "SHAP Feature Contribution Analysis",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="SHAP Value (Positive = Increases Risk, Negative = Decreases Risk)",
            yaxis_title="",
            height=500,
            showlegend=False,
            margin=dict(l=200, r=50, t=80, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig_shap.update_xaxes(gridcolor='lightgray', zerolinecolor='gray')
        fig_shap.update_yaxes(gridcolor='lightgray')
        
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # SHAP explanation legend
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("""
            <div style="background-color: #fff3cd; padding: 1rem; border-radius: 5px; border-left: 5px solid #ff4b4b;">
                <h4 style="color: #ff4b4b; margin: 0;">🔴 Positive Contribution (Increases Risk)</h4>
                <p style="margin: 0.5rem 0 0 0;">Red bars indicate that the current value of this feature increases the probability of wound infection.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_exp2:
            st.markdown("""
            <div style="background-color: #d4edda; padding: 1rem; border-radius: 5px; border-left: 5px solid #0068c9;">
                <h4 style="color: #0068c9; margin: 0;">🔵 Negative Contribution (Decreases Risk)</h4>
                <p style="margin: 0.5rem 0 0 0;">Blue bars indicate that the current value of this feature decreases the probability of wound infection.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ================================
        # Feature Contribution Table
        # ================================
        st.markdown("---")
        st.markdown("### 📋 Feature Contribution Analysis Table")
        
        # Format table data
        display_df = shap_df.sort_values('SHAP_Value', key=abs, ascending=False).copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df['SHAP_Value'] = display_df['SHAP_Value'].round(4)
        display_df['Contribution'] = display_df['SHAP_Value'].apply(
            lambda x: '↑ Increases Risk' if x > 0 else '↓ Decreases Risk'
        )
        
        # Display table
        st.dataframe(
            display_df[['Rank', 'Feature_Name', 'Value', 'SHAP_Value', 'Contribution']].rename(columns={
                'Rank': 'Rank',
                'Feature_Name': 'Feature',
                'Value': 'Input Value',
                'SHAP_Value': 'SHAP Value',
                'Contribution': 'Direction'
            }).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.warning(f"SHAP analysis unavailable: {str(e)}")

# ================================
# Instructions
# ================================
st.markdown("---")
st.markdown("## 📖 Instructions")

with st.expander("Click to expand instructions", expanded=False):
    st.markdown("""
    ### How to Use This System
    
    1. **Enter Clinical Indicators**: Fill in the patient's clinical indicators in the form above
    2. **Click Predict Button**: Click the "Predict" button to get results
    3. **View Results**: The system will display wound infection risk probability and risk level
    4. **Understand SHAP Analysis**: 
       - **Red bars**: This feature increases infection risk
       - **Blue bars**: This feature decreases infection risk
       - **Bar length**: Indicates the magnitude of the feature's impact on prediction
    
    ### Risk Level Description
    
    | Risk Level | Probability Range | Recommendation |
    |------------|-------------------|----------------|
    | 🟢 Low Risk | 0-30% | Routine care and monitoring |
    | 🟡 Medium Risk | 30-60% | Enhanced monitoring and preventive measures |
    | 🔴 High Risk | 60-100% | Aggressive prevention and treatment |
    
    ### Disclaimer
    
    This system is for clinical reference only and cannot replace professional medical judgment. 
    Actual treatment decisions should consider patient-specific circumstances and clinical experience.
    """)

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>Elderly Burn Wound Infection Risk Prediction System</p>
    <p>Powered by Stacking Ensemble Learning & SHAP | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)