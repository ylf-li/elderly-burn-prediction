"""
老年人烧伤伤口感染风险预测系统
Streamlit Web应用
=====================================
Elderly Burn Wound Infection Risk Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap
import pickle
import plotly.graph_objects as go

# ================================
# 页面配置
# ================================
st.set_page_config(
    page_title="老年人烧伤伤口感染预测",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# 自定义CSS样式
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
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc66;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 加载模型和相关文件
# ================================
@st.cache_resource
def load_model():
    """加载CatBoost模型"""
    model = CatBoostClassifier()
    model.load_model('catboost_model.cbm')
    return model

@st.cache_resource
def load_feature_names():
    """加载特征名称"""
    with open('feature_names.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_shap_explainer():
    """加载SHAP解释器"""
    with open('shap_explainer.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_ranges():
    """加载特征范围"""
    with open('feature_ranges.pkl', 'rb') as f:
        return pickle.load(f)

# 尝试加载所有资源
try:
    model = load_model()
    feature_names = load_feature_names()
    explainer = load_shap_explainer()
    feature_ranges = load_feature_ranges()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# ================================
# 特征描述和单位（中英文对照）
# ================================
FEATURE_INFO = {
    'age': {
        'name': '年龄分组 (Age Group)',
        'unit': '',
        'type': 'select',
        'options': {1: '60-69岁', 2: '70-79岁', 3: '≥80岁'},
        'description': '患者年龄分组'
    },
    'sex': {
        'name': '性别 (Sex)',
        'unit': '',
        'type': 'select',
        'options': {0: '女性', 1: '男性'},
        'description': '患者性别'
    },
    'TBSA': {
        'name': '烧伤总面积 (TBSA)',
        'unit': '%',
        'type': 'number',
        'description': '总体烧伤面积占体表面积百分比'
    },
    'with Full-thickness burn': {
        'name': '全层烧伤面积',
        'unit': '%',
        'type': 'number',
        'description': '全层烧伤（三度烧伤）面积百分比'
    },
    'with  inhalation injury': {
        'name': '吸入性损伤',
        'unit': '',
        'type': 'select',
        'options': {0: '无', 1: '有'},
        'description': '是否存在吸入性损伤'
    },
    'with shock': {
        'name': '休克',
        'unit': '',
        'type': 'select',
        'options': {0: '无', 1: '有'},
        'description': '是否发生休克'
    },
    'Multimorbidity': {
        'name': '合并症数量',
        'unit': '个',
        'type': 'number',
        'description': '患者合并症的数量'
    },
    'ICU admission': {
        'name': 'ICU入住',
        'unit': '',
        'type': 'select',
        'options': {0: '否', 1: '是'},
        'description': '是否入住ICU'
    },
    'Numbers of Indwelling Tubes': {
        'name': '留置管数量',
        'unit': '个',
        'type': 'number',
        'description': '留置管道的数量'
    },
    'surgery': {
        'name': '手术',
        'unit': '',
        'type': 'select',
        'options': {0: '无', 1: '有'},
        'description': '是否进行手术治疗'
    },
    'Classes of antibiotics ': {
        'name': '抗生素种类',
        'unit': '种',
        'type': 'number',
        'description': '使用的抗生素种类数'
    },
    'LOS': {
        'name': '住院时间 (LOS)',
        'unit': '天',
        'type': 'number',
        'description': '住院天数'
    },
    'Serum Albumin': {
        'name': '血清白蛋白',
        'unit': 'g/L',
        'type': 'number',
        'description': '血清白蛋白水平'
    },
    'BMI': {
        'name': '体重指数 (BMI)',
        'unit': 'kg/m²',
        'type': 'number',
        'description': '体重指数'
    },
    'Comorbid diabetes': {
        'name': '合并糖尿病',
        'unit': '',
        'type': 'select',
        'options': {0: '无', 1: '有'},
        'description': '是否合并糖尿病'
    },
    'Nutritional Support': {
        'name': '营养支持',
        'unit': '',
        'type': 'select',
        'options': {0: '无', 1: '有'},
        'description': '是否接受营养支持治疗'
    },
    'Using advanced wound dressings': {
        'name': '高级敷料',
        'unit': '',
        'type': 'select',
        'options': {0: '否', 1: '是'},
        'description': '是否使用高级伤口敷料'
    }
}

# ================================
# 侧边栏 - 模型信息
# ================================
with st.sidebar:
    st.markdown("## 📋 模型信息")
    st.markdown("**Model Information**")
    
    st.markdown(f"""
    - **模型类型**: CatBoost Classifier
    - **训练数据**: 老年人烧伤临床数据
    - **目标变量**: 伤口感染 (Wound Infection)
    - **特征数量**: {len(feature_names)} 个临床指标
    """)
    
    st.markdown("---")
    st.markdown("## 📊 特征说明")
    st.markdown("**Feature Description**")
    
    for feature in feature_names:
        info = FEATURE_INFO.get(feature, {'name': feature, 'unit': ''})
        if feature in feature_ranges:
            ranges = feature_ranges[feature]
            with st.expander(f"📌 {info['name']}"):
                if info.get('unit'):
                    st.markdown(f"**单位**: {info['unit']}")
                st.markdown(f"**范围**: {ranges['min']:.1f} - {ranges['max']:.1f}")
                st.markdown(f"**均值**: {ranges['mean']:.2f}")
                if 'description' in info:
                    st.markdown(f"**说明**: {info['description']}")

# ================================
# 主页面标题
# ================================
st.markdown('<p class="main-header">🏥 老年人烧伤伤口感染风险预测</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Elderly Burn Wound Infection Risk Prediction System</p>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📝 请输入患者的临床指标")
st.markdown("Please input the patient's clinical indicators:")

# ================================
# 输入表单
# ================================
col1, col2, col3 = st.columns(3)

input_values = {}

# 分配特征到三列
features_list = list(feature_names)
n_features = len(features_list)
features_per_col = (n_features + 2) // 3

for idx, feature in enumerate(features_list):
    info = FEATURE_INFO.get(feature, {'name': feature, 'unit': '', 'type': 'number'})
    ranges = feature_ranges.get(feature, {'min': 0, 'max': 100, 'median': 50, 'mean': 50})
    
    # 决定放在哪一列
    if idx < features_per_col:
        col = col1
    elif idx < features_per_col * 2:
        col = col2
    else:
        col = col3
    
    with col:
        # 构建标签
        if info.get('unit'):
            label = f"{info['name']} ({info['unit']})"
        else:
            label = info['name']
        
        if info['type'] == 'select':
            options = info.get('options', {0: '否', 1: '是'})
            option_keys = list(options.keys())
            option_labels = list(options.values())
            
            # 找到默认值的索引
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
            # 将标签转回数值
            input_values[feature] = option_keys[option_labels.index(selected_label)]
        else:
            # 数值输入
            min_val = float(ranges['min'])
            max_val = float(ranges['max'])
            default_val = float(ranges['median'])
            
            # 根据范围决定步长
            if max_val - min_val > 100:
                step = 1.0
            elif max_val - min_val > 10:
                step = 0.5
            else:
                step = 0.1
            
            input_values[feature] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val * 1.5,  # 允许一定超出范围
                value=default_val,
                step=step,
                key=feature,
                help=info.get('description', '')
            )

# ================================
# 预测按钮
# ================================
st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 预测 Predict", type="primary", use_container_width=True)

# ================================
# 预测结果
# ================================
if predict_button:
    # 准备输入数据
    input_df = pd.DataFrame([input_values])
    
    # 进行预测
    prediction_proba = model.predict_proba(input_df)[0]
    risk_probability = prediction_proba[1] * 100
    
    # ================================
    # 显示预测结果
    # ================================
    st.markdown("---")
    st.markdown("## 📊 预测结果 Prediction Results")
    
    col_result1, col_result2 = st.columns([1, 1])
    
    with col_result1:
        st.markdown("### 伤口感染风险")
        st.markdown("**Risk of Wound Infection**")
        
        # 大字体显示概率
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
            <h1 style="font-size: 4rem; margin: 0; color: #1f4e79;">{risk_probability:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # 风险等级判断
        if risk_probability < 30:
            risk_level = "低风险 (Low Risk)"
            risk_color = "#00cc66"
            risk_emoji = "🟢"
            risk_advice = "感染风险较低，建议常规护理和观察。"
        elif risk_probability < 60:
            risk_level = "中等风险 (Medium Risk)"
            risk_color = "#ffa500"
            risk_emoji = "🟡"
            risk_advice = "存在一定感染风险，建议加强监测和预防措施。"
        else:
            risk_level = "高风险 (High Risk)"
            risk_color = "#ff4b4b"
            risk_emoji = "🔴"
            risk_advice = "感染风险较高，建议采取积极预防和治疗措施。"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: {risk_color}20; 
                    border-left: 5px solid {risk_color}; border-radius: 5px; margin: 1rem 0;">
            <h3 style="color: {risk_color}; margin: 0;">{risk_emoji} {risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **建议**: {risk_advice}")
    
    with col_result2:
        # 创建仪表盘图
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_probability,
            number={'suffix': '%', 'font': {'size': 40}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "感染风险概率<br>Infection Risk", 'font': {'size': 16}},
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
    # SHAP分析
    # ================================
    st.markdown("---")
    st.markdown("## 🔍 模型解释 Model Interpretation")
    
    # 计算SHAP值
    shap_values = explainer.shap_values(input_df)
    
    # 获取基准值
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, np.ndarray):
            base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            base_value = explainer.expected_value
    else:
        base_value = 0
    
    # 处理SHAP值
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif len(shap_values.shape) == 2:
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
    
    # 创建SHAP数据框
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Feature_CN': [FEATURE_INFO.get(f, {'name': f})['name'] for f in feature_names],
        'Value': [input_values[f] for f in feature_names],
        'SHAP_Value': shap_vals
    }).sort_values('SHAP_Value', key=abs, ascending=True)
    
    # 创建水平条形图
    colors = ['#ff4b4b' if x > 0 else '#0068c9' for x in shap_df['SHAP_Value']]
    
    fig_shap = go.Figure()
    
    fig_shap.add_trace(go.Bar(
        y=[f"{row['Feature_CN']}<br>= {row['Value']:.1f}" for _, row in shap_df.iterrows()],
        x=shap_df['SHAP_Value'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.3f}" for x in shap_df['SHAP_Value']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP值: %{x:.4f}<extra></extra>'
    ))
    
    fig_shap.add_vline(x=0, line_width=2, line_dash="solid", line_color="gray")
    
    fig_shap.update_layout(
        title={
            'text': "SHAP特征贡献分析<br><sup>Feature Contribution Analysis</sup>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="SHAP值 (正值增加风险，负值降低风险)",
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
    
    # SHAP解释说明
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 5px; border-left: 5px solid #ff4b4b;">
            <h4 style="color: #ff4b4b; margin: 0;">🔴 正向贡献（增加风险）</h4>
            <p style="margin: 0.5rem 0 0 0;">红色条形表示该特征的当前值会增加伤口感染的风险概率。</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_exp2:
        st.markdown("""
        <div style="background-color: #d4edda; padding: 1rem; border-radius: 5px; border-left: 5px solid #0068c9;">
            <h4 style="color: #0068c9; margin: 0;">🔵 负向贡献（降低风险）</h4>
            <p style="margin: 0.5rem 0 0 0;">蓝色条形表示该特征的当前值会降低伤口感染的风险概率。</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ================================
    # 特征贡献表格
    # ================================
    st.markdown("---")
    st.markdown("### 📋 特征贡献分析表 Feature Contribution Analysis Table")
    
    # 格式化表格数据
    display_df = shap_df.sort_values('SHAP_Value', key=abs, ascending=False).copy()
    display_df['No.'] = range(1, len(display_df) + 1)
    display_df['SHAP_Value'] = display_df['SHAP_Value'].round(4)
    display_df['Contribution'] = display_df['SHAP_Value'].apply(
        lambda x: '↑ 增加风险' if x > 0 else '↓ 降低风险'
    )
    display_df['Abs_SHAP'] = display_df['SHAP_Value'].abs()
    
    # 显示表格
    st.dataframe(
        display_df[['No.', 'Feature_CN', 'Value', 'SHAP_Value', 'Contribution']].rename(columns={
            'No.': '排名',
            'Feature_CN': '特征名称',
            'Value': '输入值',
            'SHAP_Value': 'SHAP值',
            'Contribution': '贡献方向'
        }).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

# ================================
# 使用说明
# ================================
st.markdown("---")
st.markdown("## 📖 使用说明 Instructions")

with st.expander("点击展开使用说明 / Click to expand instructions", expanded=False):
    st.markdown("""
    ### 如何使用本系统
    
    1. **输入临床指标**: 在上方表单中填入患者的各项临床指标
    2. **点击预测按钮**: 点击"预测"按钮获取结果
    3. **查看结果**: 系统将显示伤口感染风险概率和风险等级
    4. **理解SHAP分析**: 
       - **红色条形**: 该特征增加了感染风险
       - **蓝色条形**: 该特征降低了感染风险
       - **条形长度**: 表示该特征对预测结果的影响程度
    
    ### 风险等级说明
    
    | 风险等级 | 概率范围 | 建议 |
    |---------|---------|------|
    | 🟢 低风险 | 0-30% | 常规护理和观察 |
    | 🟡 中等风险 | 30-60% | 加强监测和预防措施 |
    | 🔴 高风险 | 60-100% | 采取积极预防和治疗措施 |
    
    ### 免责声明
    
    本系统仅供临床参考，不能替代医生的专业判断。实际诊疗决策应结合患者具体情况和临床经验。
    """)

# ================================
# 页脚
# ================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>老年人烧伤伤口感染风险预测系统 | Elderly Burn Wound Infection Risk Prediction System</p>
    <p>Powered by CatBoost & SHAP | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
