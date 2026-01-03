# 🏥 老年人烧伤伤口感染风险预测系统

Elderly Burn Wound Infection Risk Prediction System

基于CatBoost机器学习模型的老年人烧伤伤口感染预测Web应用。

## ✨ 功能特点

- 🔮 **智能预测**: 输入临床指标，实时预测伤口感染风险
- 📊 **可解释性分析**: 基于SHAP的模型解释，了解各因素对预测的贡献
- 📈 **直观可视化**: 仪表盘、条形图等多种可视化展示
- 🌐 **Web应用**: 便捷的在线访问，无需安装

## 🖥️ 在线演示

[点击访问应用](https://your-app-name.streamlit.app/)

## 📋 模型信息

| 项目 | 说明 |
|------|------|
| 模型类型 | CatBoost Classifier |
| 目标变量 | 伤口感染 (Wound Infection) |
| 特征数量 | 17个临床指标 |
| 训练数据 | 老年人烧伤临床数据 |

## 📊 输入特征

| 特征 | 说明 | 单位/取值 |
|------|------|----------|
| 年龄分组 | Age Group | 1=60-69岁, 2=70-79岁, 3=≥80岁 |
| 性别 | Sex | 0=女, 1=男 |
| TBSA | 烧伤总面积 | % |
| 全层烧伤面积 | Full-thickness burn | % |
| 吸入性损伤 | Inhalation injury | 0=无, 1=有 |
| 休克 | Shock | 0=无, 1=有 |
| 合并症数量 | Multimorbidity | 个 |
| ICU入住 | ICU admission | 0=否, 1=是 |
| 留置管数量 | Indwelling Tubes | 个 |
| 手术 | Surgery | 0=无, 1=有 |
| 抗生素种类 | Classes of antibiotics | 种 |
| 住院时间 | LOS | 天 |
| 血清白蛋白 | Serum Albumin | g/L |
| BMI | 体重指数 | kg/m² |
| 合并糖尿病 | Comorbid diabetes | 0=无, 1=有 |
| 营养支持 | Nutritional Support | 0=无, 1=有 |
| 高级敷料 | Advanced wound dressings | 0=否, 1=是 |

## 🚀 本地运行

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/elderly-burn-prediction.git
cd elderly-burn-prediction
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行应用

```bash
streamlit run app.py
```

### 4. 访问应用

打开浏览器访问 `http://localhost:8501`

## 📁 文件结构

```
elderly-burn-prediction/
├── app.py                  # Streamlit应用主文件
├── train_model.py          # 模型训练脚本
├── catboost_model.cbm      # 训练好的CatBoost模型
├── feature_names.pkl       # 特征名称
├── shap_explainer.pkl      # SHAP解释器
├── feature_ranges.pkl      # 特征范围信息
├── requirements.txt        # Python依赖
├── data.csv               # 原始数据（可选）
└── README.md              # 项目说明
```

## 🔧 重新训练模型

如果需要使用新数据重新训练模型：

1. 将数据文件命名为 `data.csv` 放在项目目录下
2. 运行训练脚本：
   ```bash
   python train_model.py
   ```
3. 脚本将生成新的模型文件和相关配置

## 📝 引用

如果您在研究中使用了本系统，请引用：

```
[您的论文引用格式]
```

## ⚠️ 免责声明

本系统仅供临床参考和学术研究使用，不能替代医生的专业判断。实际诊疗决策应结合患者具体情况和临床经验。

## 📄 许可证

MIT License

## 👥 联系方式

如有问题或建议，请提交 Issue 或联系 [您的邮箱]
