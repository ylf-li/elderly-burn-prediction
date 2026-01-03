"""
Elderly Burn Wound Infection Prediction Model Training Script
Using Ensemble Learning (Stacking)
=====================================
This script will generate:
1. ensemble_model.cbm - Trained ensemble model
2. feature_names.pkl - Feature names list
3. shap_explainer.pkl - SHAP explainer object
4. feature_ranges.pkl - Feature ranges information
5. model_performance.txt - Model performance metrics
6. base_models_performance.txt - Base models performance comparison

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
import pickle
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ================================
# 1. Data Loading and Preprocessing
# ================================
print("=" * 80)
print("Step 1: Load Data")
print("=" * 80)

# Read data
df = pd.read_csv('data.csv')
print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")

# View target variable distribution
print(f"\nTarget variable 'Wound Infection' distribution:")
print(df['Wound Infection'].value_counts())
print(f"Infection rate: {df['Wound Infection'].mean():.2%}")

# ================================
# 2. Feature Selection
# ================================
print("\n" + "=" * 80)
print("Step 2: Feature Selection")
print("=" * 80)

# Define feature columns
selected_features = [
    'age',
    'sex',
    'TBSA',
    'with Full-thickness burn',
    'with  inhalation injury',
    'with shock',
    'Multimorbidity',
    'ICU admission',
    'Numbers of Indwelling Tubes',
    'surgery',
    'Classes of antibiotics ',
    'LOS',
    'Serum Albumin',
    'BMI',
    'Comorbid diabetes',
    'Nutritional Support',
    'Using advanced wound dressings'
]

# Check if features exist
available_features = []
for f in selected_features:
    if f in df.columns:
        available_features.append(f)
    else:
        print(f"Warning: Feature '{f}' does not exist in dataset")

print(f"\nSuccessfully selected features: {len(available_features)}")
print("\nFeature list:")
for i, f in enumerate(available_features, 1):
    print(f"  {i:2d}. {f}")

# Prepare feature matrix and target variable
X = df[available_features].copy()
y = df['Wound Infection'].copy()

# Handle missing values
missing_before = X.isnull().sum().sum()
if missing_before > 0:
    print(f"\nFound {missing_before} missing values, filling with median...")
    X = X.fillna(X.median())

print(f"\nFeature matrix size: {X.shape}")
print(f"Target variable size: {y.shape}")

# ================================
# 3. Train-Test Split
# ================================
print("\n" + "=" * 80)
print("Step 3: Split Training and Test Sets")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X):.1%})")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X):.1%})")
print(f"Training set infection rate: {y_train.mean():.2%}")
print(f"Test set infection rate: {y_test.mean():.2%}")

# ================================
# 4. Train Base Models
# ================================
print("\n" + "=" * 80)
print("Step 4: Train Base Models")
print("=" * 80)

# Define base models
base_models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
}

# Store base model predictions and performance
base_predictions_train = {}
base_predictions_test = {}
base_model_performance = {}

print("\nTraining base models...\n")

for name, model in base_models.items():
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    pred_train = model.predict_proba(X_train)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]
    pred_test_class = model.predict(X_test)
    
    # Store predictions
    base_predictions_train[name] = pred_train
    base_predictions_test[name] = pred_test
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, pred_test_class)
    auc = roc_auc_score(y_test, pred_test)
    
    base_model_performance[name] = {
        'accuracy': accuracy,
        'auc': auc,
        'model': model
    }
    
    print(f"  ✓ {name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# Display base models comparison
print("\n" + "=" * 80)
print("Base Models Performance Comparison")
print("=" * 80)
print(f"{'Model':<25} {'Accuracy':>12} {'AUC':>12}")
print("-" * 80)
for name, perf in base_model_performance.items():
    print(f"{name:<25} {perf['accuracy']:>12.4f} {perf['auc']:>12.4f}")
print("-" * 80)

# ================================
# 5. Create Meta-Features for Stacking
# ================================
print("\n" + "=" * 80)
print("Step 5: Create Meta-Features for Stacking")
print("=" * 80)

# Create meta-features from base model predictions
X_train_meta = pd.DataFrame(base_predictions_train)
X_test_meta = pd.DataFrame(base_predictions_test)

print(f"Meta-features training set size: {X_train_meta.shape}")
print(f"Meta-features test set size: {X_test_meta.shape}")
print("\nMeta-features (first 5 rows):")
print(X_train_meta.head())

# ================================
# 6. Train Meta-Learner (Ensemble Model)
# ================================
print("\n" + "=" * 80)
print("Step 6: Train Meta-Learner (Ensemble Model)")
print("=" * 80)

# Use CatBoost as meta-learner
meta_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

print("\nTraining ensemble model (meta-learner)...\n")

# Train meta-learner
meta_model.fit(
    X_train_meta, y_train,
    eval_set=(X_test_meta, y_test),
    use_best_model=True
)

print(f"\n✓ Ensemble model training completed!")
print(f"Best iteration: {meta_model.best_iteration_}")

# ================================
# 7. Evaluate Ensemble Model
# ================================
print("\n" + "=" * 80)
print("Step 7: Evaluate Ensemble Model")
print("=" * 80)

# Make predictions
y_pred = meta_model.predict(X_test_meta)
y_pred_proba = meta_model.predict_proba(X_test_meta)[:, 1]

# Calculate metrics
ensemble_accuracy = accuracy_score(y_test, y_pred)
ensemble_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'='*60}")
print(f"ENSEMBLE MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy: {ensemble_accuracy:.4f}")
print(f"AUC: {ensemble_auc:.4f}")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['No Infection (0)', 'Infection (1)']))

print("="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted No    Predicted Yes")
print(f"Actual No       {cm[0,0]:5d}           {cm[0,1]:5d}")
print(f"Actual Yes      {cm[1,0]:5d}           {cm[1,1]:5d}")

# ================================
# 8. Compare All Models
# ================================
print("\n" + "=" * 80)
print("Step 8: Final Model Comparison")
print("=" * 80)

print(f"\n{'Model':<25} {'Accuracy':>12} {'AUC':>12} {'Improvement':>15}")
print("-" * 80)

# Display base models
for name, perf in base_model_performance.items():
    print(f"{name:<25} {perf['accuracy']:>12.4f} {perf['auc']:>12.4f} {'-':>15}")

# Display ensemble model
best_base_auc = max([perf['auc'] for perf in base_model_performance.values()])
improvement = ((ensemble_auc - best_base_auc) / best_base_auc) * 100

print("-" * 80)
print(f"{'ENSEMBLE (Stacking)':<25} {ensemble_accuracy:>12.4f} {ensemble_auc:>12.4f} {f'+{improvement:.2f}%':>15}")
print("=" * 80)

# ================================
# 9. Save Performance Reports
# ================================
print("\n" + "=" * 80)
print("Step 9: Save Performance Reports")
print("=" * 80)

# Save ensemble model performance
with open('model_performance.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Elderly Burn Wound Infection Prediction Model Performance Report\n")
    f.write("Ensemble Learning (Stacking) Approach\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Dataset Information:\n")
    f.write(f"  - Total samples: {df.shape[0]} records\n")
    f.write(f"  - Number of features: {len(available_features)}\n")
    f.write(f"  - Infection samples: {y.sum()} ({y.mean():.2%})\n")
    f.write(f"  - Non-infection samples: {len(y) - y.sum()} ({1-y.mean():.2%})\n\n")
    
    f.write(f"Training Parameters:\n")
    f.write(f"  - Training set size: {X_train.shape[0]}\n")
    f.write(f"  - Test set size: {X_test.shape[0]}\n")
    f.write(f"  - Best iteration: {meta_model.best_iteration_}\n\n")
    
    f.write(f"Ensemble Model Performance:\n")
    f.write(f"  - Accuracy: {ensemble_accuracy:.4f}\n")
    f.write(f"  - AUC: {ensemble_auc:.4f}\n\n")
    
    f.write(f"Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['No Infection (0)', 'Infection (1)']))
    
    f.write(f"\nConfusion Matrix:\n")
    f.write(f"              Predicted No    Predicted Yes\n")
    f.write(f"Actual No       {cm[0,0]:5d}           {cm[0,1]:5d}\n")
    f.write(f"Actual Yes      {cm[1,0]:5d}           {cm[1,1]:5d}\n")

print("✓ Performance report saved to: model_performance.txt")

# Save base models comparison
with open('base_models_performance.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Base Models Performance Comparison\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"{'Model':<25} {'Accuracy':>12} {'AUC':>12}\n")
    f.write("-" * 80 + "\n")
    
    for name, perf in base_model_performance.items():
        f.write(f"{name:<25} {perf['accuracy']:>12.4f} {perf['auc']:>12.4f}\n")
    
    f.write("-" * 80 + "\n")
    f.write(f"{'ENSEMBLE (Stacking)':<25} {ensemble_accuracy:>12.4f} {ensemble_auc:>12.4f}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Performance Improvement:\n")
    f.write(f"  - Best base model AUC: {best_base_auc:.4f}\n")
    f.write(f"  - Ensemble model AUC: {ensemble_auc:.4f}\n")
    f.write(f"  - Improvement: +{improvement:.2f}%\n")

print("✓ Base models comparison saved to: base_models_performance.txt")

# ================================
# 10. Feature Importance Analysis
# ================================
print("\n" + "=" * 80)
print("Step 10: Feature Importance Analysis")
print("=" * 80)

# Get feature importance from meta-learner
feature_importance = meta_model.get_feature_importance()
importance_df = pd.DataFrame({
    'Base_Model': list(base_models.keys()),
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nBase Model Importance in Ensemble:")
print("-" * 60)
print(f"{'Rank':<6} {'Base Model':<25} {'Importance':>12}")
print("-" * 60)
for rank, (idx, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{rank:<6} {row['Base_Model']:<25} {row['Importance']:>12.4f}")
print("-" * 60)

# Save feature importance
importance_df.to_csv('ensemble_feature_importance.csv', index=False)
print("\n✓ Feature importance saved to: ensemble_feature_importance.csv")

# ================================
# 11. SHAP Analysis for Ensemble Model
# ================================
print("\n" + "=" * 80)
print("Step 11: Create SHAP Explainer for Ensemble Model")
print("=" * 80)

print("\nCreating SHAP explainer...")
explainer = shap.TreeExplainer(meta_model)
print("✓ SHAP explainer created successfully!")

print("\nCalculating SHAP values (this may take some time)...")
shap_values_test = explainer.shap_values(X_test_meta)
print(f"✓ SHAP values calculated, shape: {shap_values_test.shape}")

# Calculate global SHAP importance
shap_importance = np.abs(shap_values_test).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Base_Model': list(base_models.keys()),
    'SHAP_Importance': shap_importance
}).sort_values('SHAP_Importance', ascending=False)

print("\nSHAP Importance Ranking:")
print("-" * 60)
print(f"{'Rank':<6} {'Base Model':<25} {'SHAP Value':>12}")
print("-" * 60)
for rank, (idx, row) in enumerate(shap_importance_df.iterrows(), 1):
    print(f"{rank:<6} {row['Base_Model']:<25} {row['SHAP_Importance']:>12.4f}")
print("-" * 60)

# Save SHAP importance
shap_importance_df.to_csv('ensemble_shap_importance.csv', index=False)
print("\n✓ SHAP importance saved to: ensemble_shap_importance.csv")

# ================================
# 12. Save Models and Files
# ================================
print("\n" + "=" * 80)
print("Step 12: Save Models and Files")
print("=" * 80)

# 12.1 Save ensemble model
meta_model.save_model('ensemble_model.cbm')
print("✓ Ensemble model saved: ensemble_model.cbm")

# 12.2 Save base models
with open('base_models.pkl', 'wb') as f:
    pickle.dump(base_model_performance, f)
print("✓ Base models saved: base_models.pkl")

# 12.3 Save feature names (use base model names as features for meta-learner)
meta_feature_names = list(base_models.keys())
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(meta_feature_names, f)
print("✓ Feature names saved: feature_names.pkl")

# 12.4 Save SHAP explainer
with open('shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)
print("✓ SHAP explainer saved: shap_explainer.pkl")

# 12.5 Save feature ranges (for base predictions)
feature_ranges = {}
for feature_name in meta_feature_names:
    col_data = X_test_meta[feature_name]
    feature_ranges[feature_name] = {
        'min': float(col_data.min()),
        'max': float(col_data.max()),
        'mean': float(col_data.mean()),
        'median': float(col_data.median()),
        'std': float(col_data.std())
    }

with open('feature_ranges.pkl', 'wb') as f:
    pickle.dump(feature_ranges, f)
print("✓ Feature ranges saved: feature_ranges.pkl")

# 12.6 Save original feature information (for Streamlit app)
original_feature_ranges = {}
for feature in available_features:
    original_feature_ranges[feature] = {
        'min': float(X[feature].min()),
        'max': float(X[feature].max()),
        'mean': float(X[feature].mean()),
        'median': float(X[feature].median()),
        'std': float(X[feature].std())
    }

with open('original_feature_ranges.pkl', 'wb') as f:
    pickle.dump(original_feature_ranges, f)
print("✓ Original feature ranges saved: original_feature_ranges.pkl")

# 12.7 Save original feature names
with open('original_feature_names.pkl', 'wb') as f:
    pickle.dump(available_features, f)
print("✓ Original feature names saved: original_feature_names.pkl")

# ================================
# 13. Plot ROC Curves
# ================================
print("\n" + "=" * 80)
print("Step 13: Generate ROC Curves")
print("=" * 80)

plt.figure(figsize=(10, 8))

# Plot ROC curves for base models
for name, perf in base_model_performance.items():
    model = perf['model']
    y_pred_proba_base = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_base)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {perf['auc']:.3f})", linewidth=2)

# Plot ROC curve for ensemble model
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr_ensemble, tpr_ensemble, 
         label=f"Ensemble (AUC = {ensemble_auc:.3f})", 
         linewidth=3, color='red', linestyle='--')

# Plot diagonal line
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison\nBase Models vs Ensemble Model', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved to: roc_curves_comparison.png")
plt.close()

# ================================
# 14. Completion Summary
# ================================
print("\n" + "=" * 80)
print("TRAINING COMPLETED!")
print("=" * 80)

print("\nGenerated files:")
print("-" * 80)
print("  Core files (required for deployment):")
print("    1. ensemble_model.cbm              - Ensemble model file")
print("    2. base_models.pkl                 - Base models")
print("    3. feature_names.pkl               - Meta-feature names")
print("    4. original_feature_names.pkl      - Original feature names")
print("    5. shap_explainer.pkl              - SHAP explainer")
print("    6. feature_ranges.pkl              - Meta-feature ranges")
print("    7. original_feature_ranges.pkl     - Original feature ranges")
print("")
print("  Analysis reports (optional):")
print("    8. model_performance.txt           - Model performance report")
print("    9. base_models_performance.txt     - Base models comparison")
print("   10. ensemble_feature_importance.csv - Feature importance")
print("   11. ensemble_shap_importance.csv    - SHAP importance")
print("   12. roc_curves_comparison.png       - ROC curves visualization")
print("-" * 80)

print("\nModel Performance Summary:")
print(f"  Best Base Model AUC: {best_base_auc:.4f}")
print(f"  Ensemble Model AUC:  {ensemble_auc:.4f}")
print(f"  Improvement:         +{improvement:.2f}%")

print("\nNext steps:")
print("  1. Upload all core files to GitHub repository")
print("  2. Ensure app.py and requirements.txt are uploaded")
print("  3. Deploy application on Streamlit Cloud")
print("\nGood luck!")

