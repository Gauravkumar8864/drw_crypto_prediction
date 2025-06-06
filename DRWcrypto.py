import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from scipy.stats import pearsonr
import lightgbm as lgb

# 1. Load parquet files correctly
train_df = pd.read_parquet("D:\\2nd Sem\\drw-crypto-market-prediction\\train.parquet")
test_df = pd.read_parquet("D:\\2nd Sem\\drw-crypto-market-prediction\\test.parquet")
sample_submission = pd.read_csv("D:\\2nd Sem\\drw-crypto-market-prediction\\sample_submission.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)

# 2. Basic EDA & Visualizations
plt.figure(figsize=(8,5))
plt.hist(train_df['prediction'], bins=40, color='skyblue', edgecolor='black')
plt.title('Target Distribution')
plt.xlabel('Prediction')
plt.ylabel('Count')
plt.show()

# Correlation heatmap (on features + target)
plt.figure(figsize=(12,10))
corr = train_df.drop('ID', axis=1).corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# 3. Feature engineering: create rolling stats for features (example for first 5 features)
features = [col for col in train_df.columns if col not in ['ID', 'prediction']]
train_df = train_df.sort_values('ID')  # assuming ID is time-ordered; if timestamp exists, use it

for f in features[:5]:
    train_df[f + '_rolling_mean'] = train_df[f].rolling(window=5, min_periods=1).mean()
    train_df[f + '_rolling_std'] = train_df[f].rolling(window=5, min_periods=1).std().fillna(0)

# Apply same feature engineering to test
test_df = test_df.sort_values('ID')
for f in features[:5]:
    test_df[f + '_rolling_mean'] = test_df[f].rolling(window=5, min_periods=1).mean()
    test_df[f + '_rolling_std'] = test_df[f].rolling(window=5, min_periods=1).std().fillna(0)

# 4. Prepare data for modeling
X = train_df.drop(['ID', 'prediction'], axis=1)
y = train_df['prediction']
X_test = test_df.drop(['ID'], axis=1)

# 5. Time-based train-validation split (80%-20%)
split_index = int(len(train_df) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 7. Train LightGBM model
lgb_train = lgb.Dataset(X_train_scaled, y_train)
lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'seed': 42
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    early_stopping_rounds=50,
    verbose_eval=100
)

# 8. Predict on validation
y_val_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)

# 9. Validation performance
pearson_corr, _ = pearsonr(y_val, y_val_pred)
print(f'Validation Pearson Correlation: {pearson_corr:.4f}')

# Plot actual vs predicted on validation
plt.figure(figsize=(8,6))
plt.scatter(y_val, y_val_pred, alpha=0.5, color='blue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted on Validation Set')
plt.show()

# 10. Feature importance plot
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance()
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=importance_df.head(20), x='importance', y='feature', palette='viridis')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.show()

# 11. Predict on test set
test_preds = model.predict(X_test_scaled, num_iteration=model.best_iteration)

# 12. Prepare submission
submission = sample_submission.copy()
submission['prediction'] = test_preds
# submission.to_csv("D:\\2nd Sem\\drw-crypto-market-prediction\\submission_lgb.csv", index=False)
# print("Submission file 'submission_lgb.csv' created.")
output_path = "D:\\2nd Sem\\drw-crypto-market-prediction\\submission_lgb.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file saved to: {output_path}")
