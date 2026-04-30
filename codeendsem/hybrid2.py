# HYBRID 2 -> GEORESNET + XGBOOST
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

print("--- Training Hybrid: GeoResNet + XGBoost ---")

# 1. Extract the deep spatial features using your already-trained ResNet model
# (Assuming your 'model' from the HybridWideAndDeep class is still in memory)
print("Extracting deep embeddings for Train and Test sets...")
train_embeddings = model.extract_embeddings(X_p_train)
test_embeddings = model.extract_embeddings(X_p_test)

# 2. Fuse the Deep features (512) with the Tabular features (12)
hybrid_X_train = np.hstack([train_embeddings, X_t_train])
hybrid_X_test = np.hstack([test_embeddings, X_t_test])

print(f"Fused Feature Matrix Shape: {hybrid_X_train.shape}")

# 3. Train the XGBoost Classifier on the Fused Features
hybrid_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
hybrid_xgb.fit(hybrid_X_train, y_train)

# 4. Evaluate
preds = hybrid_xgb.predict(hybrid_X_test)
print("\n--- Evaluation on Test Set ---")
print(classification_report(y_test, preds, target_names=['Not Suitable', 'Suitable']))
print(f"Hybrid (ResNet + XGBoost) Accuracy: {accuracy_score(y_test, preds):.4f}")