import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

def train_xgboost(X_train_tab, y_train, X_test_tab, y_test):
    print("--- Training Pure Tabular XGBoost ---")

    # Name the 12 features so the chart makes sense!
    feature_names = [
        "Slope_Mean", "TWI_Mean", "Drainage_Mean", "LULC_Mean", "Rainfall_Mean", "NDVI_Mean",
        "Slope_Std", "TWI_Std", "Drainage_Std", "LULC_Std", "Rainfall_Std", "NDVI_Std"
    ]

    xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42)

    # Train
    xgb_model.fit(X_train_tab, y_train)

    # Evaluate
    preds = xgb_model.predict(X_test_tab)
    print("\n--- Evaluating XGBoost ---")
    print(classification_report(y_test, preds, target_names=['Not Suitable', 'Suitable']))
    print(f"XGBoost Test Accuracy: {accuracy_score(y_test, preds):.4f}")

    # Plot Feature Importance (TAKE A SCREENSHOT OF THIS FOR YOUR REPORT!)
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model, max_num_features=12, importance_type='weight',
                        title='What features matter most for Well Siting?',
                        xlabel='F-Score (Importance)')
    plt.yticks(range(12), [feature_names[i] for i in xgb_model.feature_importances_.argsort()])
    plt.tight_layout()
    plt.show()

# Run it! (Uses the X_t_train tabular features)
train_xgboost(X_t_train, y_train, X_t_test, y_test)