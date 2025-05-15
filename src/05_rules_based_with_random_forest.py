import joblib
import numpy as np
import pandas as pd


# Load trained model and scaler
model = joblib.load('random_forest_anomaly_model.pkl')
scaler = joblib.load('scaler.pkl')

def hybrid_anomaly_check(new_tx: dict, history_df: pd.DataFrame) -> dict:
    """
    Hybrid anomaly detection using threshold and Random Forest.
    
    Returns:
        {
            "is_anomaly": 0 or 1,
            "source": "threshold" or "random_forest",
            "violated_threshold": [rules],
            "rf_probability": float,
            "rf_prediction": 0 or 1
        }
    """
    results = {
        "is_anomaly": 0,
        "source": None,
        "violated_threshold": [],
        "rf_prediction": 0,
        "rf_probability": 0.0
    }
    # print(model.feature_names_in_)

    # 1. Extract info
    merchant_id = new_tx['Sub-Merchant ID']
    amount = new_tx['Transaction Amount']
    date = new_tx['Transaction Date']

    month = date // 100

    # Compute daily total
    daily_total = history_df.loc[
        (history_df['Sub-Merchant ID'] == merchant_id) &
        (history_df['Transaction Date'] == date),
        'Transaction Amount'
    ].sum() + amount

    # Compute monthly total
    history_df['Transaction_Month'] = history_df['Transaction Date'] // 100
    monthly_total = history_df.loc[
        (history_df['Sub-Merchant ID'] == merchant_id) &
        (history_df['Transaction_Month'] == month),
        'Transaction Amount'
    ].sum() + amount

    if daily_total > 500_000:
        results["violated_threshold"].append("DailyLimit")
    if monthly_total > 3_000_000:
        results["violated_threshold"].append("MonthlyLimit")

    if results["violated_threshold"]:
        results["is_anomaly"] = 1
        results["source"] = "threshold"
        return results  # short-circuit: no need for model

    # 2. Prepare features for Random Forest
    from pandas import DataFrame
    tx_df = DataFrame([{
        'Log_Transaction_Amount': np.log1p(np.abs(new_tx['Transaction Amount'])),
        'Distance': new_tx['Distance'],
        'Customer Age': new_tx['Customer Age'],
        'Transaction_Hour': new_tx['Transaction Time'] // 10000,
        'Customer_Encoded': new_tx['Customer_Encoded'],
        'Merchant_Encoded': new_tx['Merchant_Encoded'],
        'Sub-Merchant ID': new_tx['Sub-Merchant ID'],


    }])
    print(tx_df)
    # Standardize numeric columns
    cols_to_scale = ['Log_Transaction_Amount', 'Distance', 'Customer Age', 'Transaction_Hour', 'Customer_Encoded', 'Merchant_Encoded', 'Sub-Merchant ID']
    tx_df[cols_to_scale] = scaler.transform(tx_df[cols_to_scale])
    print(tx_df)
    # Predict with RF
    prob = model.predict_proba(tx_df)[0][1]
    pred = model.predict(tx_df)[0]

    results["rf_prediction"] = pred
    results["rf_probability"] = prob

    if pred == 1:
        results["is_anomaly"] = 1
        results["source"] = "random_forest"

    return results


new_tx = {
    'Transaction Amount': 120000,
    'Distance': 1,
    'Customer Age': 5,
    'Transaction Time': 113015,  
    'Transaction Date': 20250101,
    'Sub-Merchant ID': 50,
    'Merchant_Encoded':35,
    'Customer_Encoded':35
}
df_history = pd.read_csv('../dataset/cleaned_BizPlay_2025_JAN.csv')
result = hybrid_anomaly_check(new_tx, df_history)
print("🚨 Anomaly?", result["is_anomaly"])
print("🔍 Source:", result["source"])
print("❌ Thresholds Violated:", result["violated_threshold"])
print("🧠 RF Prediction:", result["rf_prediction"])
print("🔢 RF Probability:", result["rf_probability"])
