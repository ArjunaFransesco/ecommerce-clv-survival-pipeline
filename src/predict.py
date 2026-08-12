import joblib
import pandas as pd
import os

def predict_clv_and_churn(customer_data_dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clv_model = joblib.load(os.path.join(base_dir, "models", "clv_gradient_boosting.joblib"))
    churn_model = joblib.load(os.path.join(base_dir, "models", "churn_risk_classifier.joblib"))
    df = pd.DataFrame([customer_data_dict])
    clv_pred = clv_model.predict(df)[0]
    churn_prob = churn_model.predict_proba(df)[0][1]
    return {"predicted_1yr_clv": round(float(clv_pred), 2), "churn_probability": round(float(churn_prob), 4)}
