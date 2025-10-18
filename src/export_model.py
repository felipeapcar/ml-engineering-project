import joblib

def save_model(model, path='fraud_model.pkl'):
    joblib.dump(model, path)