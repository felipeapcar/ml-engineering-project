from xgboost import XGBClassifier

def train_model(X_train, y_train, best_params):
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model