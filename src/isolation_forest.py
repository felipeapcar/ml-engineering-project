from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
import sys

def train_isolation_forest(train_data, contamination=0.01, random_state=42):
    df = pd.read_csv(train_data)
    y = df['Class']

    df.drop(columns=['Class'], inplace=True)

    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    df['outlier_iso'] = (iso_forest.fit_predict(df) == -1).astype(int)
    df['Class'] = y

    joblib.dump(iso_forest, 'iso.pkl')

    return df
