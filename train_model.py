import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import argparse

# 1. Load data
def load_data(path):
    df = pd.read_csv(path)
    # Features: typing_speed, pause_time, error_rate, sentiment_score
    X = df[['typing_speed', 'pause_time', 'error_rate', 'sentiment_score']]
    y = df['strain_label']  # 0 = normal, 1 = high strain
    return X, y

# 2. Train model
def train(path, out_model):
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print('Accuracy:', model.score(X_test, y_test))
    joblib.dump(model, out_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='strain_model.pkl')
    args = parser.parse_args()
    train(args.data, args.out)
