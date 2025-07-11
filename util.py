import pandas as pd

# Append data to CSV

def save_typing(records):
    df = pd.DataFrame(records)
    df['typing_speed'] = df['time'].diff().dt.seconds.fillna(0)
    df.to_csv('../collected_data.csv', mode='a', header=False, index=False)


def save_text(text):
    # Compute sentiment
    from transformers import pipeline
    sentiment = pipeline('sentiment-analysis')
    res = sentiment(text)[0]
    df = pd.DataFrame([{ 'text': text, 'sentiment_score': res['score'] }])
    df.to_csv('../collected_data.csv', mode='a', header=False, index=False)


def compute_features():
    df = pd.read_csv('../collected_data.csv')
    return [
        df['typing_speed'].mean(),
        df['pause_time'].mean(),
        df['error_rate'].mean(),
        df['sentiment_score'].mean(),
    ]
