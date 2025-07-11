from fastapi import FastAPI, Request
import joblib
import utils

app = FastAPI()
model = joblib.load('strain_model.pkl')

@app.post('/collect/typing')
async def collect_typing(data: Request):
    records = await data.json()
    utils.save_typing(records)
    return {'status': 'ok'}

@app.post('/collect/text')
async def collect_text(data: Request):
    payload = await data.json()
    utils.save_text(payload['text'])
    return {'status': 'ok'}

@app.get('/predict')
def predict():
    features = utils.compute_features()
    score = model.predict_proba([features])[0][1]
    strain = score > 0.5
    return {'strain_score': score, 'alert': bool(strain)}
