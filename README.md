# Fraud prediction demo

This small demo serves a vanilla HTML/CSS/JS front-end and uses `testing.py` (a Flask app) as the backend.

Quick start (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
python testing.py
# open http://127.0.0.1:5000 in your browser
```

Notes:
- `testing.py` expects a trained Keras model and some pickle encoding files at locations configured inside the file. If you get errors when starting, check the model path and `txn_type2id`/`merchant2id` pickle paths inside `testing.py`.
- The front-end form posts JSON to the `/predict` endpoint and shows the JSON response.
 - The front-end fetches the last 10 transactions from `/history` and shows a chart of amounts (requires Chart.js CDN).
 - When the model flags a transaction as fraud (predicted_class == 1), the server computes a SHAP explanation and returns the top contributors as `shap`; the front-end shows these contributors under the Result area.
 - Additionally, the model computes per-feature deviation for the numeric features (abs((value-mean)/std)), using feature stats saved at training time to `feature_stats.pkl`; the server returns top deviated features as `feature_deviations`.
