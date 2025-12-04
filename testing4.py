# testing.py (updated)
from flask import Flask, request, jsonify, send_file
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import traceback
import shap
import os
import hashlib
print("Starting application...")
app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.expanduser("~"), "Desktop", "fraud_rnn_keras9.keras")
model = tf.keras.models.load_model(model_path)

# Load encoders
with open(r"C:\Users\Nishok\Desktop\txn_type2id.pkl", "rb") as f:
    txn_type2id = pickle.load(f)
with open(r"C:\Users\Nishok\Desktop\merchant2id.pkl", "rb") as f:
    merchant2id = pickle.load(f)

MAX_SEQ_LEN = 10
# ---------------- FEATURE STATS (dynamic from Excel) ----------------



# Numeric features used by the model (5 features expected by the RNN)
numeric_features_list = [
    'amount_deviation_score',
    'location_change_flag',
    'is_new_device',
    'hour_sin',
    'hour_cos'
]
# ---------------- FEATURE STATS (dynamic from Excel) ----------------
feature_stats = {'means': {}, 'stds': {}}
try:
    df_train = pd.read_excel(r"C:\Users\Nishok\Desktop\datasetgeek.xlsx")
    df_train['transaction_type_id'] = df_train.get('transaction_type', pd.Series()).map(txn_type2id).fillna(0).astype(int)
    df_train['merchant_category_id'] = df_train.get('merchant_category', pd.Series()).map(merchant2id).fillna(0).astype(int)
    if 'timestamp' in df_train.columns:
        _, df_train['hour_sin'], df_train['hour_cos'] = zip(*df_train['timestamp'].apply(time_to_float))
    else:
        df_train['hour_sin'] = 0.0
        df_train['hour_cos'] = 1.0
    for feat in numeric_features_list:
        if feat in df_train.columns:
            mean_val = df_train[feat].mean()
            std_val = df_train[feat].std(ddof=1)
            if std_val == 0 or pd.isna(std_val):
                std_val = 1.0
            feature_stats['means'][feat] = float(mean_val)
            feature_stats['stds'][feat] = float(std_val)
except Exception as e:
    print("Error computing feature stats from Excel:", e)


categorical_features_list = ['transaction_type_id', 'merchant_category_id']

def time_to_float(t):
    try:
        if t is None:
            h = 0; m = 0
        elif hasattr(t, 'hour') and hasattr(t, 'minute'):
            h = int(t.hour); m = int(t.minute)
        else:
            s = str(t).strip()
            try:
                dt = datetime.fromisoformat(s)
                h = dt.hour; m = dt.minute
            except Exception:
                try:
                    time_part = s.split()[-1]
                    hour_min = time_part.split(':')[:2]
                    h = int(hour_min[0]); m = int(hour_min[1])
                except Exception:
                    try:
                        hour_min = s.split(':')[:2]
                        h = int(hour_min[0]); m = int(hour_min[1])
                    except Exception:
                        h = 0; m = 0
        hour_float = h + m/60
        hour_sin = np.sin(2*np.pi*hour_float/24)
        hour_cos = np.cos(2*np.pi*hour_float/24)
        return hour_float, hour_sin, hour_cos
    except Exception:
        traceback.print_exc()
        return 0.0, 0.0, 0.0

def preprocess_transaction(df):
    df = df.copy()

    # ensure timestamp column exists when calling preprocess
    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now().strftime('%H:%M')

    df['hour_float'], df['hour_sin'], df['hour_cos'] = zip(*df['timestamp'].apply(time_to_float))
    df['transaction_type_id'] = df.get('transaction_type', pd.Series()).map(txn_type2id).fillna(0).astype(int)
    df['merchant_category_id'] = df.get('merchant_category', pd.Series()).map(merchant2id).fillna(0).astype(int)

    numeric_array = df[numeric_features_list].values.astype(np.float32)
    categorical_array = df[['transaction_type_id','merchant_category_id']].values.astype(np.int32)

    pad_len = MAX_SEQ_LEN - numeric_array.shape[0]

    if pad_len > 0:
        numeric_array = np.pad(numeric_array, ((pad_len,0),(0,0)), mode='constant')
        categorical_array = np.pad(categorical_array, ((pad_len,0),(0,0)), mode='constant')

    numeric_array = np.expand_dims(numeric_array, axis=0)
    categorical_array = np.expand_dims(categorical_array, axis=0)

    return numeric_array, categorical_array

# NOTE: SHAP explainer will be initialized later after DB connection is available.
explainer = None

# Load feature stats (absolute path on Desktop)
feature_stats = None
feature_stats_path = r"C:\Users\Nishok\Desktop\feature_stats.pkl"
try:
    with open(feature_stats_path, 'rb') as f:
        feature_stats = pickle.load(f)
        print('Loaded feature_stats from', feature_stats_path)
except Exception as e:
    feature_stats = None
    print('feature_stats.pkl not found or failed to load:', e)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("transactions.db", check_same_thread=False)
c = conn.cursor()

# canonical complete schema: columns we will use/insert (name -> SQL type)
REQUIRED_COLUMNS = {
    'transaction_id': 'TEXT',
    'user_id': 'TEXT',
    'amount': 'REAL',
    'transaction_type': 'TEXT',
    'merchant_category': 'TEXT',
    'location': 'INTEGER',
    'timestamp': 'TEXT',
    'avg_amount_last_3_txns': 'REAL',
    'transactions_today': 'INTEGER',
    'session_length': 'REAL',
    'is_new_device': 'INTEGER',
    'avg_daily_transactions_30d': 'REAL',
    'avg_transaction_amount_30d': 'REAL',
    'amount_deviation_score': 'REAL',
    'location_change_flag': 'INTEGER',
    'hour_sin': 'REAL',
    'hour_cos': 'REAL',
    'has_previous_history': 'INTEGER',
    # categorical ids used for embedding
    'transaction_type_id': 'INTEGER',
    'merchant_category_id': 'INTEGER'
}

def ensure_transactions_table_and_columns():
    # Create base table if missing (only columns that are safe). We'll ensure all required cols after.
    c.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT,
        user_id TEXT,
        amount REAL,
        transaction_type TEXT,
        merchant_category TEXT,
        location INTEGER,
        timestamp TEXT
    )
    """)
    conn.commit()

    # read existing columns
    existing = [row[1] for row in c.execute("PRAGMA table_info(transactions)").fetchall()]

    # add any missing columns from REQUIRED_COLUMNS using ALTER TABLE
    for col, col_type in REQUIRED_COLUMNS.items():
        if col not in existing:
            try:
                alter_sql = f"ALTER TABLE transactions ADD COLUMN {col} {col_type}"
                c.execute(alter_sql)
                print(f"Added missing column: {col} {col_type}")
            except Exception as e:
                print(f"Failed to add column {col}: {e}")
    conn.commit()

# ensure device_fingerprints exists (used by check_new_device)
c.execute("""
CREATE TABLE IF NOT EXISTS device_fingerprints (
    user_id TEXT,
    fingerprint TEXT
)
""")
conn.commit()

# make sure transactions table and all columns exist
ensure_transactions_table_and_columns()

# ---------------- SHAP EXPLAINER SETUP (AFTER DB READY) ----------------
try:
    bg_df = None
    bg_path = r"C:\Users\Nishok\Desktop\datasetgeek.xlsx"
    if os.path.exists(bg_path):
        try:
            df_bg = pd.read_excel(bg_path)
            if not df_bg.empty:
                bg_df = df_bg.head(50)
        except Exception:
            bg_df = None

    if bg_df is None:
        try:
            raw_bg = pd.read_sql("SELECT * FROM transactions ORDER BY ROWID DESC LIMIT 50", conn)
            if not raw_bg.empty:
                bg_df = raw_bg
        except Exception:
            bg_df = None

    if bg_df is not None and not bg_df.empty:
        if len(bg_df) >= MAX_SEQ_LEN:
            bg_sample = bg_df.tail(MAX_SEQ_LEN).copy()
        else:
            bg_sample = bg_df.copy()
        bg_numeric, bg_cat = preprocess_transaction(bg_sample)
        explainer = shap.DeepExplainer(model, [bg_numeric, bg_cat])
        print('SHAP explainer initialized')
    else:
        print('No background data available for SHAP; skipping explainer initialization')
except Exception as e:
    explainer = None
    print('Failed to initialize SHAP explainer:', e)

# ---------------- DEVICE TRACKING ----------------
def get_client_ip():
    if request.headers.get("X-Forwarded-For"):
        return request.headers.get("X-Forwarded-For").split(",")[0]
    return request.remote_addr

def generate_fingerprint():
    ip = get_client_ip()
    user_agent = request.headers.get("User-Agent", "unknown")
    return f"{ip}_{user_agent}"

def check_new_device(user_id):
    fingerprint = generate_fingerprint()
    result = c.execute("""
        SELECT fingerprint FROM device_fingerprints
        WHERE user_id=? AND fingerprint=?
    """, (user_id, fingerprint)).fetchone()
    if result:
        return 0
    else:
        c.execute("INSERT INTO device_fingerprints (user_id, fingerprint) VALUES (?, ?)", (user_id, fingerprint))
        conn.commit()
        return 1

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return send_file('index.html')

# ---------------- PREDICT ROUTE ----------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    user_id = data['user_id']
    transaction_id = data['transaction_id']
    amount = float(data['amount'])
    transaction_type = data.get('transaction_type', '')
    merchant_category = data.get('merchant_category', '')

    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')  # full iso-like string
    today_str = now.strftime('%Y-%m-%d')

    # fetch last 30 amounts for this user (if any)
    df_last_30 = pd.read_sql(f"""
        SELECT amount, timestamp
        FROM transactions
        WHERE user_id=?
        ORDER BY ROWID DESC LIMIT 30
    """, conn, params=(user_id,))

    if len(df_last_30) > 0:
        df_last_3 = df_last_30.head(3)
        avg_amount_last_3_txns = df_last_3['amount'].mean() if not df_last_3.empty else amount
        avg_last_30_txns = df_last_30['amount'].mean()
        has_previous_history = 1
    else:
        avg_amount_last_3_txns = amount
        avg_last_30_txns = amount
        has_previous_history = 0

    # deviation based on last 30 (or fewer if not available)
    amount_deviation_score = (amount - avg_last_30_txns) / (avg_last_30_txns + 1e-6)

    is_new_device = check_new_device(user_id)
    location_change_flag = 0

    # compute hour sin/cos from the timestamp string
    _, hour_sin, hour_cos = time_to_float(timestamp)

    # build df_input with all columns we may insert (use defaults for missing ones)
    df_input = pd.DataFrame([{
        'transaction_id': transaction_id,
        'user_id': user_id,
        'amount': amount,
        'transaction_type': transaction_type,
        'merchant_category': merchant_category,
        'location': 0,
        'timestamp': timestamp,
        'avg_amount_last_3_txns': avg_amount_last_3_txns,
        'transactions_today': 1,
        'session_length': 100,
        'is_new_device': int(is_new_device),
        'avg_daily_transactions_30d': 0.0,
        'avg_transaction_amount_30d': avg_last_30_txns,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'amount_deviation_score': amount_deviation_score,
        'location_change_flag': int(location_change_flag),
        'has_previous_history': int(has_previous_history)
    }])

    # map categorical ids safely (default 0)
    df_input['transaction_type_id'] = df_input['transaction_type'].map(txn_type2id).fillna(0).astype(int)
    df_input['merchant_category_id'] = df_input['merchant_category'].map(merchant2id).fillna(0).astype(int)

    # preprocess and predict
    numeric_array = df_input[numeric_features_list].values.astype(np.float32)
    categorical_array = df_input[categorical_features_list].values.astype(np.int32)

    pad_len = MAX_SEQ_LEN - numeric_array.shape[0]
    if pad_len > 0:
        numeric_array = np.pad(numeric_array, ((pad_len,0),(0,0)), mode='constant')
        categorical_array = np.pad(categorical_array, ((pad_len,0),(0,0)), mode='constant')

    numeric_array = np.expand_dims(numeric_array, axis=0)
    categorical_array = np.expand_dims(categorical_array, axis=0)

    preds = model.predict([numeric_array, categorical_array], verbose=0)
    pred_class = int(np.argmax(preds, axis=1)[0])

    # ensure DB schema (just in case) before inserting
    ensure_transactions_table_and_columns()

    # append to DB
    try:
        df_input.to_sql('transactions', conn, if_exists='append', index=False)
    except Exception as e:
        # if insert fails, print useful debugging info and fallback to per-column insert
        print("Failed df_input.to_sql append:", e)
        print("df_input columns:", df_input.columns.tolist())
        # fallback: try inserting using parameterized SQL to avoid wide to_sql issues
        cols = df_input.columns.tolist()
        placeholders = ", ".join(["?"] * len(cols))
        insert_sql = f"INSERT INTO transactions ({', '.join(cols)}) VALUES ({placeholders})"
        try:
            c.executemany(insert_sql, df_input.values.tolist())
            conn.commit()
        except Exception as e2:
            print("Fallback insert also failed:", e2)

    # SHAP explanation for fraud class only (optional)
    shap_details = None
    try:
        if explainer is not None and pred_class == 1:
            shap_values = explainer.shap_values([numeric_array, categorical_array])
            shap_for_class = shap_values[pred_class][0]
            shap_mean = np.mean(shap_for_class, axis=0)
            feature_names = numeric_features_list + categorical_features_list
            feature_impact = sorted(zip(feature_names, shap_mean), key=lambda x: abs(x[1]), reverse=True)[:5]
            shap_details = [{'feature': f, 'value': float(v)} for f, v in feature_impact]
    except Exception as e:
        print("SHAP error:", e)
        shap_details = None

    # Compute feature deviations (z-scores) for numeric features when fraud predicted
    feature_deviations = None
    try:
      if feature_stats is not None and pred_class == 1:
        means = feature_stats.get('means', {})
        stds = feature_stats.get('stds', {})
        deviations = []
        for feat in numeric_features_list:
            if feat in df_input.columns:
                val = float(df_input[feat].iloc[0])
            else:
                val = 0.0
            mean = float(means.get(feat, 0.0))
            std = float(stds.get(feat, 0.0)) if stds.get(feat, None) is not None else 1.0
            if std == 0:
                std = 1.0
            z = (val - mean) / std
            deviations.append((feat, z, abs(z), val, mean, std))
        deviations_sorted = sorted(deviations, key=lambda x: x[2], reverse=True)
        feature_deviations = [
            {'feature': d[0], 'z': float(d[1]), 'abs_z': float(d[2]), 'value': float(d[3]),
             'mean': float(d[4]), 'std': float(d[5])}
            for d in deviations_sorted[:3]
        ]
        print('Top deviations for fraud:', feature_deviations)
    except Exception as e:
     feature_deviations = None
     print('Error computing a feature deviation in testing1:', e)

    # Build readable explanation text for the frontend
    explanation_text = None
    if pred_class == 1 and feature_deviations:
        lines = ["Fraud Likely – Top Deviations:"]
        for item in feature_deviations:
          lines.append(
            f"{item['feature']} -> value {item['value']} (z = {item['z']:.2f})"
          )
        explanation_text = "\n".join(lines)


    return jsonify({
        "transaction_id": transaction_id,
        "user_id": user_id,
        "amount": amount,
        "avg_last_3_txns": avg_amount_last_3_txns,
        "avg_last_30_txns": avg_last_30_txns,
        "amount_deviation_score": amount_deviation_score,
        "is_new_device": int(is_new_device),
        "location_change_flag": int(location_change_flag),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "transactions_today": 1,
        "avg_daily_transactions_30d": 0,
        "has_previous_history": int(has_previous_history),
        "predicted_class": pred_class,
        "probabilities": preds[0].tolist(),
        "shap": shap_details,
        "feature_deviations": feature_deviations,
        "explanation_text": explanation_text

    })

@app.route('/version')
def version():
    try:
        path = os.path.realpath(__file__)
        mtime = os.path.getmtime(path)
        with open(path, 'rb') as f:
            data = f.read()
        h = hashlib.md5(data).hexdigest()
        return jsonify({'file': path, 'mtime': mtime, 'md5': h})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history', methods=['GET'])
def history():
    user_id = request.args.get('user_id', '')
    if not user_id:
        return jsonify({'history': []})
    try:
        df_hist = pd.read_sql(f"""
            SELECT transaction_id, amount, timestamp
            FROM transactions
            WHERE user_id=?
            ORDER BY ROWID DESC LIMIT 10
        """, conn, params=(user_id,))
    except Exception as e:
        return jsonify({'history': [], 'error': str(e)})
    if df_hist.empty:
        return jsonify({'history': []})
    df_hist = df_hist.iloc[::-1]
    hist = []
    for _, row in df_hist.iterrows():
        hist.append({'transaction_id': row['transaction_id'], 'amount': float(row['amount']), 'timestamp': row['timestamp']})
    return jsonify({'history': hist})
print("hello")
if __name__ == "__main__":
    app.run(debug=True)
