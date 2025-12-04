import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.environ.get('APP_DATA_DIR', os.path.join(BASE_DIR, 'data'))
os.makedirs(DATA_DIR, exist_ok=True)
DATASETX_PATH = os.environ.get('DATASETX_PATH', os.path.join(DATA_DIR, 'datasetgeek.xlsx'))

# =========================
# 1️⃣ Load Excel file
# =========================
# Load from configured data directory; fallback will be attempted automatically by pandas if path doesn't exist
file_path = DATASETX_PATH
df = pd.read_excel(file_path)
print(df.head())
print(df.columns)

# =========================
# 2️⃣ Timestamp → hour_float + circular encoding
# =========================
def time_to_float(t):
    if isinstance(t, pd.Timestamp):
        return t.hour + t.minute / 60
    elif isinstance(t, str):
        h, m = map(int, t.split(':'))
        return h + m/60
    else:
        return 0

df['hour_float'] = df['timestamp'].apply(time_to_float)
df['hour_sin'] = np.sin(2 * np.pi * df['hour_float'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_float'] / 24)

# =========================
# 3️⃣ Categorical → integer IDs for embedding
# =========================
txn_types = df['transaction_type'].unique()
txn_type2id = {t:i for i,t in enumerate(txn_types)}
df['transaction_type_id'] = df['transaction_type'].map(txn_type2id)

merchant_cats = df['merchant_category'].unique()
merchant2id = {m:i for i,m in enumerate(merchant_cats)}
df['merchant_category_id'] = df['merchant_category'].map(merchant2id)

# =========================
# 4️⃣ Numeric & categorical features
# =========================
# =========================
# 4️⃣ Numeric & categorical features (modified)
# =========================
numeric_features = [
    'amount_deviation_score',
    'location_change_flag',
    'is_new_device',
    'hour_sin',
    'hour_cos'
]

categorical_features = ['transaction_type_id', 'merchant_category_id']


# =========================
# 5️⃣ Build sequences per user
# =========================
MAX_SEQ_LEN = 10
user_sequences_numeric, user_sequences_cat, user_labels = [], [], []

df = df.sort_values(['user_id', 'timestamp'])

for user_id, group in df.groupby('user_id'):
    group = group.reset_index(drop=True)
    seq_numeric = []
    seq_cat = []
    seq_labels = []
    
    for i in range(len(group)):
        num_feat = group.loc[i, numeric_features].values.astype(float)
        cat_feat = group.loc[i, categorical_features].values.astype(int)
        seq_numeric.append(num_feat)
        seq_cat.append(cat_feat)
        seq_labels.append(group.loc[i, 'is_fraud'])
    
    # pad sequences if too short
    while len(seq_numeric) < MAX_SEQ_LEN:
        seq_numeric.insert(0, np.zeros(len(numeric_features)))
        seq_cat.insert(0, np.zeros(len(categorical_features)))
        seq_labels.insert(0, 0)
    
    user_sequences_numeric.append(seq_numeric[-MAX_SEQ_LEN:])
    user_sequences_cat.append(seq_cat[-MAX_SEQ_LEN:])
    user_labels.append(seq_labels[-MAX_SEQ_LEN:])

user_sequences_numeric = np.array(user_sequences_numeric, dtype=np.float32)
user_sequences_cat = np.array(user_sequences_cat, dtype=np.int32)
user_labels = np.array(user_labels, dtype=np.int32)

# =========================
# 6️⃣ Build TensorFlow Dataset
# =========================
BATCH_SIZE = 2
dataset = tf.data.Dataset.from_tensor_slices((user_sequences_numeric, user_sequences_cat, user_labels))
dataset = dataset.shuffle(buffer_size=len(user_sequences_numeric)).batch(BATCH_SIZE)

# =========================
# 7️⃣ Build Keras Model
# =========================
NUM_TXN_TYPES = len(txn_types)
NUM_MERCHANTS = len(merchant_cats)
EMBED_DIM = 4
NUMERIC_DIM = len(numeric_features)
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3  # 0,1,2

# Inputs
numeric_input = Input(shape=(MAX_SEQ_LEN, NUMERIC_DIM), name='numeric_input')
cat_input = Input(shape=(MAX_SEQ_LEN, 2), dtype='int32', name='categorical_input')

# Embeddings
txn_emb = Embedding(input_dim=NUM_TXN_TYPES, output_dim=EMBED_DIM)(cat_input[:,:,0])
merchant_emb = Embedding(input_dim=NUM_MERCHANTS, output_dim=EMBED_DIM)(cat_input[:,:,1])

# Concatenate numeric + embeddings
x = Concatenate(axis=-1)([numeric_input, txn_emb, merchant_emb])

# LSTM
x = LSTM(HIDDEN_SIZE, return_sequences=False)(x)
output = Dense(OUTPUT_SIZE, activation='softmax')(x)

model = Model(inputs=[numeric_input, cat_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# =========================
# 8️⃣ Training Loop
# =========================
EPOCHS = 5
for epoch in range(EPOCHS):
    for batch_numeric, batch_cat, batch_labels in dataset:
        # We only need the last label in sequence
        batch_labels_last = batch_labels[:,-1]
        loss, acc = model.train_on_batch([batch_numeric, batch_cat], batch_labels_last)
    print(f"Epoch {epoch+1}/{EPOCHS} done, last batch loss: {loss:.4f}, acc: {acc:.4f}")

# =========================
# 9️⃣ Save model
# =========================
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(DATA_DIR, 'fraud_rnn_keras9.keras'))
model.save(MODEL_PATH)  # notice the .keras extension

print(f"✅ Model saved as {MODEL_PATH}")

# =========================
# 1️⃣ Prepare last 2 rows of each user's sequence for prediction
# =========================
# =========================
# Prepare last 2 rows for prediction
# =========================
all_preds = []
all_labels = []

for batch_numeric, batch_cat, batch_labels in dataset:
    last_numeric = batch_numeric[:, -2:, :]
    last_cat = batch_cat[:, -2:, :]
    last_labels = batch_labels[:, -2:]
    
    # pad sequences to MAX_SEQ_LEN (prepend zeros)
    pad_len = MAX_SEQ_LEN - last_numeric.shape[1]
    if pad_len > 0:
        last_numeric = np.pad(last_numeric, ((0,0),(pad_len,0),(0,0)), mode='constant')
        last_cat = np.pad(last_cat, ((0,0),(pad_len,0),(0,0)), mode='constant')
    
    preds = model.predict([last_numeric, last_cat], verbose=0)
    preds_class = np.argmax(preds, axis=1)
    
    all_preds.extend(preds_class.tolist())
    all_labels.extend(last_labels[:,-1].numpy().tolist())

# =========================
# 2️⃣ Compute accuracy
# =========================
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
accuracy = (all_preds == all_labels).mean()
print("Predictions:", all_preds)
print("Actual labels:", all_labels)
print(f"Accuracy on last 2 rows of training data: {accuracy*100:.2f}%")

# =========================
# 3️⃣ Save encodings
# =========================
TXN_TYPE_PATH = os.environ.get('TXN_TYPE_PATH', os.path.join(DATA_DIR, 'txn_type2id.pkl'))
MERCHANT_PATH = os.environ.get('MERCHANT_PATH', os.path.join(DATA_DIR, 'merchant2id.pkl'))
with open(TXN_TYPE_PATH, "wb") as f:
    pickle.dump(txn_type2id, f)
with open(MERCHANT_PATH, "wb") as f:
    pickle.dump(merchant2id, f)
print("✅ Encodings saved as pickle files")

# =========================
# 4️⃣ Compute & save feature stats (means, stds)
# =========================
FEATURE_STATS_PATH = os.environ.get('FEATURE_STATS_PATH', os.path.join(DATA_DIR, 'feature_stats.pkl'))
features_for_stats = numeric_features  # same numeric features used in model
feature_means = df[features_for_stats].mean().to_dict()
feature_stds = df[features_for_stats].std().to_dict()
with open(FEATURE_STATS_PATH, "wb") as f:
    pickle.dump({'means': feature_means, 'stds': feature_stds}, f)
print(f"✅ Feature stats saved to {FEATURE_STATS_PATH}")
