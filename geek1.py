import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# =========================
# 1️⃣ Load Excel file
# =========================
file_path = r"C:\Users\Nishok\Desktop\datasetgeek.xlsx"
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
numeric_features = [
    'amount',
    'avg_amount_last_3_txns',
    'transactions_today',
    'session_length',
    'is_new_device',
    'avg_daily_transactions_30d',
    'avg_transaction_amount_30d',
    'amount_deviation_score',
    'location_change_flag',
    'has_previous_history',
    'hour_sin',
    'hour_cos'
]

categorical_features = ['transaction_type_id', 'merchant_category_id']

# =========================
# 5️⃣ Build sequences per user
# =========================
MAX_SEQ_LEN = 10
user_sequences, user_labels = [], []

df = df.sort_values(['user_id', 'timestamp'])

for user_id, group in df.groupby('user_id'):
    group = group.reset_index(drop=True)
    seq_features = []
    seq_labels = []
    
    for i in range(len(group)):
        num_feat = group.loc[i, numeric_features].values.astype(float)
        cat_feat = group.loc[i, categorical_features].values.astype(int)
        seq_features.append({'numeric': num_feat, 'categorical': cat_feat})
        seq_labels.append(group.loc[i, 'is_fraud'])
    
    # pad sequences if too short
    while len(seq_features) < MAX_SEQ_LEN:
        seq_features.insert(0, {'numeric': np.zeros(len(numeric_features)), 
                                'categorical': np.zeros(len(categorical_features), dtype=int)})
        seq_labels.insert(0, 0)
    
    user_sequences.append(seq_features[-MAX_SEQ_LEN:])
    user_labels.append(seq_labels[-MAX_SEQ_LEN:])

# =========================
# 6️⃣ Custom Dataset
# =========================
class FraudDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        numeric_seq = torch.tensor([x['numeric'] for x in self.sequences[idx]], dtype=torch.float32)
        categorical_seq = torch.tensor([x['categorical'] for x in self.sequences[idx]], dtype=torch.long)
        labels_seq = torch.tensor(self.labels[idx], dtype=torch.long)
        return numeric_seq, categorical_seq, labels_seq

dataset = FraudDataset(user_sequences, user_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# =========================
# 7️⃣ RNN Model with Embeddings
# =========================
NUM_TXN_TYPES = len(txn_types)
NUM_MERCHANTS = len(merchant_cats)
EMBED_DIM = 4
NUMERIC_DIM = len(numeric_features)
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3  # 0,1,2

class FraudRNN(nn.Module):
    def __init__(self):
        super(FraudRNN, self).__init__()
        self.txn_emb = nn.Embedding(NUM_TXN_TYPES, EMBED_DIM)
        self.merchant_emb = nn.Embedding(NUM_MERCHANTS, EMBED_DIM)
        self.rnn = nn.RNN(input_size=NUMERIC_DIM + EMBED_DIM*2, hidden_size=HIDDEN_SIZE, 
                          num_layers=2, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
    
    def forward(self, numeric_seq, categorical_seq):
        txn_embed = self.txn_emb(categorical_seq[:,:,0])
        merchant_embed = self.merchant_emb(categorical_seq[:,:,1])
        x = torch.cat([numeric_seq, txn_embed, merchant_embed], dim=-1)
        out, _ = self.rnn(x)
        out = self.fc(out[:,-1,:])  # last timestep
        return out  # use CrossEntropyLoss, no sigmoid

model = FraudRNN()

# =========================
# 8️⃣ Training Setup
# =========================
criterion = nn.CrossEntropyLoss()  # expects raw logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    for numeric_seq, categorical_seq, labels_seq in dataloader:
        optimizer.zero_grad()
        outputs = model(numeric_seq, categorical_seq)
        loss = criterion(outputs, labels_seq[:,-1])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} done, last batch loss: {loss.item():.4f}")

# =========================
# 9️⃣ Save model
# =========================
torch.save(model.state_dict(), "fraud_rnn.pt")
print("✅ Model saved as fraud_rnn.pt")

# =========================
# 1️⃣ Prepare last 2 rows of each user's sequence for prediction
# =========================
model.eval()  # set model to eval mode
all_preds = []
all_labels = []

with torch.no_grad():
    for numeric_seq, categorical_seq, labels_seq in dataloader:
        last_numeric = numeric_seq[:, -2:, :]
        last_categorical = categorical_seq[:, -2:, :]
        last_labels = labels_seq[:, -2:]
        
        outputs = model(last_numeric, last_categorical)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.tolist())
        all_labels.extend(last_labels[:, -1].tolist())

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
# 3️⃣ Save whole model & encodings
# =========================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "fraud_rnn_full11.pt")
# save weights only
torch.save(model.state_dict(), "fraud_rnn_weights.pt")

print(f"✅ Whole model saved as {desktop_path}")

# Save txn_type2id and merchant2id
with open(r"C:\Users\Nishok\Desktop\txn_type2id.pkl", "wb") as f:
    pickle.dump(txn_type2id, f)
with open(r"C:\Users\Nishok\Desktop\merchant2id.pkl", "wb") as f:
    pickle.dump(merchant2id, f)

