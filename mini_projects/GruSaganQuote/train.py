import torch
import torch.nn as nn
from model.gru_model import CharGRU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import *

# Parameters
seq_length = 100
hidden_size = 256
batch_size = 64
epochs = 30
lr = 0.003

# Load data
text = load_and_clean_text("carl_sagan_quotes.txt")
char2idx, idx2char = create_char_mappings(text)
X, y = create_sequences(text, char2idx, seq_length)

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = CharGRU(len(char2idx), hidden_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        hidden = model.init_hidden(batch_x.size(0)).to(device)

        optimizer.zero_grad()
        output, hidden = model(batch_x, hidden)
        loss = criterion(output, batch_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"\nEpoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "sagan_gru.pth")
