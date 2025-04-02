import torch
from model.gru_model import CharGRU
from utils import *


def generate(model, start, char2idx, idx2char, length=300, temperature=1.0):
    model.eval()
    input_seq = torch.tensor(
        [char2idx[c] for c in start.lower()], dtype=torch.long
    ).unsqueeze(0)
    hidden = model.init_hidden(1)
    generated = start

    for _ in range(length):
        input_seq = input_seq[:, -1:].to(next(model.parameters()).device)
        output, hidden = model(input_seq, hidden)
        output_dist = torch.softmax(output[-1] / temperature, dim=0)
        char_id = torch.multinomial(output_dist, 1).item()
        generated += idx2char[char_id]
        input_seq = torch.cat([input_seq, torch.tensor([[char_id]])], dim=1)
    return generated


# Load mappings
text = load_and_clean_text("data/carl_sagan_quotes.txt")
char2idx, idx2char = create_char_mappings(text)

# Load model
model = CharGRU(len(char2idx), hidden_size=256)
model.load_state_dict(torch.load("sagan_gru.pth", map_location="cpu"))

# Generate
seed = "we are"
output = generate(model, seed, char2idx, idx2char)
print("\nGenerated Text:\n", output)
