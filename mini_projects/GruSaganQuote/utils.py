import re


def load_and_clean_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().lower()
    text = re.sub(r"[^a-z0-9\s.,;:!?\'\"\n()-]", "", text)  # keep basic punctuation
    text = re.sub(r"\s+", " ", text)
    return text


def create_char_mappings(text):
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char


def create_sequences(text, char2idx, seq_length=100):
    # helps predits the next character at each time stamp
    encoded = [char2idx[c] for c in text]
    X, y = [], []
    for i in range(len(encoded) - seq_length):
        X.append(encoded[i : i + seq_length])
        y.append(encoded[i + 1 : i + seq_length + 1])
    return X, y


if __name__ == "__main__":
    text = "hi this is some text"
    a, b = create_char_mappings(text)
    print(f"{a} \n {b}\n")
    X, y = create_sequences(text, a, 5)
    print(f"{X} \n{y}")
