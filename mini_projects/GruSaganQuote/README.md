
```markdown
# Carl Sagan GRU Quote Generator

A character-level neural text generation model trained to emulate the writing style of Carl Sagan, using a Gated Recurrent Unit (GRU). Built from scratch in PyTorch, with custom preprocessing and temperature-based sampling.

> “We are a way for the cosmos to know itself.” — Carl Sagan

---

## Project Summary

This project uses a GRU-based sequence model to generate cosmic, poetic-style quotes inspired by Carl Sagan. It's trained on a dataset of over 1000  quotes from wikiquote

### Features
- Character-level GRU sequence model
- Temperature sampling to control creativity
- Manual data collection 

---
##  Folder Structure

```
GruSaganQuote/
├── model/
│   └── gru_model.py           # GRU architecture
├── carl_sagan_quotes.txt      #  text corpus
├── get_sagan_quotes.py        # Wikiquote scraping
├── train.py                   # Training script
├── utils.py                   # Preprocessing helpers
├── sagan_gru.pth              # Trained model weights
├── gru_generate.ipynb         # Inference & examples
└── README.md                  # you're reading it ^_^
```

---

## Sample Output

| Temperature | Output Example |
|-------------|----------------|
| `0.8` | *"we are on the entire umitical. that's insights in the plong to believe..."* |
| `0.4` | *"we are the stars of science and the extray, but in many the fincledition..."* |
| `0.1` | *"we are the could be so to thing our contrary to the inside of the stars..."* |

See `gru_generate.ipynb` 

---

## Technical Concepts

- **GRU architecture** to retain temporal dependencies
- **Character-level modeling** for open-vocab text generation
- **Temperature-controlled sampling**:
  - `High (1.0)`: more creative, chaotic
  - `Low (0.1)`: safer, more repetitive
- **CrossEntropy Loss** over predicted character sequences

---

## Future Directions

- Try LSTM or Transformer-based models
- Compare character vs word-level generation
- Make changes to architecture for better results, only used 1000 examples due to limited resources
- Add interactive UI using Gradio or Streamlit

---


> “Somewhere, something incredible is waiting to be known.” — Carl Sagan
```

---


## To Run

`get_sagan_quotes.py` - get Carl Sagan quotes from wikiquote using the wikiquote library

PS if using conda, you cannot install the library, use pip, while using pip to make sure it installs it in your conda env that is activated run `python -m pip install wikiquote`

