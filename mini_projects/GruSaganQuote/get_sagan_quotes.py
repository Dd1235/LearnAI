import wikiquote

# Get a list of quote lines (split and parsed cleanly)
quotes = wikiquote.quotes("Carl Sagan", max_quotes=1000, lang="en")

with open("carl_sagan_quotes.txt", "w", encoding="utf-8") as f:
    for quote in quotes:
        f.write(quote.strip() + "\n")

print(f"Saved len{quotes} quotes of Carl Sagan from wikiquotes")
