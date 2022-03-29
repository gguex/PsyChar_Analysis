import pandas as pd

# The book name
book_name = "Romeo&Juliet"

# Load df
df = pd.read_csv(f"corpora/{book_name}.tsv", sep="\t")

# Get all char
all_char = df["char_from"].unique()

# Loop on char
for char in all_char:
    # Extract sentences and join them
    char_all_text = "\n".join(list(df[df["char_from"] == char]["sentence"]))
    # Save them
    with open(f"corpora/{book_name}/{char}.txt", "w") as char_file:
        char_file.write(char_all_text)