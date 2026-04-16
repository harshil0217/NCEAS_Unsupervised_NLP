import os
import sys
import pandas as pd
import nltk
from nltk.corpus import reuters

target_folder = "src"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.getcwd()
while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        raise FileNotFoundError(f"{target_folder} not found in the directory tree.")
    current_dir = parent_dir

os.chdir(current_dir)
sys.path.insert(0, current_dir)

DATA_GEN_DIR = "data/rcv1"
os.makedirs(DATA_GEN_DIR, exist_ok=True)

filename = f'{DATA_GEN_DIR}/rcv1.csv'

def main():
    print("Fetching NLTK Reuters...")
    nltk.download('reuters')
    
    file_ids = reuters.fileids()
    data_list = []
    for f in file_ids:
        text = reuters.raw(f).replace('\n', ' ').strip()
        cats = reuters.categories(f)
        if len(cats) < 2: continue

        data_list.append({
            "item_id": f,
            "topic": text,
            "category_0": cats[0],
            "category_1": cats[1],
        })

    df = pd.DataFrame(data_list)
    df.to_csv(filename, index=False)
    print(f"Success! Data saved to {filename}")

if __name__ == "__main__":
    main()
