# arXiv Dataset

A 30,000 paper subset of the full arXiv metadata, used for benchmark evaluation.

## Download

1. Download `arxiv-metadata-oai-snapshot.json` from Kaggle:
   https://www.kaggle.com/datasets/Cornell-University/arxiv

2. Place it at:
   ```
   data/arxiv/arxiv-metadata-oai-snapshot.json
   ```

3. Run the cleaning script to generate `arxiv_clean.csv`:
   ```bash
   python data/arxiv/clean_arxiv.py
   ```

## Expected output
```
data/arxiv/arxiv_clean.csv
```
