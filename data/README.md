# Dataset layout
Use this folder structure so Keras `image_dataset_from_directory` works without extra code:
```text
data/
  raw/
    empty/       # >= 100 slot crops
    occupied/    # >= 100 slot crops
  splits/        # created by scripts/prepare_dataset.py
    train/{empty,occupied}
    val/{empty,occupied}
    test/{empty,occupied}
  samples/       # full parking-lot images for the Streamlit demo
```
## Recommended real dataset
[PKLot](http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz) contains thousands of parking-slot images in sunny, cloudy, and rainy weather with occupied/empty labels. For this assignment, copy at least 100 crops per class into `data/raw/empty` and `data/raw/occupied`.
Then:
```bash
python scripts/prepare_dataset.py
python scripts/train.py
python scripts/evaluate.py
```
If you do not have PKLot yet, generate a labelled subset:
```bash
python scripts/generate_synthetic_data.py --per-class 120
```
