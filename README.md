# IADAI201-1000442-ZENE-SOPHIE-ANAND-SA

### Option A — PKLot (recommended for the report)
Download [PKLot](http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz) (sunny / cloudy / rainy slot crops). Copy at least 100 images into:
```text
data/raw/empty/
data/raw/occupied/
```
### Option B — bundled generator (runs immediately)
```bash
python scripts/generate_synthetic_data.py --per-class 120
python scripts/prepare_dataset.py
```
`prepare_dataset.py` writes:
```text
data/splits/train/  (70%)
data/splits/val/    (15%)
data/splits/test/   (15%)
```
each with `empty/` and `occupied/`.
---
## Model development
```bash
python -m pip install -r requirements.txt
python scripts/generate_synthetic_data.py --per-class 120
python scripts/prepare_dataset.py
python scripts/train.py
python scripts/evaluate.py
```
**MobileNetV2 (recommended for the report, needs TensorFlow):**
```bash
python -m pip install tensorflow
python scripts/train_mobilenet.py --epochs 18 --batch-size 16
```
- Preferred backbone: **MobileNetV2** (ImageNet), 224×224, softmax (empty / occupied).
- Augmentation: flip, rotation, brightness (and extra crops in the CPU trainer).
- Epochs: **15–30** for MobileNet; MLP uses `max_iter` in the same range.
- Metrics: `models/metrics.json`
- Confusion matrix: `models/confusion_matrix.png`
### Improving the model
1. Replace synthetic crops with real PKLot images.
2. Fix mislabelled files in `data/raw`.
3. Unfreeze the last MobileNet blocks and train at `1e-5` for a few extra epochs.
4. Increase `--per-class` or add night-time photos.
---
## Parking insight logic
After each slot is labelled:
```text
occupancy % = occupied / total × 100
```
| Occupancy | Congestion | Recommendation |
| --- | --- | --- |
| < 40% | Low | Slots available — proceed |
| 40–75% | Moderate | Some slots available — proceed, but expect limited space |
| > 75% | High | Parking full — try another area |
Implemented in `src/insights.py`.
---
## Run the web app locally
```bash
streamlit run app.py
```
Users can:
1. Upload a parking image (or pick a demo lot).
2. Set grid rows/columns for a marked lot, **or** classify a single slot crop.
3. See the annotated image (green = empty, red = occupied).
4. Read total / occupied / available / occupancy % and the recommendation.
---
## Testing
```bash
python -m pytest tests -q
python scripts/evaluate.py
```
Use the Streamlit demo images under different congestion levels (`lot_low_congestion.jpg`, `lot_moderate_congestion.jpg`, `lot_high_congestion.jpg`) plus your own photos at different angles and lighting. Note failures (glare, distant cameras, wrong grid size) and tighten the grid or add more labelled crops.
---
## Deploy on Streamlit Cloud
1. Create a GitHub repository and push this folder (including `models/parking_mobilenet.keras` after training).
2. Open [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. **New app** → select this repo → main file `app.py`.
4. Share the public URL in your report.
`packages.txt` installs system libraries some Cloud images need for OpenCV.
---
## GitHub checklist (rubric)
- [x] Problem and input–output mapping in this README
- [x] MobileNetV2 training + accuracy / confusion matrix scripts
- [x] Insight thresholds and recommendations
- [x] Streamlit UI with image, predictions, metrics
- [x] Test split + demo lots for unseen-style checks
- [x] Deployment path (Streamlit Cloud + this repo)
After you deploy, add the live URL here:
```text
Live app: https://<your-app>.streamlit.app
```
---
## Screenshots to capture for the report
1. Streamlit home page with demo lot.
2. Annotated lot (green/red boxes).
3. Metrics + congestion recommendation.
4. Confusion matrix from `models/confusion_matrix.png`.
5. Training accuracy/loss curves.
