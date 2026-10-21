# IADAI201-1000442-ZENE-SOPHIE-ANAND-SA
# ParkVision AI

AI-powered parking occupancy analysis using MobileNetV2
and Streamlit.

## Problem
Parking availability is difficult to estimate quickly.
ParkVision AI classifies individual parking-slot images
as Empty or Occupied and estimates overall utilisation.

## Input
A parking-lot photograph with a known, regular slot layout.

## Output
- Total slots
- Occupied slots
- Available slots
- Occupancy percentage
- Congestion level
- Annotated slot predictions
- CSV report

## Model
MobileNetV2 transfer learning with 224x224 images.
Training uses rotation, horizontal flipping and brightness
augmentation.

## Dataset
Prepare individual, labeled parking-slot crops in:
- source_data/empty/
- source_data/occupied/

At least 100 images per class are required for this project.
The preparation script creates a 70/15/15 train/validation/test
split.

## Training
```bash
python prepare_dataset.py
python train.py
