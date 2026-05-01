# Data Processing
- **mergeAndCrop.py**: Aggregates raw `.tfrecord.gz` batches from Google Earth Engine and crops them into `$64 \times 64$` patches.
- **checkTFRecord.py**: Utility script to verify the integrity and shape of the generated TFRecord files.
- **dataloader.py**: The core data engine. Loads images, generates 12 tabular statistical features (Mean/Std), and performs the stratified train-test split.
- **heatmap.py**: To produce heatmap codes by grid inference
# Model Architectures
- **purecnn.py**: Baseline ResNet18 model using only spatial image patches.
- **xgboost.py**: Baseline Gradient Boosting model using only the 12 tabular statistical features.
- **hybrid1.py**: Proposed Hybrid "Wide & Deep" architecture fusing ResNet spatial embeddings with a Random Forest classifier.
- **hybrid2.py**: Enhanced Hybrid architecture using ResNet spatial embeddings with an XGBoost classifier for improved recall in dry-zone detection.