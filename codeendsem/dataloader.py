# DATA LOADER TO RUN PURECNN, XGBOOST
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_dataset(tfrecord_path):
    print(f"Loading data from: {tfrecord_path}")
    X_list, y_list = [], []
    for raw_record in tqdm(tf.data.TFRecordDataset([tfrecord_path])):
        ex = tf.train.Example.FromString(raw_record.numpy())
        y_list.append(ex.features.feature['label'].int64_list.value[0])
        img_bytes = ex.features.feature['image'].bytes_list.value[0]
        # Reshape and transpose for PyTorch (Channels, Height, Width)
        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(64, 64, 6).transpose(2, 0, 1)
        X_list.append(img)
    return np.array(X_list), np.array(y_list)

# 1. Load the merged file
tfrecord_file = "/content/drive/MyDrive/Project_WISE/data/merged_64x64_wells.tfrecord"
X_patches, y = load_dataset(tfrecord_file)

# 2. Generate the 12 tabular features (Mean and Std) for XGBoost
print("Generating tabular features...")
X_tabular = np.hstack([X_patches.mean(axis=(2, 3)), X_patches.std(axis=(2, 3))])

# 3. Split the data globally so all models can use the exact same test set
print("Splitting data into Train and Test sets...")
X_p_train, X_p_test, X_t_train, X_t_test, y_train, y_test = train_test_split(
    X_patches, X_tabular, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nSuccess! Training patches shape: {X_p_train.shape}")
print(f"Training tabular shape: {X_t_train.shape}")
print("You are now clear to run the Pure CNN and XGBoost cells!")