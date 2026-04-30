import tensorflow as tf
import numpy as np

# Pointing to the new merged file you just created
TFRECORD_PATH = "/content/drive/MyDrive/Project_WISE/data/merged_64x64_wells.tfrecord"

def inspect_tfrecord(file_path, num_samples=3):
    print(f"Inspecting TFRecord: {file_path}\n")
    raw_dataset = tf.data.TFRecordDataset([file_path])

    for i, raw_record in enumerate(raw_dataset.take(num_samples)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        label = example.features.feature['label'].int64_list.value[0]
        height = example.features.feature['height'].int64_list.value[0]
        width = example.features.feature['width'].int64_list.value[0]
        bands = example.features.feature['bands'].int64_list.value[0]

        img_bytes = example.features.feature['image'].bytes_list.value[0]
        img_flat = np.frombuffer(img_bytes, dtype=np.float32)
        img = img_flat.reshape(height, width, bands)

        print(f"--- Sample {i+1} ---")
        print(f"Label (0=Unsuitable, 1=Suitable): {label}")
        print(f"Tensor Shape: {img.shape}  <-- Should be (64, 64, 6)")
        print(f"Data Min: {img.min():.4f} | Data Max: {img.max():.4f}")
        print("-" * 25)

inspect_tfrecord(TFRECORD_PATH, num_samples=3)