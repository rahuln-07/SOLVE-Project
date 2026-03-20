import glob, tensorflow as tf
from google.protobuf.json_format import MessageToDict
import numpy as np
import sys
import os

searchPath = os.path.join("..", "data", "drive-download-20260203T102551Z-3-001", "*.tfrecord.gz")
files = sorted(glob.glob(searchPath))

if not files:
    print(f"No TFRecord files found in: {searchPath}")
    sys.exit(1)

f = files[0]
print("Peeking file:", f)


rawIter = tf.data.TFRecordDataset([f], compression_type='GZIP').take(1).as_numpy_iterator()

try:
    raw = next(rawIter)
except StopIteration:
    print("No records in file.")
    sys.exit(1)

ex = tf.train.Example.FromString(raw)
d = MessageToDict(ex)

print("\n" + "="*40)
print(f" RECORD SUMMARY")
print("="*40)

features = d['features']['feature']

for key, val_container in features.items():
    if 'floatList' in val_container:
        dtype = "Float"
        data = val_container['floatList'].get('value', [])
    elif 'int64List' in val_container:
        dtype = "Int64"
        data = val_container['int64List'].get('value', [])
    elif 'bytesList' in val_container:
        dtype = "Bytes"
        data = val_container['bytesList'].get('value', [])
    else:
        dtype = "Empty/Unknown"
        data = []

    print(f"Feature: '{key}'")
    print(f"  Type:  {dtype}")
    print(f"  Shape: {len(data)} items (65 height x 65 width x 6 channels)")

    if len(data) == 1:
        print(f"  Value: {data[0]}")
    elif len(data) > 10 and dtype != "Bytes":
        arr = np.array(data)
        print(f"  Preview: {data[:3]} ... (truncated)")
        print(f"  Stats:   Min={arr.min():.4f} | Max={arr.max():.4f} | Mean={arr.mean():.4f}")
    else:
        print(f"  Values: {data}")
    
    print("-" * 20)