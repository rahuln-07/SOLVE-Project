import glob, os, sys, tensorflow as tf, numpy as np
from tqdm import tqdm

INPUT_DIR = os.path.join("..", "data", "drive-download-20260203T102551Z-3-001")
INPUT_GLOB = os.path.join(INPUT_DIR, "*.tfrecord.gz")

OUT_TFRECORD = os.path.join("..", "data", "merged_64x64.tfrecord")
PATCH = 65                # Input size
OUT_PATCH = 64            # Output size
BANDS = 6                 # Channels
EXPECTED_FLAT = PATCH * PATCH * BANDS
SKIP_MALFORMED = True

files = sorted(glob.glob(INPUT_GLOB))
if not files:
    print(f"No files found at: {INPUT_GLOB}")
    sys.exit(1)

print(f"Found {len(files)} files. Merging...")

def parse_example_bytesproto(raw_bytes):
    ex = tf.train.Example.FromString(raw_bytes)
    feat = ex.features.feature
    
    # --- PATCH EXTRACTION ---
    # We know your key is 'patch' from the previous inspection
    if 'patch' not in feat:
        raise ValueError("Key 'patch' not found in record.")
        
    fpatch = feat['patch']
    arr = None
    
    # Your data uses float_list
    if len(fpatch.float_list.value) > 0:
        arr = np.array(fpatch.float_list.value, dtype=np.float32)
    
    if arr is None:
        raise ValueError("Could not decode 'patch' data.")

    # --- LABEL EXTRACTION ---
    # We know your key is 'label'
    if 'label' not in feat:
        raise ValueError("Key 'label' not found.")
        
    flabel = feat['label']
    
    # Handle float labels (convert 1.0 -> 1)
    if len(flabel.float_list.value) > 0:
        label = int(flabel.float_list.value[0])
    elif len(flabel.int64_list.value) > 0:
        label = int(flabel.int64_list.value[0])
    else:
        raise ValueError("Label is empty.")

    return arr, label

def center_crop_arr(arr_flat):
    if arr_flat.size != EXPECTED_FLAT:
        raise ValueError(f"Size mismatch: {arr_flat.size} vs {EXPECTED_FLAT}")
        
    # Reshape to 3D Cube
    img = arr_flat.reshape((PATCH, PATCH, BANDS))
    
    # Crop 65 -> 64 (Top-Left alignment)
    # This drops the last row and last column
    cropped = img[0:OUT_PATCH, 0:OUT_PATCH, :]
    
    return cropped

# --- WRITER LOOP ---
writer = tf.io.TFRecordWriter(OUT_TFRECORD)
total_in = 0
total_written = 0
errors = 0

# Use tqdm for a progress bar
for f in tqdm(files):
    # 2. FIXED: Added compression_type='GZIP'
    try:
        ds = tf.data.TFRecordDataset([f], compression_type='GZIP')
        
        for raw in ds.as_numpy_iterator():
            total_in += 1
            try:
                arr_flat, label = parse_example_bytesproto(raw)
                cropped = center_crop_arr(arr_flat)
                
                # Serialize to Bytes (efficient storage)
                img_bytes = cropped.tobytes()
                
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[OUT_PATCH])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[OUT_PATCH])),
                    'bands': tf.train.Feature(int64_list=tf.train.Int64List(value=[BANDS]))
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                total_written += 1
                
            except Exception as e:
                errors += 1
                if not SKIP_MALFORMED: raise e
                
    except Exception as e:
        print(f"Error reading file {f}: {e}")

writer.close()
print(f"Done. Input records: {total_in}, Written: {total_written}, Errors: {errors}")
print("Merged TFRecord written to:", OUT_TFRECORD)