import glob, os, sys, tensorflow as tf, numpy as np
from tqdm import tqdm

INPUT_DIR = "/content/drive/MyDrive/Project_WISE/data/ProjectWise_Data/"
INPUT_GLOB = os.path.join(INPUT_DIR, "Anantapur_wellsiting_batch_*.tfrecord.gz")
OUT_TFRECORD = "/content/drive/MyDrive/Project_WISE/data/merged_64x64_wells.tfrecord"

PATCH = 65
OUT_PATCH = 64
BANDS = 6
EXPECTED_FLAT = PATCH * PATCH * BANDS
SKIP_MALFORMED = True

all_files = glob.glob(INPUT_GLOB)
all_files.sort()  # sort alphabetically so batch_0 through batch_7 are in order
files = all_files

if not files:
    print(f"No files found at: {INPUT_GLOB}")
    sys.exit(1)

print(f"Selected the {len(files)} newest files:")
for f in files: print(" -", os.path.basename(f))

def parse_example_bytesproto(raw_bytes):
    ex = tf.train.Example.FromString(raw_bytes)
    feat = ex.features.feature
    if 'patch' not in feat: raise ValueError("Key 'patch' not found in record.")

    fpatch = feat['patch']
    arr = None
    if len(fpatch.float_list.value) > 0: arr = np.array(fpatch.float_list.value, dtype=np.float32)
    if arr is None: raise ValueError("Could not decode 'patch' data.")

    if 'label' not in feat: raise ValueError("Key 'label' not found.")
    flabel = feat['label']
    if len(flabel.float_list.value) > 0: label = int(flabel.float_list.value[0])
    elif len(flabel.int64_list.value) > 0: label = int(flabel.int64_list.value[0])
    else: raise ValueError("Label is empty.")

    return arr, label

def center_crop_arr(arr_flat):
    if arr_flat.size != EXPECTED_FLAT: raise ValueError(f"Size mismatch: {arr_flat.size} vs {EXPECTED_FLAT}")
    img = arr_flat.reshape((PATCH, PATCH, BANDS))
    return img[0:OUT_PATCH, 0:OUT_PATCH, :]

writer = tf.io.TFRecordWriter(OUT_TFRECORD)
total_in, total_written, errors = 0, 0, 0

for f in tqdm(files):
    try:
        ds = tf.data.TFRecordDataset([f], compression_type='GZIP')
        for raw in ds.as_numpy_iterator():
            total_in += 1
            try:
                arr_flat, label = parse_example_bytesproto(raw)
                cropped = center_crop_arr(arr_flat)
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
print(f"\nDone. Input records: {total_in}, Written: {total_written}, Errors: {errors}")