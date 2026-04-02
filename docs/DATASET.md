# Dataset Capture

Use the live capture script to build a dataset from webcam images.

```bash
python -m ml.pipeline.data_collection.collect_dataset_live
```

Options:
```bash
--num-images 200 --delay-ms 150 --flip
--label <LABEL>
--labels-file <FILE>
```

Captured images are saved to:
```
data/raw/static/<label>/
```

Labels like `space` and `full-stop` are supported.

Examples:
```bash
# Capture a single label and exit
python -m ml.pipeline.data_collection.collect_dataset_live --label space

# Capture labels listed in a file (one per line)
python -m ml.pipeline.data_collection.collect_dataset_live --labels-file labels.txt
```

