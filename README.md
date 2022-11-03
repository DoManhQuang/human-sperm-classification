## HaUI human sperm classification

### Train
```commandline
python tools/train.py -v "version-0.0" -rp "runs/results" -tp "runs/training" -ep 10 -bsize 16 -verbose 1 -train "dataset/smids/smids_train.data" -val "dataset/smids/smids_valid.data" -test "dataset/smids/smids_datatest.data" -name "smids-dataset"
```
