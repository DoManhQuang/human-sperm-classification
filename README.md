## HaUI human sperm classification

### Install Lib
```commandline
git clone https://github.com/DoManhQuang/human-sperm-classification.git
cd human-sperm-classification
pip install -r requirement.txt
```

### Train
```commandline
python tools/train.py -v "version-0.0" -ep 10 -bsize 16 -verbose 1 -train "dataset/smids/smids_train.data" -val "dataset/smids/smids_valid.data" -test "dataset/smids/smids_datatest.data" -name "smids-dataset"
```
