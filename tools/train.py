import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import Input
from keras.layers import Dense
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.models import Model
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from core.utils import load_data, set_gpu_limit, get_callbacks_list, write_score_kfold
from core.model import model_classification
from sklearn import preprocessing


# tf.keras.utils.plot_model(model)


# # Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--memory", default=0, type=int, help="set gpu memory limit")
parser.add_argument("-v", "--version", default="version-0.0", help="version running")
parser.add_argument("-rp", "--result_path", default="../runs/results", help="path result ")
parser.add_argument("-tp", "--training_path", default="../runs/training", help="path training model")
parser.add_argument("-ep", "--epochs", default=1, type=int, help="epochs training")
parser.add_argument("-bsize", "--bath_size", default=32, type=int, help="bath size training")
parser.add_argument("-verbose", "--verbose", default=1, type=int, help="verbose training")
parser.add_argument("-train", "--train_data_path", default="../dataset/smids/SMIDS/dataset/smids_train.data", help="data training")
parser.add_argument("-val", "--val_data_path", default="../dataset/smids/SMIDS/dataset/smids_valid.data", help="data val")
parser.add_argument("-test", "--test_data_path", default="../dataset/smids/SMIDS/dataset/smids_datatest.data", help="data test")
parser.add_argument("-name", "--name_model", default="model_ai_name", help="model name")
# parser.add_argument("-cls", "--number_class", default=3, type=int, help="number class")
args = vars(parser.parse_args())

# Set up parameters
version = args["version"]
training_path = args["training_path"]
result_path = args["result_path"]
epochs = args["epochs"]
bath_size = args["bath_size"]
verbose = args["verbose"]
gpu_memory = args["memory"]
train_path = args["train_data_path"]
val_path = args["val_data_path"]
test_path = args["test_data_path"]
model_name = args["name_model"]
# num_classes = args["number_class"]

print("=========Start=========")
if gpu_memory > 0:
    set_gpu_limit(int(gpu_memory))  # set GPU

print("=====loading dataset ...======")
global_dataset_train, global_labels_train = load_data(train_path)
global_dataset_val, global_labels_val = load_data(val_path)
global_dataset_test, global_labels_test = load_data(test_path)

print("TRAIN : ", global_dataset_train.shape, " - ", global_labels_train.shape)
print("VAL : ", global_dataset_val.shape, " - ", global_labels_val.shape)
print("TEST : ", global_dataset_test.shape, " - ", global_labels_test.shape)

print("=======loading dataset done!!=======")
num_classes = len(np.unique(global_labels_train))
ip_shape = global_dataset_train[0].shape
metrics = [
    tfa.metrics.F1Score(num_classes=num_classes, average='weighted')
]

model = model_classification(input_layer=ip_shape, num_class=num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=metrics)
model.summary()
weights_init = model.get_weights()
print("model loading done!!")

# created folder

if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

training_path = os.path.join(training_path, model_name)
if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

training_path = os.path.join(training_path, version)
if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

if not os.path.exists(os.path.join(training_path, 'model-save')):
    os.makedirs(os.path.join(training_path, 'model-save'))
    print("created folder : ", os.path.join(training_path, 'model-save'))


if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)


result_path = os.path.join(result_path, model_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)

# training

file_ckpt_model = "best-weights-training-file-" + model_name + "-" + version + ".h5"
# callback list
callbacks_list, save_list = get_callbacks_list(training_path,
                                               status_tensorboard=True,
                                               status_checkpoint=True,
                                               status_earlystop=True,
                                               file_ckpt=file_ckpt_model,
                                               ckpt_monitor='val_f1_score',
                                               ckpt_mode='max',
                                               early_stop_monitor="val_loss",
                                               early_stop_mode="min",
                                               early_stop_patience=100
                                               )
print("Callbacks List: ", callbacks_list)
print("Save List: ", save_list)

print("===========Training==============")

print("===Labels fit transform ===")
lb = preprocessing.LabelBinarizer()
labels_train_one_hot = lb.fit_transform(global_labels_train)
labels_val_one_hot = lb.fit_transform(global_labels_val)
labels_test_one_hot = lb.fit_transform(global_labels_test)

print("TRAIN : ", labels_train_one_hot.shape)
print("VAL : ", labels_val_one_hot.shape)
print("TEST : ", labels_test_one_hot.shape)
# labels_test_one_hot = tf.keras.utils.to_categorical(global_labels_test, num_classes=num_classes)
# labels_train_one_hot = tf.keras.utils.to_categorical(global_labels_train, num_classes=num_classes)
# labels_val_one_hot = tf.keras.utils.to_categorical(global_labels_val, num_classes=num_classes)

model.set_weights(weights_init)
model_history = model.fit(global_dataset_train, labels_train_one_hot, epochs=epochs, batch_size=bath_size,
                          verbose=verbose, validation_data=(global_dataset_val, labels_val_one_hot),
                          shuffle=True, callbacks=callbacks_list)
print("===========Training Done !!==============")
model_save_file = "model-" + model_name + "-" + version + ".h5"
model.save(os.path.join(training_path, 'model-save', model_save_file), save_format='h5')
print("Save model done!!")

print("testing model.....")

scores = model.evaluate(global_dataset_test, labels_test_one_hot, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
y_predict = model.predict(global_dataset_test)

y_true = np.argmax(labels_test_one_hot, axis=1)
y_target = np.argmax(y_predict, axis=1)

print("save results ......")
file_result = model_name + version + "score.txt"
auc_score = 0
# if num_classes == 2:
#     auc_score = roc_auc_score(y_true, y_target)
# else:
#     auc_score = roc_auc_score(y_true, y_target, multi_class='ovr', labels=np.unique(global_labels_train).shape)

write_score_kfold(path=os.path.join(result_path, file_result),
                  mode_write="a",
                  rows="STT",
                  cols=['F1', 'Acc'])

write_score_kfold(path=os.path.join(result_path, file_result),
                  mode_write="a",
                  rows="results",
                  cols=np.around([f1_score(y_true, y_target, average='weighted'),
                                  accuracy_score(y_true, y_target)], decimals=4))
print("save results done!!")
print("History training loading ...")
cmd = 'tensorboard --logdir "path-tensorboard-logs/"'
print("CMD: ", cmd)
for file_log in save_list:
    print("file_log: ", file_log)
print("============END=============")
