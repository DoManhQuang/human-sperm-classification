from keras import backend as K
from keras.layers import ReLU, ELU, LeakyReLU


def seg_relu(x):
    return K.switch(x > 0, x, K.softsign(x))


activation_functions = {
    'relu': ReLU(),
    'ELU': ELU(),
    'LeakyReLU': LeakyReLU(),
    'seg_relu': seg_relu
}
