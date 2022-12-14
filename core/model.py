from keras import Input
from keras.layers import Conv2D, Dropout, GlobalAveragePooling2D
from keras.layers import Dense, BatchNormalization, MaxPooling2D, concatenate
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
import sys
import os
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from core.utils_activation import activation_functions


def block_conv_a(x, filter_cnv_a, filter_cnv_b, filter_cnv_c, activation_block='relu', name="Block_Conv_A"):
    convA = Conv2D(filter_cnv_a, kernel_size=(3, 3), strides=(2, 2), activation=activation_block, padding='same')(x)
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)
    filter_cnv_a = filter_cnv_a / 2
    convA = Conv2D(filter_cnv_a, kernel_size=(3, 3), strides=(1, 1), activation=activation_block, padding='same')(convA)
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)

    convB = Conv2D(filter_cnv_b, kernel_size=(5, 5), strides=(2, 2), activation=activation_block, padding='same')(x)
    convB = Conv2D(filter_cnv_b, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convB)
    filter_cnv_b = filter_cnv_b / 2
    convB = Conv2D(filter_cnv_b, kernel_size=(5, 5), strides=(1, 1), activation=activation_block, padding='same')(convB)
    convB = Conv2D(filter_cnv_b, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convB)

    poolC = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    poolC = Conv2D(filter_cnv_c, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(poolC)
    filter_cnv_c = filter_cnv_c / 2
    poolC = Conv2D(filter_cnv_c, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(poolC)
    poolC = BatchNormalization()(poolC)

    output = concatenate([convA, convB, poolC], name=name)
    return output


def block_conv_b(x, filter_cnv_a, filter_cnv_b, activation_block='relu', name="Block_Conv_B"):
    convA = Conv2D(filter_cnv_a, kernel_size=(5, 5), strides=(2, 2), activation=activation_block, padding='same')(x)
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)
    filter_cnv_a = filter_cnv_a / 2
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)
    convA = BatchNormalization()(convA)

    poolB = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(x)
    poolB = Conv2D(filter_cnv_b, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(poolB)
    filter_cnv_b = filter_cnv_b / 2
    poolB = Conv2D(filter_cnv_b, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(poolB)

    poolC = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(x)

    output = concatenate([convA, poolB, poolC], name=name)
    return output


def block_identity_a(x, filter_cnv_a, filter_cnv_b, activation_block='relu', name="Block_Identity_A"):
    convA = Conv2D(filter_cnv_a, kernel_size=(3, 3), strides=(1, 1), activation=activation_block, padding='same')(x)
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)
    filter_cnv_a = filter_cnv_a / 2
    convA = Conv2D(filter_cnv_a, kernel_size=(3, 3), strides=(1, 1), activation=activation_block, padding='same')(convA)
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)

    convB = Conv2D(filter_cnv_b, kernel_size=(5, 5), strides=(1, 1), activation=activation_block, padding='same')(x)
    convB = Conv2D(filter_cnv_b, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convB)
    filter_cnv_b = filter_cnv_b / 2
    convB = Conv2D(filter_cnv_b, kernel_size=(5, 5), strides=(1, 1), activation=activation_block, padding='same')(convB)
    convB = Conv2D(filter_cnv_b, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convB)
    convB = BatchNormalization()(convB)

    output = concatenate([convA, convB, x], name=name)

    return output


def block_identity_b(x, filter_cnv_a, filter_cnv_b, activation_block='relu', name="Block_Identity_B"):
    convA = Conv2D(filter_cnv_a, kernel_size=(5, 5), strides=(1, 1), activation=activation_block, padding='same')(x)
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)
    filter_cnv_a = filter_cnv_a / 2
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)
    convA = BatchNormalization()(convA)

    convB = Conv2D(filter_cnv_b, kernel_size=(3, 3), strides=(1, 1), activation=activation_block, padding='same')(x)
    filter_cnv_b = filter_cnv_b / 2
    convB = Conv2D(filter_cnv_b, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convB)

    output = concatenate([convA, convB, x], name=name)
    return output


def block_stem(x, filter_cnv_a, activation_block='relu', name="Block_Stem"):
    convA = Conv2D(filter_cnv_a, kernel_size=(7, 7), strides=(1, 1), activation=activation_block, padding='same')(x)
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same')(convA)
    convA = MaxPooling2D((2, 2))(convA)

    filter_cnv_a = filter_cnv_a / 2

    convA = Conv2D(filter_cnv_a, kernel_size=(3, 3), strides=(1, 1), activation=activation_block, padding='same')(convA)
    convA = Conv2D(filter_cnv_a, kernel_size=(1, 1), strides=(1, 1), activation=activation_block, padding='same', name=name)(convA)
    return convA


def created_model_medium(input_layer, activation_block='relu', name="Model_Medium"):
    xS = block_stem(input_layer, filter_cnv_a=256, activation_block=activation_block, name="Block_Stem")
    xS = MaxPooling2D((2, 2), name="MaxPooling2D_S")(xS)

    xA = block_conv_a(xS, filter_cnv_a=128, filter_cnv_b=128, filter_cnv_c=128, activation_block=activation_block, name="Block_Conv_A_1")
    xA = block_identity_a(xA, filter_cnv_a=64, filter_cnv_b=64, activation_block=activation_block, name="Block_Identity_A_1")
    xA = MaxPooling2D((2, 2))(xA)
    xA = block_conv_a(xA, filter_cnv_a=32, filter_cnv_b=32, filter_cnv_c=32, activation_block=activation_block, name="Block_Conv_A_2")
    xA = block_identity_a(xA, filter_cnv_a=8, filter_cnv_b=8, activation_block=activation_block, name="Block_Identity_A_2")

    xB = block_conv_b(xS, filter_cnv_a=128, filter_cnv_b=128, activation_block=activation_block, name="Block_Conv_B_1")
    xB = block_identity_b(xB, filter_cnv_a=64, filter_cnv_b=64, activation_block=activation_block, name="Block_Identity_B_1")
    xB = MaxPooling2D((2, 2))(xB)
    xB = block_conv_b(xB, filter_cnv_a=32, filter_cnv_b=32, activation_block=activation_block, name="Block_Conv_B_2")
    xB = block_identity_b(xB, filter_cnv_a=8, filter_cnv_b=8, activation_block=activation_block, name="Block_Identity_B_2")

    output = concatenate([xA, xB], name=name)
    return output


def created_model_small(input_layer, activation_block='relu', name="Model_small"):
    xS = block_stem(input_layer, filter_cnv_a=256, activation_block=activation_block, name="Block_Stem")
    xS = AveragePooling2D((2, 2), name="AveragePooling2D_1")(xS)

    #     xA = block_conv_a(xS, filter_cnv_a=128, filter_cnv_b=128, filter_cnv_c=128, name="Block_Conv_A_1")
    #     xA = block_identity_a(xA, filter_cnv_a=64, filter_cnv_b=64, name="Block_Identity_A_1")
    #     xA = MaxPooling2D((2, 2))(xA)
    #     xA = block_conv_a(xA, filter_cnv_a=32, filter_cnv_b=32, filter_cnv_c=32, name="Block_Conv_A_2")
    #     xA = block_identity_a(xA, filter_cnv_a=8, filter_cnv_b=8, name="Block_Identity_A_2")

    xB = block_conv_b(xS, filter_cnv_a=32, filter_cnv_b=32,  activation_block=activation_block, name="Block_Conv_B_1")
    xB = block_identity_b(xB, filter_cnv_a=8, filter_cnv_b=8, activation_block=activation_block, name="Block_Identity_B_1" + name)
    #     xB = MaxPooling2D((2, 2))(xB)
    #     xB = block_conv_b(xB, filter_cnv_a=32, filter_cnv_b=32, filter_cnv_c=32, name="Block_Conv_B_2")
    #     xB = block_identity_b(xB, filter_cnv_a=8, filter_cnv_b=8, name="Block_Identity_B_2")

    return xB


def model_classification(input_layer, num_class=2, activation_block='relu', activation_dense='softmax'):
    input_layer = Input(shape=input_layer)
    x = created_model_medium(input_layer, activation_block=activation_functions[activation_block])
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_class, activation=activation_dense)(x)
    return Model(inputs=input_layer, outputs=x)
