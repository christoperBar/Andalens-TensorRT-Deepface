import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
import gdown
from pathlib import Path
import cv2

from typing import Union, Any, Optional, Dict, Tuple, List
import numpy as np
from retinafacelib import preprocess
from retinafacelib import postprocess
from tensorflow.keras.models import Model

# Step 1: Recreate the EXACT model architecture from the codebase
def build_retinaface_model():
    """
    This is the EXACT same architecture from retinaface_model.py
    Every layer name, parameter, and connection must match perfectly
    """
    
    # Import the correct Keras modules based on TensorFlow version
    tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
    
    if tf_version == 1:
        from keras.models import Model
        from keras.layers import (
            Input, BatchNormalization, ZeroPadding2D, Conv2D, ReLU,
            MaxPool2D, Add, UpSampling2D, concatenate, Softmax,
        )
    else:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, BatchNormalization, ZeroPadding2D, Conv2D, ReLU,
            MaxPool2D, Add, UpSampling2D, concatenate, Softmax,
        )
    
    # START: Copy the EXACT architecture from lines 99-1516
    data = Input(dtype=tf.float32, shape=(None, None, 3), name="data")

    bn_data = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn_data", trainable=False)(
        data
    )

    conv0_pad = ZeroPadding2D(padding=tuple([3, 3]))(bn_data)

    conv0 = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        name="conv0",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(conv0_pad)

    bn0 = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn0", trainable=False)(conv0)

    relu0 = ReLU(name="relu0")(bn0)

    pooling0_pad = ZeroPadding2D(padding=tuple([1, 1]))(relu0)

    pooling0 = MaxPool2D((3, 3), (2, 2), padding="valid", name="pooling0")(pooling0_pad)

    stage1_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn1", trainable=False
    )(pooling0)

    stage1_unit1_relu1 = ReLU(name="stage1_unit1_relu1")(stage1_unit1_bn1)

    stage1_unit1_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu1)

    stage1_unit1_sc = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit1_sc",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu1)

    stage1_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn2", trainable=False
    )(stage1_unit1_conv1)

    stage1_unit1_relu2 = ReLU(name="stage1_unit1_relu2")(stage1_unit1_bn2)

    stage1_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit1_relu2)

    stage1_unit1_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit1_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_conv2_pad)

    stage1_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn3", trainable=False
    )(stage1_unit1_conv2)

    stage1_unit1_relu3 = ReLU(name="stage1_unit1_relu3")(stage1_unit1_bn3)

    stage1_unit1_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu3)

    plus0_v1 = Add()([stage1_unit1_conv3, stage1_unit1_sc])

    stage1_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn1", trainable=False
    )(plus0_v1)

    stage1_unit2_relu1 = ReLU(name="stage1_unit2_relu1")(stage1_unit2_bn1)

    stage1_unit2_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_relu1)

    stage1_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn2", trainable=False
    )(stage1_unit2_conv1)

    stage1_unit2_relu2 = ReLU(name="stage1_unit2_relu2")(stage1_unit2_bn2)

    stage1_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit2_relu2)

    stage1_unit2_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_conv2_pad)

    stage1_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn3", trainable=False
    )(stage1_unit2_conv2)

    stage1_unit2_relu3 = ReLU(name="stage1_unit2_relu3")(stage1_unit2_bn3)

    stage1_unit2_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_relu3)

    plus1_v2 = Add()([stage1_unit2_conv3, plus0_v1])

    stage1_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn1", trainable=False
    )(plus1_v2)

    stage1_unit3_relu1 = ReLU(name="stage1_unit3_relu1")(stage1_unit3_bn1)

    stage1_unit3_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_relu1)

    stage1_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn2", trainable=False
    )(stage1_unit3_conv1)

    stage1_unit3_relu2 = ReLU(name="stage1_unit3_relu2")(stage1_unit3_bn2)

    stage1_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit3_relu2)

    stage1_unit3_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_conv2_pad)

    stage1_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn3", trainable=False
    )(stage1_unit3_conv2)

    stage1_unit3_relu3 = ReLU(name="stage1_unit3_relu3")(stage1_unit3_bn3)

    stage1_unit3_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_relu3)

    plus2 = Add()([stage1_unit3_conv3, plus1_v2])

    stage2_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn1", trainable=False
    )(plus2)

    stage2_unit1_relu1 = ReLU(name="stage2_unit1_relu1")(stage2_unit1_bn1)

    stage2_unit1_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu1)

    stage2_unit1_sc = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu1)

    stage2_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn2", trainable=False
    )(stage2_unit1_conv1)

    stage2_unit1_relu2 = ReLU(name="stage2_unit1_relu2")(stage2_unit1_bn2)

    stage2_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit1_relu2)

    stage2_unit1_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_conv2_pad)

    stage2_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn3", trainable=False
    )(stage2_unit1_conv2)

    stage2_unit1_relu3 = ReLU(name="stage2_unit1_relu3")(stage2_unit1_bn3)

    stage2_unit1_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu3)

    plus3 = Add()([stage2_unit1_conv3, stage2_unit1_sc])

    stage2_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn1", trainable=False
    )(plus3)

    stage2_unit2_relu1 = ReLU(name="stage2_unit2_relu1")(stage2_unit2_bn1)

    stage2_unit2_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_relu1)

    stage2_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn2", trainable=False
    )(stage2_unit2_conv1)

    stage2_unit2_relu2 = ReLU(name="stage2_unit2_relu2")(stage2_unit2_bn2)

    stage2_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit2_relu2)

    stage2_unit2_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_conv2_pad)

    stage2_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn3", trainable=False
    )(stage2_unit2_conv2)

    stage2_unit2_relu3 = ReLU(name="stage2_unit2_relu3")(stage2_unit2_bn3)

    stage2_unit2_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_relu3)

    plus4 = Add()([stage2_unit2_conv3, plus3])

    stage2_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn1", trainable=False
    )(plus4)

    stage2_unit3_relu1 = ReLU(name="stage2_unit3_relu1")(stage2_unit3_bn1)

    stage2_unit3_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_relu1)

    stage2_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn2", trainable=False
    )(stage2_unit3_conv1)

    stage2_unit3_relu2 = ReLU(name="stage2_unit3_relu2")(stage2_unit3_bn2)

    stage2_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit3_relu2)

    stage2_unit3_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_conv2_pad)

    stage2_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn3", trainable=False
    )(stage2_unit3_conv2)

    stage2_unit3_relu3 = ReLU(name="stage2_unit3_relu3")(stage2_unit3_bn3)

    stage2_unit3_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_relu3)

    plus5 = Add()([stage2_unit3_conv3, plus4])

    stage2_unit4_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn1", trainable=False
    )(plus5)

    stage2_unit4_relu1 = ReLU(name="stage2_unit4_relu1")(stage2_unit4_bn1)

    stage2_unit4_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit4_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_relu1)

    stage2_unit4_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn2", trainable=False
    )(stage2_unit4_conv1)

    stage2_unit4_relu2 = ReLU(name="stage2_unit4_relu2")(stage2_unit4_bn2)

    stage2_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit4_relu2)

    stage2_unit4_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit4_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_conv2_pad)

    stage2_unit4_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn3", trainable=False
    )(stage2_unit4_conv2)

    stage2_unit4_relu3 = ReLU(name="stage2_unit4_relu3")(stage2_unit4_bn3)

    stage2_unit4_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit4_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_relu3)

    plus6 = Add()([stage2_unit4_conv3, plus5])

    stage3_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn1", trainable=False
    )(plus6)

    stage3_unit1_relu1 = ReLU(name="stage3_unit1_relu1")(stage3_unit1_bn1)

    stage3_unit1_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu1)

    stage3_unit1_sc = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu1)

    stage3_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn2", trainable=False
    )(stage3_unit1_conv1)

    stage3_unit1_relu2 = ReLU(name="stage3_unit1_relu2")(stage3_unit1_bn2)

    stage3_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit1_relu2)

    stage3_unit1_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_conv2_pad)

    ssh_m1_red_conv = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_m1_red_conv",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(stage3_unit1_relu2)

    stage3_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn3", trainable=False
    )(stage3_unit1_conv2)

    ssh_m1_red_conv_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_red_conv_bn", trainable=False
    )(ssh_m1_red_conv)

    stage3_unit1_relu3 = ReLU(name="stage3_unit1_relu3")(stage3_unit1_bn3)

    ssh_m1_red_conv_relu = ReLU(name="ssh_m1_red_conv_relu")(ssh_m1_red_conv_bn)

    stage3_unit1_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu3)

    plus7 = Add()([stage3_unit1_conv3, stage3_unit1_sc])

    stage3_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn1", trainable=False
    )(plus7)

    stage3_unit2_relu1 = ReLU(name="stage3_unit2_relu1")(stage3_unit2_bn1)

    stage3_unit2_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_relu1)

    stage3_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn2", trainable=False
    )(stage3_unit2_conv1)

    stage3_unit2_relu2 = ReLU(name="stage3_unit2_relu2")(stage3_unit2_bn2)

    stage3_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit2_relu2)

    stage3_unit2_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_conv2_pad)

    stage3_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn3", trainable=False
    )(stage3_unit2_conv2)

    stage3_unit2_relu3 = ReLU(name="stage3_unit2_relu3")(stage3_unit2_bn3)

    stage3_unit2_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_relu3)

    plus8 = Add()([stage3_unit2_conv3, plus7])

    stage3_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn1", trainable=False
    )(plus8)

    stage3_unit3_relu1 = ReLU(name="stage3_unit3_relu1")(stage3_unit3_bn1)

    stage3_unit3_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_relu1)

    stage3_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn2", trainable=False
    )(stage3_unit3_conv1)

    stage3_unit3_relu2 = ReLU(name="stage3_unit3_relu2")(stage3_unit3_bn2)

    stage3_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit3_relu2)

    stage3_unit3_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_conv2_pad)

    stage3_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn3", trainable=False
    )(stage3_unit3_conv2)

    stage3_unit3_relu3 = ReLU(name="stage3_unit3_relu3")(stage3_unit3_bn3)

    stage3_unit3_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_relu3)

    plus9 = Add()([stage3_unit3_conv3, plus8])

    stage3_unit4_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn1", trainable=False
    )(plus9)

    stage3_unit4_relu1 = ReLU(name="stage3_unit4_relu1")(stage3_unit4_bn1)

    stage3_unit4_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit4_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_relu1)

    stage3_unit4_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn2", trainable=False
    )(stage3_unit4_conv1)

    stage3_unit4_relu2 = ReLU(name="stage3_unit4_relu2")(stage3_unit4_bn2)

    stage3_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit4_relu2)

    stage3_unit4_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit4_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_conv2_pad)

    stage3_unit4_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn3", trainable=False
    )(stage3_unit4_conv2)

    stage3_unit4_relu3 = ReLU(name="stage3_unit4_relu3")(stage3_unit4_bn3)

    stage3_unit4_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit4_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_relu3)

    plus10 = Add()([stage3_unit4_conv3, plus9])

    stage3_unit5_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn1", trainable=False
    )(plus10)

    stage3_unit5_relu1 = ReLU(name="stage3_unit5_relu1")(stage3_unit5_bn1)

    stage3_unit5_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit5_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_relu1)

    stage3_unit5_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn2", trainable=False
    )(stage3_unit5_conv1)

    stage3_unit5_relu2 = ReLU(name="stage3_unit5_relu2")(stage3_unit5_bn2)

    stage3_unit5_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit5_relu2)

    stage3_unit5_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit5_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_conv2_pad)

    stage3_unit5_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn3", trainable=False
    )(stage3_unit5_conv2)

    stage3_unit5_relu3 = ReLU(name="stage3_unit5_relu3")(stage3_unit5_bn3)

    stage3_unit5_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit5_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_relu3)

    plus11 = Add()([stage3_unit5_conv3, plus10])

    stage3_unit6_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn1", trainable=False
    )(plus11)

    stage3_unit6_relu1 = ReLU(name="stage3_unit6_relu1")(stage3_unit6_bn1)

    stage3_unit6_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit6_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_relu1)

    stage3_unit6_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn2", trainable=False
    )(stage3_unit6_conv1)

    stage3_unit6_relu2 = ReLU(name="stage3_unit6_relu2")(stage3_unit6_bn2)

    stage3_unit6_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit6_relu2)

    stage3_unit6_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit6_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_conv2_pad)

    stage3_unit6_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn3", trainable=False
    )(stage3_unit6_conv2)

    stage3_unit6_relu3 = ReLU(name="stage3_unit6_relu3")(stage3_unit6_bn3)

    stage3_unit6_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit6_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_relu3)

    plus12 = Add()([stage3_unit6_conv3, plus11])

    stage4_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn1", trainable=False
    )(plus12)

    stage4_unit1_relu1 = ReLU(name="stage4_unit1_relu1")(stage4_unit1_bn1)

    stage4_unit1_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu1)

    stage4_unit1_sc = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu1)

    stage4_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn2", trainable=False
    )(stage4_unit1_conv1)

    stage4_unit1_relu2 = ReLU(name="stage4_unit1_relu2")(stage4_unit1_bn2)

    stage4_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit1_relu2)

    stage4_unit1_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_conv2_pad)

    ssh_c2_lateral = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_c2_lateral",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(stage4_unit1_relu2)

    stage4_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn3", trainable=False
    )(stage4_unit1_conv2)

    ssh_c2_lateral_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c2_lateral_bn", trainable=False
    )(ssh_c2_lateral)

    stage4_unit1_relu3 = ReLU(name="stage4_unit1_relu3")(stage4_unit1_bn3)

    ssh_c2_lateral_relu = ReLU(name="ssh_c2_lateral_relu")(ssh_c2_lateral_bn)

    stage4_unit1_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu3)

    plus13 = Add()([stage4_unit1_conv3, stage4_unit1_sc])

    stage4_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn1", trainable=False
    )(plus13)

    stage4_unit2_relu1 = ReLU(name="stage4_unit2_relu1")(stage4_unit2_bn1)

    stage4_unit2_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_relu1)

    stage4_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn2", trainable=False
    )(stage4_unit2_conv1)

    stage4_unit2_relu2 = ReLU(name="stage4_unit2_relu2")(stage4_unit2_bn2)

    stage4_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit2_relu2)

    stage4_unit2_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_conv2_pad)

    stage4_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn3", trainable=False
    )(stage4_unit2_conv2)

    stage4_unit2_relu3 = ReLU(name="stage4_unit2_relu3")(stage4_unit2_bn3)

    stage4_unit2_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_relu3)

    plus14 = Add()([stage4_unit2_conv3, plus13])

    stage4_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn1", trainable=False
    )(plus14)

    stage4_unit3_relu1 = ReLU(name="stage4_unit3_relu1")(stage4_unit3_bn1)

    stage4_unit3_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_relu1)

    stage4_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn2", trainable=False
    )(stage4_unit3_conv1)

    stage4_unit3_relu2 = ReLU(name="stage4_unit3_relu2")(stage4_unit3_bn2)

    stage4_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit3_relu2)

    stage4_unit3_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_conv2_pad)

    stage4_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn3", trainable=False
    )(stage4_unit3_conv2)

    stage4_unit3_relu3 = ReLU(name="stage4_unit3_relu3")(stage4_unit3_bn3)

    stage4_unit3_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_relu3)

    plus15 = Add()([stage4_unit3_conv3, plus14])

    bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn1", trainable=False)(plus15)

    relu1 = ReLU(name="relu1")(bn1)

    ssh_c3_lateral = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_c3_lateral",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(relu1)

    ssh_c3_lateral_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c3_lateral_bn", trainable=False
    )(ssh_c3_lateral)

    ssh_c3_lateral_relu = ReLU(name="ssh_c3_lateral_relu")(ssh_c3_lateral_bn)

    ssh_m3_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)

    ssh_m3_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m3_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_conv1_pad)

    ssh_m3_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)

    ssh_m3_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv1_pad)

    ssh_c3_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_c3_up")(
        ssh_c3_lateral_relu
    )

    ssh_m3_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_conv1_bn", trainable=False
    )(ssh_m3_det_conv1)

    ssh_m3_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv1_bn", trainable=False
    )(ssh_m3_det_context_conv1)

    x1_shape = tf.shape(ssh_c3_up)
    x2_shape = tf.shape(ssh_c2_lateral_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop0 = tf.slice(ssh_c3_up, offsets, size, "crop0")

    ssh_m3_det_context_conv1_relu = ReLU(name="ssh_m3_det_context_conv1_relu")(
        ssh_m3_det_context_conv1_bn
    )

    plus0_v2 = Add()([ssh_c2_lateral_relu, crop0])

    ssh_m3_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv1_relu
    )

    ssh_m3_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv2_pad)

    ssh_m3_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv1_relu
    )

    ssh_m3_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv3_1_pad)

    ssh_c2_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus0_v2)

    ssh_c2_aggr = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_c2_aggr",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_c2_aggr_pad)

    ssh_m3_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv2_bn", trainable=False
    )(ssh_m3_det_context_conv2)

    ssh_m3_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv3_1_bn", trainable=False
    )(ssh_m3_det_context_conv3_1)

    ssh_c2_aggr_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c2_aggr_bn", trainable=False
    )(ssh_c2_aggr)

    ssh_m3_det_context_conv3_1_relu = ReLU(name="ssh_m3_det_context_conv3_1_relu")(
        ssh_m3_det_context_conv3_1_bn
    )

    ssh_c2_aggr_relu = ReLU(name="ssh_c2_aggr_relu")(ssh_c2_aggr_bn)

    ssh_m3_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv3_1_relu
    )

    ssh_m3_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv3_2_pad)

    ssh_m2_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)

    ssh_m2_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m2_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_conv1_pad)

    ssh_m2_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)

    ssh_m2_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv1_pad)

    ssh_m2_red_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_m2_red_up")(
        ssh_c2_aggr_relu
    )

    ssh_m3_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv3_2_bn", trainable=False
    )(ssh_m3_det_context_conv3_2)

    ssh_m2_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_conv1_bn", trainable=False
    )(ssh_m2_det_conv1)

    ssh_m2_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv1_bn", trainable=False
    )(ssh_m2_det_context_conv1)

    x1_shape = tf.shape(ssh_m2_red_up)
    x2_shape = tf.shape(ssh_m1_red_conv_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop1 = tf.slice(ssh_m2_red_up, offsets, size, "crop1")

    ssh_m3_det_concat = concatenate(
        [ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn],
        3,
        name="ssh_m3_det_concat",
    )

    ssh_m2_det_context_conv1_relu = ReLU(name="ssh_m2_det_context_conv1_relu")(
        ssh_m2_det_context_conv1_bn
    )

    plus1_v1 = Add()([ssh_m1_red_conv_relu, crop1])

    ssh_m3_det_concat_relu = ReLU(name="ssh_m3_det_concat_relu")(ssh_m3_det_concat)

    ssh_m2_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv1_relu
    )

    ssh_m2_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv2_pad)

    ssh_m2_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv1_relu
    )

    ssh_m2_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv3_1_pad)

    ssh_c1_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus1_v1)

    ssh_c1_aggr = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_c1_aggr",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_c1_aggr_pad)

    face_rpn_cls_score_stride32 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride32[:, :, :, 0], face_rpn_cls_score_stride32[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride32[:, :, :, 2], face_rpn_cls_score_stride32[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride32 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride32"
    )

    face_rpn_bbox_pred_stride32 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    face_rpn_landmark_pred_stride32 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    ssh_m2_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv2_bn", trainable=False
    )(ssh_m2_det_context_conv2)

    ssh_m2_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv3_1_bn", trainable=False
    )(ssh_m2_det_context_conv3_1)

    ssh_c1_aggr_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c1_aggr_bn", trainable=False
    )(ssh_c1_aggr)

    ssh_m2_det_context_conv3_1_relu = ReLU(name="ssh_m2_det_context_conv3_1_relu")(
        ssh_m2_det_context_conv3_1_bn
    )

    ssh_c1_aggr_relu = ReLU(name="ssh_c1_aggr_relu")(ssh_c1_aggr_bn)

    face_rpn_cls_prob_stride32 = Softmax(name="face_rpn_cls_prob_stride32")(
        face_rpn_cls_score_reshape_stride32
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride32)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride32[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride32[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride32[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride32[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride32 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride32"
    )

    ssh_m2_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv3_1_relu
    )

    ssh_m2_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv3_2_pad)

    ssh_m1_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)

    ssh_m1_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m1_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_conv1_pad)

    ssh_m1_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)

    ssh_m1_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv1_pad)

    ssh_m2_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv3_2_bn", trainable=False
    )(ssh_m2_det_context_conv3_2)

    ssh_m1_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_conv1_bn", trainable=False
    )(ssh_m1_det_conv1)

    ssh_m1_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv1_bn", trainable=False
    )(ssh_m1_det_context_conv1)

    ssh_m2_det_concat = concatenate(
        [ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn],
        3,
        name="ssh_m2_det_concat",
    )

    ssh_m1_det_context_conv1_relu = ReLU(name="ssh_m1_det_context_conv1_relu")(
        ssh_m1_det_context_conv1_bn
    )

    ssh_m2_det_concat_relu = ReLU(name="ssh_m2_det_concat_relu")(ssh_m2_det_concat)

    ssh_m1_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv1_relu
    )

    ssh_m1_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv2_pad)

    ssh_m1_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv1_relu
    )

    ssh_m1_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv3_1_pad)

    face_rpn_cls_score_stride16 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride16[:, :, :, 0], face_rpn_cls_score_stride16[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride16[:, :, :, 2], face_rpn_cls_score_stride16[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride16 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride16"
    )

    face_rpn_bbox_pred_stride16 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    face_rpn_landmark_pred_stride16 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    ssh_m1_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv2_bn", trainable=False
    )(ssh_m1_det_context_conv2)

    ssh_m1_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv3_1_bn", trainable=False
    )(ssh_m1_det_context_conv3_1)

    ssh_m1_det_context_conv3_1_relu = ReLU(name="ssh_m1_det_context_conv3_1_relu")(
        ssh_m1_det_context_conv3_1_bn
    )

    face_rpn_cls_prob_stride16 = Softmax(name="face_rpn_cls_prob_stride16")(
        face_rpn_cls_score_reshape_stride16
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride16)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride16[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride16[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride16[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride16[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride16 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride16"
    )

    ssh_m1_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv3_1_relu
    )

    ssh_m1_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv3_2_pad)

    ssh_m1_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv3_2_bn", trainable=False
    )(ssh_m1_det_context_conv3_2)

    ssh_m1_det_concat = concatenate(
        [ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn],
        3,
        name="ssh_m1_det_concat",
    )

    ssh_m1_det_concat_relu = ReLU(name="ssh_m1_det_concat_relu")(ssh_m1_det_concat)
    face_rpn_cls_score_stride8 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride8[:, :, :, 0], face_rpn_cls_score_stride8[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride8[:, :, :, 2], face_rpn_cls_score_stride8[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride8 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride8"
    )

    face_rpn_bbox_pred_stride8 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    face_rpn_landmark_pred_stride8 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    face_rpn_cls_prob_stride8 = Softmax(name="face_rpn_cls_prob_stride8")(
        face_rpn_cls_score_reshape_stride8
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride8)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride8[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride8[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride8[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride8[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride8 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride8"
    )

    model = Model(
        inputs=data,
        outputs=[
            face_rpn_cls_prob_reshape_stride32,
            face_rpn_bbox_pred_stride32,
            face_rpn_landmark_pred_stride32,
            face_rpn_cls_prob_reshape_stride16,
            face_rpn_bbox_pred_stride16,
            face_rpn_landmark_pred_stride16,
            face_rpn_cls_prob_reshape_stride8,
            face_rpn_bbox_pred_stride8,
            face_rpn_landmark_pred_stride8,
        ],
    )
    return model

# Step 2: Download weights function
def download_retinaface_weights():
    home = str(Path.home())
    weights_dir = os.path.join(home, ".deepface", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    weights_path = os.path.join(weights_dir, "retinafacelib/retinaface.h5")
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"
    
    if not os.path.exists(weights_path):
        print("Downloading RetinaFace weights...")
        gdown.download(url, weights_path, quiet=False)
        print(f"Weights downloaded to: {weights_path}")
    
    return weights_path

# Step 3: Load weights into model
def load_weights_into_model(model, weights_path):
    """Load the downloaded .h5 weights into the model"""
    if not os.path.exists(weights_path):
        raise ValueError(f"Weights file not found: {weights_path}")
    
    print("Loading weights into model...")
    model.load_weights(weights_path)
    print("Weights loaded successfully!")
    
    return model

# Step 4: Complete workflow
def create_retinaface_model():
    """Complete workflow to create RetinaFace model with pre-trained weights"""
    
    # 1. Download weights
    # weights_path = download_retinaface_weights()
    weights_path = "retinafacelib/retinaface.h5"
    # weights_path = "/content/drive/MyDrive/Colab_Notebooks/libs/retinafacelib/retinaface.h5"
    
    # 2. Build model architecture
    print("Building RetinaFace model architecture...")
    model = build_retinaface_model()
    
    # 3. Load pre-trained weights
    model = load_weights_into_model(model, weights_path)
    
    # 4. Convert to TensorFlow function for optimization (as in original code)
    model_func = tf.function(
        model,
        input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),)
    )
    
    return model_func

def detect_faces(
    img_path: Union[str, np.ndarray],
    threshold: float = 0.9,
    model: Optional[Model] = None,
    allow_upscaling: bool = True,
) -> Dict[str, Any]:
    """
    Detect the facial area for a given image
    Args:
        img_path (str or numpy array): given image
        threshold (float): threshold for detection
        model (Model): pre-trained model can be given
        allow_upscaling (bool): allowing up-scaling
    Returns:
        detected faces as:
        {
            "face_1": {
                "score": 0.9993440508842468,
                "facial_area": [155, 81, 434, 443],
                "landmarks": {
                    "right_eye": [257.82974, 209.64787],
                    "left_eye": [374.93427, 251.78687],
                    "nose": [303.4773, 299.91144],
                    "mouth_right": [228.37329, 338.73193],
                    "mouth_left": [320.21982, 374.58798]
                }
            }
        }
    """
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    resp = {}
    img = preprocess.get_image(img_path)

    # ---------------------------

    if model is None:
        model = create_retinaface_model()

    # ---------------------------

    nms_threshold = 0.4
    decay4 = 0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        "stride32": np.array(
            [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32
        ),
        "stride16": np.array(
            [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32
        ),
        "stride8": np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
    }

    _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    # ---------------------------

    proposals_list = []
    scores_list = []
    landmarks_list = []
    im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
    net_out = model(im_tensor)
    net_out = [elt.numpy() for elt in net_out]
    sym_idx = 0

    for _, s in enumerate(_feat_stride_fpn):
        # _key = f"stride{s}"
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors[f"stride{s}"] :]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors[f"stride{s}"]
        K = height * width
        anchors_fpn = _anchors_fpn[f"stride{s}"]
        anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)

        proposals = postprocess.clip_boxes(proposals, im_info[:2])

        if s == 4 and decay4 < 1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3] // A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)

    if proposals.shape[0] == 0:
        return resp

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)

    # nms = cpu_nms_wrapper(nms_threshold)
    # keep = nms(pre_det)
    keep = postprocess.cpu_nms(pre_det, nms_threshold)

    det = np.hstack((pre_det, proposals[:, 4:]))
    det = det[keep, :]
    landmarks = landmarks[keep]

    for idx, face in enumerate(det):
        label = "face_" + str(idx + 1)
        resp[label] = {}
        resp[label]["score"] = face[4]

        resp[label]["facial_area"] = list(face[0:4].astype(int))

        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])

    return resp

def detect_faces_batch(
    imgs_bgr: List[np.ndarray],
    orig_shapes: List[tuple],
    target_sizes: List[tuple],
    threshold: float = 0.9,
    model: Optional[Any] = None,
    allow_upscaling: bool = True
) -> List[List[Dict[str, Any]]]:
    """
    Deteksi wajah dari batch gambar yang sudah di-letterbox.
    Mengembalikan koordinat wajah dalam ukuran asli gambar.
    """
    from retina import preprocess, postprocess, create_retinaface_model

    if model is None:
        model = create_retinaface_model()

    nms_threshold = 0.4
    decay4 = 0.5
    _feat_stride_fpn = [32, 16, 8]
    _anchors_fpn = {
        "stride32": np.array(
            [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32
        ),
        "stride16": np.array(
            [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32
        ),
        "stride8": np.array(
            [[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32
        ),
    }
    _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    # 1. Preprocess semua letterbox image jadi batch tensor
    im_tensors = []
    im_infos = []
    im_scales = []
    for img in imgs_bgr:
        im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
        im_tensors.append(im_tensor)
        im_infos.append(im_info)
        im_scales.append(im_scale)

    batch_tensor = np.vstack(im_tensors)

    # 2. Forward pass sekali jalan
    net_out = model(batch_tensor)
    net_out = [elt.numpy() for elt in net_out]

    # 3. Parsing hasil per gambar
    all_results = []
    sym_stride_count = len(_feat_stride_fpn) * 3
    batch_size = batch_tensor.shape[0]

    for b in range( batch_size):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        sym_idx = 0

        for _, s in enumerate(_feat_stride_fpn):
            scores = net_out[sym_idx][b]
            scores = scores[:, :, _num_anchors[f"stride{s}"] :]
            bbox_deltas = net_out[sym_idx + 1][b]
            height, width = bbox_deltas.shape[0], bbox_deltas.shape[1]

            A = _num_anchors[f"stride{s}"]
            K = height * width
            anchors_fpn = _anchors_fpn[f"stride{s}"]
            anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.reshape((-1, 1))

            bbox_stds = [1.0, 1.0, 1.0, 1.0]
            bbox_pred_len = bbox_deltas.shape[2] // A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] *= bbox_stds[0]
            bbox_deltas[:, 1::4] *= bbox_stds[1]
            bbox_deltas[:, 2::4] *= bbox_stds[2]
            bbox_deltas[:, 3::4] *= bbox_stds[3]
            proposals = postprocess.bbox_pred(anchors, bbox_deltas)
            proposals = postprocess.clip_boxes(proposals, im_infos[b][:2])

            if s == 4 and decay4 < 1.0:
                scores *= decay4

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:, 0:4] /= im_scales[b]
            proposals_list.append(proposals)
            scores_list.append(scores)

            landmark_deltas = net_out[sym_idx + 2][b]
            landmark_pred_len = landmark_deltas.shape[2] // A
            landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
            landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]
            landmarks[:, :, 0:2] /= im_scales[b]
            landmarks_list.append(landmarks)
            sym_idx += 3

        # Gabung semua hasil level FPN
        proposals = np.vstack(proposals_list)
        if proposals.shape[0] == 0:
            all_results.append([])
            continue

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]
        landmarks = np.vstack(landmarks_list)[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
        keep = postprocess.cpu_nms(pre_det, nms_threshold)

        det = np.hstack((pre_det, proposals[:, 4:]))[keep, :]
        landmarks = landmarks[keep]

        # 4. Konversi koordinat letterbox → ukuran asli gambar
        orig_h, orig_w = orig_shapes[b]
        target_w, target_h = target_sizes[b]
        scale = min(target_w / orig_w, target_h / orig_h)
        pad_w = (target_w - int(orig_w * scale)) // 2
        pad_h = (target_h - int(orig_h * scale)) // 2

        faces_out = []
        for idx, face in enumerate(det):
            # Bounding box
            x1 = (face[0] - pad_w) / scale
            y1 = (face[1] - pad_h) / scale
            x2 = (face[2] - pad_w) / scale
            y2 = (face[3] - pad_h) / scale

            # Landmarks
            lm = landmarks[idx]
            lm_unletterboxed = []
            for px, py in lm:
                lm_unletterboxed.append([
                    (px - pad_w) / scale,
                    (py - pad_h) / scale
                ])

            faces_out.append({
                "score": float(face[4]),
                "facial_area": [int(x1), int(y1), int(x2), int(y2)],
                "landmarks": {
                    "right_eye": list(map(float, lm_unletterboxed[0])),
                    "left_eye": list(map(float, lm_unletterboxed[1])),
                    "nose": list(map(float, lm_unletterboxed[2])),
                    "mouth_right": list(map(float, lm_unletterboxed[3])),
                    "mouth_left": list(map(float, lm_unletterboxed[4])),
                }
            })

        all_results.append(faces_out)

    return all_results




def extract_faces(
    img_path: Union[str, np.ndarray],
    threshold: float = 0.9,
    model: Optional[Model] = None,
    align: bool = False,
    allow_upscaling: bool = True,
    expand_face_area: int = 0,
    target_size: Optional[Tuple[int, int]] = None,
    min_max_norm: bool = True,
) -> List[np.ndarray]:
    """
    Extract detected and aligned faces
    Args:
        img_path (str or numpy): given image
        threshold (float): detection threshold
        model (Model): pre-trained model can be passed to the function
        align (bool): enable or disable alignment
        allow_upscaling (bool): allowing up-scaling
        expand_face_area (int): expand detected facial area with a percentage
        target_size (optional tuple): resize the image by padding it with black pixels
            to fit the specified dimensions. default is None
        min_max_norm (bool): set this to True if you want to normalize image in [0, 1].
            this is only running when target_size is not none.
            for instance, matplotlib expects inputs in this scale. (default is True)
    Returns:
        result (List[np.ndarray]): list of extracted faces
    """
    resp = []

    # ---------------------------

    img = preprocess.get_image(img_path)

    # ---------------------------

    obj = detect_faces(
        img_path=img, threshold=threshold, model=model, allow_upscaling=allow_upscaling
    )

    if not isinstance(obj, dict):
        return resp

    for _, identity in obj.items():
        facial_area = identity["facial_area"]
        rotate_angle = 0
        rotate_direction = 1

        x = facial_area[0]
        y = facial_area[1]
        w = facial_area[2] - x
        h = facial_area[3] - y

        if expand_face_area > 0:
            expanded_w = w + int(w * expand_face_area / 100)
            expanded_h = h + int(h * expand_face_area / 100)

            # overwrite facial area
            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(img.shape[1] - x, expanded_w)
            h = min(img.shape[0] - y, expanded_h)

        facial_img = img[y : y + h, x : x + w]

        if align is True:
            landmarks = identity["landmarks"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            nose = landmarks["nose"]
            # mouth_right = landmarks["mouth_right"]
            # mouth_left = landmarks["mouth_left"]

            # notice that left eye of one is seen on the right from your perspective
            aligned_img, rotate_angle, rotate_direction = postprocess.alignment_procedure(
                img=img, left_eye=right_eye, right_eye=left_eye, nose=nose
            )

            # find new facial area coordinates after alignment
            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = postprocess.rotate_facial_area(
                (x, y, x + w, y + h), rotate_angle, rotate_direction, (img.shape[0], img.shape[1])
            )
            facial_img = aligned_img[
                int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
            ]

        if target_size is not None:
            facial_img = postprocess.resize_image(
                img=facial_img, target_size=target_size, min_max_norm=min_max_norm
            )

        # to rgb
        facial_img = facial_img[:, :, ::-1]

        resp.append(facial_img)

    return resp
