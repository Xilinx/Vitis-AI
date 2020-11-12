# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization,ReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate

def model_forward(inputs):
    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = ReLU()(c3)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = ReLU()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(256, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = ReLU()(c4)  
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(256, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = ReLU()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(512, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization()(c5)
    c5 = ReLU()(c5)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(512, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization()(c5)
    c5 = ReLU()(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = ReLU()(c6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(256, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization()(c6)
    c6 = ReLU()(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = ReLU()(c7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(128, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization()(c7)
    c7 = ReLU()(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = ReLU()(c8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(64, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c8)
    c8 = BatchNormalization()(c8)
    c8 = ReLU()(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = ReLU()(c9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(32, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (c9)
    c9 = BatchNormalization()(c9)
    c9 = ReLU()(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    return outputs
