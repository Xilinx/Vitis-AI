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

import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

EFFICIENTNET_MEAN_RGB = [127.0, 127.0, 127.0]
EFFICIENTNET_STDDEV_RGB = [128.0, 128.0, 128.0]
CROP_PADDING = 32

class DataLoader(object):
    def __init__(self, height=224, width=224):
        self.output_height = height
        self.output_width = width
        self.resize_side = 256

    def _smallest_size_at_least(self, height, width, smallest_side):
        """Computes new shape with the smallest side equal to `smallest_side`.

        Computes new shape with the smallest side equal to `smallest_side` while
        preserving the original aspect ratio.

        Args:
          height: an int32 scalar tensor indicating the current height.
          width: an int32 scalar tensor indicating the current width.
          smallest_side: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.

        Returns:
          new_height: an int32 scalar tensor indicating the new height.
          new_width: and int32 scalar tensor indicating the new width.
        """
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

        height = tf.cast(height, dtype=tf.float32)
        width = tf.cast(width, dtype=tf.float32)
        smallest_side = tf.cast(smallest_side, dtype=tf.float32)

        scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width,
                        lambda: smallest_side / height)
        new_height = tf.cast(tf.math.rint(height * scale), dtype=tf.int32)
        new_width = tf.cast(tf.math.rint(width * scale), dtype=tf.int32)
        return new_height, new_width

    def _aspect_preserving_resize(self, image, smallest_side):
        """Resize images preserving the original aspect ratio.

        Args:
          image: A 3-D image `Tensor`.
          smallest_side: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.

        Returns:
          resized_image: A 3-D tensor containing the resized image.
        """
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        new_height, new_width = self._smallest_size_at_least(
            height, width, smallest_side)
        image = tf.expand_dims(image, 0)
        resized_image = tf.compat.v1.image.resize_bilinear(image, [new_height, new_width],
                                                 align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return resized_image

    def _mean_image_subtraction(self, image, means):
        """Subtracts the given means from each image channel.

        For example:
          means = [123.68, 116.779, 103.939]
          image = _mean_image_subtraction(image, means)

        Note that the rank of `image` must be known.

        Args:
          image: a tensor of size [height, width, C].
          means: a C-vector of values to subtract from each channel.

        Returns:
          the centered image.

        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(
            axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

    def _stddev_image_division(self, image, stddev):
        """derive the given stddev from each image channel.

        For example:
          stddev = [128.0, 128.0, 128.0]
          image = _stddev_image_division(image, stddev)

        Note that the rank of `image` must be known.

        Args:
          image: a tensor of size [height, width, C].
          stddev: a C-vector of values to subtract from each channel.

        Returns:
          the centered image.

        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `stddev`.
        """
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(stddev) != num_channels:
            raise ValueError('len(stddev) must match the number of channels')

        channels = tf.split(
            axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] /= stddev[i]
        return tf.concat(axis=2, values=channels)

    def _efficientnet_preprocess(self, image):
        assert image is not None, "image cannot be None"
        assert self.output_height == self.output_width, "image output_width must be equal with output_height"
        image_size = self.output_height
        shape = tf.shape(image)
        image_height = shape[0]
        image_width = shape[1]

        padded_center_crop_size = tf.cast(
            ((image_size / (image_size + CROP_PADDING)) *
             tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

        offset_height = ((image_height - padded_center_crop_size) + 1) // 2
        offset_width = ((image_width - padded_center_crop_size) + 1) // 2
        crop_window = tf.stack([
            offset_height, offset_width, padded_center_crop_size,
            padded_center_crop_size
        ])

        image = image[offset_height:offset_height + padded_center_crop_size,
                      offset_width:offset_width + padded_center_crop_size, :]
        image = tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]

        image = self._mean_image_subtraction(image, EFFICIENTNET_MEAN_RGB)
        image = self._stddev_image_division(image, EFFICIENTNET_STDDEV_RGB)
        image = tf.expand_dims(image, 0)
        return image

    def _vgg_preprocess(self, image):
        image = self._aspect_preserving_resize(image, self.resize_side)

        # central crop
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        offset_height = (image_height - self.output_height) // 2
        offset_width = (image_width - self.output_height) // 2
        image = image[offset_height:offset_height + self.output_height,
                      offset_width:offset_width + self.output_width, :]
        image.set_shape([self.output_height, self.output_width, 3])

        image = tf.cast(image, dtype=tf.float32)
        image = self._mean_image_subtraction(
            image, [_R_MEAN, _G_MEAN, _B_MEAN])
        image = tf.expand_dims(image, 0)
        return image

    def _inception_preprocess(self, image, central_fraction=0.875,
                              central_crop=True):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_crop and central_fraction:
            image = tf.image.central_crop(
                image, central_fraction=central_fraction)

        if self.output_height and self.output_width:
            image = tf.expand_dims(image, 0)
            image = tf.compat.v1.image.resize_bilinear(image,
                                             [self.output_height,
                                                 self.output_width],
                                             align_corners=False)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

    def _build_placeholder(self):
        input_plhd = tf.compat.v1.placeholder(
            tf.string, shape=(None), name="input_image_path")
        image = tf.io.read_file(input_plhd)
        image = tf.image.decode_jpeg(image, channels=3)
        return image, input_plhd

    def build_preprocess(self, style="inception"):
        image, input_plhd = self._build_placeholder()
        preprocess_map = {"efficientnet": self._efficientnet_preprocess,
                          "vgg": self._vgg_preprocess,
                          "inception": self._inception_preprocess}
        if style in preprocess_map:
            image = preprocess_map[style](image)
        return image, input_plhd
