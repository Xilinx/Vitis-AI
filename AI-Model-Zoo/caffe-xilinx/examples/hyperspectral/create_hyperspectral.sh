#!/usr/bin/env sh
# This script converts the hyperspectral data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

EXAMPLE=examples/hyperspectral
DATA=data/hyperspectral
BUILD=build/examples/hyperspectral

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/hyperspectral_train_${BACKEND}
rm -rf $EXAMPLE/hyperspectral_test_${BACKEND}

$BUILD/convert_hyperspectral_data.bin $DATA/train-images-idx4-float \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/hyperspectral_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_hyperspectral_data.bin $DATA/t300-images-idx4-float \
  $DATA/t300-labels-idx1-ubyte $EXAMPLE/hyperspectral_test_${BACKEND} --backend=${BACKEND}

echo "Done."
