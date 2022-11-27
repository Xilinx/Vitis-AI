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
# ==============================================================================
"""Python implementation for calibrator ."""

import tensorflow as tf
import numpy as np

from scipy.stats import entropy
from collections import Counter


def collect(inputs,
            calib_hist,
            calib_bin_edges,
            bit_width,
            num_bins,
            unsigned,
            skip_zeros,
            axis=None):
  """method: collect input histogram of tensor statistics used to compute amax

  Args:
    inputs: A tensor for data input
    calib_hist: A Variable of histogram for input
    calib_bin_edges: A Variable linspace for calib_hist
    bit_width: An integer. Number of bits of quantization.
    num_bins: An integer. Number of histograms bins. Default 2048.
    unsigned: A boolean. using unsigned quantization.
    skip_zeros: A boolean. If True, skips zeros when collecting data for histogram. Default False.
    axis: A tuple.
  Returns:
    calib_hist: A Variable of histogram for input
    calib_bin_edges: A Variable linspace for calib_hist
  """

  inputs_np = np.absolute(inputs)
  if skip_zeros:
    # whether skip zero
    inputs_np = inputs_np[np.where(inputs_np != 0)]

  if calib_hist.size == 1:
    # first time it uses num_bins to compute histogram
    calib_hist, calib_bin_edges = np.histogram(inputs_np, bins=num_bins)
  # intailized
  else:
    temp_amax = np.max(inputs_np)
    if temp_amax > calib_bin_edges[-1]:
      # increase the number of bins
      width = calib_bin_edges[1] - calib_bin_edges[0]
      # NOTE: np.arange may create an extra bin after the one containing temp_amax
      new_bin_edges = np.arange(calib_bin_edges[-1] + width, temp_amax + width,
                                width)
      calib_bin_edges = np.hstack((calib_bin_edges, new_bin_edges))
    hist, calib_bin_edges = np.histogram(inputs_np, bins=calib_bin_edges)
    hist[:len(calib_hist)] += calib_hist
    calib_hist = hist
  calib_hist = calib_hist.astype(np.int32)
  calib_bin_edges = calib_bin_edges.astype(np.float32)

  return calib_hist, calib_bin_edges


def compute_amax_percentile(percentile, calib_hist, calib_bin_edges):
  """Returns amax that clips the percentile fraction from the collected data
  
  Args:
    percentile: A percentile Value
    calib_hist: A Variable of histogram for input
    calib_bin_edges: A Variable linspace for calib_hist
  Returns:
    percentile_amax: amax that clips the percentile fraction
  """
  if percentile < 0 or percentile > 100:
    raise ValueError(
        "Invalid percentile. Must be in range 0 <= percentile <= 100.")

  # If calibrator hasn't collected any data, return none
  if calib_bin_edges is None and calib_hist is None:
    return None

  total = calib_hist.sum()
  cdf = np.cumsum(calib_hist / total)
  idx = np.searchsorted(cdf, percentile / 100)
  percentile_amax = calib_bin_edges[idx]
  return percentile_amax.astype(np.float32)


def compute_amax_entropy(calib_hist, calib_bin_edges, bit_width, unsigned,
                         stride, start_bin):
  """Returns amax that minimizes KL-Divergence from the collected histogram

  Args:
    calib_hist: A Variable of histogram for input
    calib_bin_edges: A Variable linspace for calib_hist
    bit_width: An integer. Number of bits of quantization.
    unsigned: A boolean. using unsigned quantization.
    stride: Stride for calib_bin_edges, default is 1
    start_bin: Start of calib_bin_edges, default is 128.
  Returns:
    calib_amax: Amax of histogram when use entropy method
  """

  # If calibrator hasn't collected any data, return none
  if calib_bin_edges is None and calib_hist is None:
    return None

  def _normalize_distr(distr):
    summ = np.sum(distr)
    if summ != 0:
      distr = distr / summ

  bins = calib_hist[:]
  bins[0] = bins[1]

  total_data = np.sum(bins)

  divergences = []
  arguments = []

  # we are quantizing to 128 values + sign if num_bits=8
  nbins = 1 << (bit_width - 1 + int(unsigned))

  starting = start_bin
  stop = len(bins)

  new_density_counts = np.zeros(nbins, dtype=np.float64)

  for i in range(starting, stop + 1, stride):
    new_density_counts.fill(0)
    space = np.linspace(0, i, num=nbins + 1)
    digitized_space = np.digitize(range(i), space) - 1

    digitized_space[bins[:i] == 0] = -1

    for idx, digitized in enumerate(digitized_space):
      if digitized != -1:
        new_density_counts[digitized] += bins[idx]

    counter = Counter(digitized_space)
    for key, val in counter.items():
      if key != -1:
        new_density_counts[key] = new_density_counts[key] / val

    new_density = np.zeros(i, dtype=np.float64)
    for idx, digitized in enumerate(digitized_space):
      if digitized != -1:
        new_density[idx] = new_density_counts[digitized]

    total_counts_new = np.sum(new_density) + np.sum(bins[i:])
    _normalize_distr(new_density)

    reference_density = np.array(bins[:len(digitized_space)])
    reference_density[-1] += np.sum(bins[i:])

    total_counts_old = np.sum(reference_density)
    if round(total_counts_new) != total_data or round(
        total_counts_old) != total_data:
      raise RuntimeError(
          "Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}"
          .format(total_counts_new, total_counts_old, total_data))

    _normalize_distr(reference_density)

    ent = entropy(reference_density, new_density)
    divergences.append(ent)
    arguments.append(i)
  divergences = np.array(divergences)
  # get the calib_bin_edges index
  last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
  calib_amax = calib_bin_edges[last_argmin * stride + starting]
  return calib_amax.astype(np.float32)


def numpy_collect(inputs,
                  calib_hist,
                  calib_bin_edges,
                  bit_width: int = 8,
                  num_bins: int = 2048,
                  unsigned: bool = False,
                  skip_zeros: bool = False):
  """numpy wrapper for collect."""
  calib_hist, calib_bin_edges = tf.numpy_function(collect, [
      inputs, calib_hist, calib_bin_edges, bit_width, num_bins, unsigned,
      skip_zeros
  ], [np.int32, np.float32])
  return calib_hist, calib_bin_edges


def numpy_kl_div(calib_hist: int,
                 calib_bin_edges: float,
                 bit_width: int = 8,
                 unsigned: bool = False,
                 stride: int = 1,
                 start_bin: int = 128):
  """numpy wrapper for compute_amax_entropy."""
  amax = tf.numpy_function(
      compute_amax_entropy,
      [calib_hist, calib_bin_edges, bit_width, unsigned, stride, start_bin],
      np.float32)
  return amax


def numpy_percentile(percentile: float, calib_hist: int,
                     calib_bin_edges: float):
  """numpy wrapper for compute_amax_percentile."""
  amax = tf.numpy_function(compute_amax_percentile,
                           [percentile, calib_hist, calib_bin_edges],
                           np.float32)
  return amax
