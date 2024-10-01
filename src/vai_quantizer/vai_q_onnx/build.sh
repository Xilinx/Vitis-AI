#!/usr/bin/env bash
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
set -e
set -x

bash ./pip_pkg.sh ./pkgs/ --release
