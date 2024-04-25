import torch
import numpy as np

# Copyright © 2024 voxelmorph
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# https://github.com/voxelmorph/voxelmorph
#
# Edited by Jamie Temple
# Date: 2024-04-08


class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, window=None, weight=None):
        super().__init__()
        self.window = window
        self.weight = weight

    def forward(self, target, prediction):
        t = target
        p = prediction

        n_dims = len(t.shape) - 2
        assert n_dims in [1, 2, 3], (
            "volumes should be 1 to 3 dimensions. found: %d" % n_dims
        )

        window = [9] * n_dims if self.window is None else self.window
        sum_filter = torch.ones([1, 1, *window]).to(t.device)

        same_padding = window[0] // 2
        stride = tuple([1] * n_dims)
        padding = tuple([same_padding] * n_dims)

        # get convolution function
        conv_fn = getattr(torch.nn.functional, "conv%dd" % n_dims)

        # compute CC squares
        t_sqr = t * t
        p_sqr = p * p
        t_p = t * p

        t_sum = conv_fn(t, sum_filter, stride=stride, padding=padding)
        p_sum = conv_fn(p, sum_filter, stride=stride, padding=padding)
        t_sqr_sum = conv_fn(t_sqr, sum_filter, stride=stride, padding=padding)
        p_sqr_sum = conv_fn(p_sqr, sum_filter, stride=stride, padding=padding)
        t_p_sum = conv_fn(t_p, sum_filter, stride=stride, padding=padding)

        window_size = np.prod(window)
        t_mean = t_sum / window_size
        p_mean = p_sum / window_size

        # covariance and variance over window
        cross = (
            t_p_sum - p_mean * t_sum - t_mean * p_sum + t_mean * p_mean * window_size
        )
        t_var = t_sqr_sum - 2 * t_mean * t_sum + t_mean * t_mean * window_size
        p_var = p_sqr_sum - 2 * p_mean * p_sum + p_mean * p_mean * window_size

        # correlation coefficient
        cc = cross * cross / (t_var * p_var + 1e-5)

        return -torch.mean(cc)
