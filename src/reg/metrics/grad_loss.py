import torch

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


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l2", weight=None):
        super().__init__()
        assert (penalty == "l1" or penalty == "l2"), "penalty can only be l1 or l2. Got: %s" % self.penalty
        
        self.penalty = penalty
        self.weight = weight

    def forward(self, flow):
        y = flow

        vol_shape = y.shape[2:]  # (batch, channel, ...)
        n_dims = len(vol_shape)

        assert n_dims > 0, "Requires at least one spatial dimension."

        df = [None] * n_dims
        for i in range(n_dims):

            # compute gradient along dimension d
            d = i + 2
            # (d, 0, 1, ..., d-1, d+1, ..., n_dims + 1)
            r = [d, *range(0, d), *range(d + 1, n_dims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            r = [*range(1, d + 1), 0, *range(d + 1, n_dims + 2)]
            df[i] = dfi.permute(r)

        if self.penalty == "l1":
            dif = [torch.abs(f) for f in df]
        else:
            dif = [f * f for f in df]

        d = [g.mean() for g in dif]
        grad_loss = sum(d) / len(d)

        if self.weight is not None:
            grad_loss *= self.weight

        return grad_loss
