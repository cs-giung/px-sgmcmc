# Copyright 2025 cs-giung
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

import copy
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.dtypes import canonicalize_dtype


Dtype = Any


class FilterResponseNorm(nn.Module):
    """
    Filter Response Normalization Layer
    https://arxiv.org/abs/1911.09737
    """
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = jax.nn.initializers.zeros
    scale_init: Callable = jax.nn.initializers.ones
    threshold_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        y = inputs
        nu2 = jnp.mean(jnp.square(inputs), axis=(1, 2), keepdims=True)
        mul = jax.lax.rsqrt(nu2 + self.epsilon)
        if self.use_scale:
            scale = self.param(
                'scale', self.scale_init, (inputs.shape[-1],),
                self.param_dtype).reshape((1, 1, 1, -1))
            mul *= scale
        y *= mul
        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (inputs.shape[-1],),
                self.param_dtype).reshape((1, 1, 1, -1))
            y += bias
        tau = self.param(
            'threshold', self.threshold_init, (inputs.shape[-1],),
            self.param_dtype).reshape((1, 1, 1, -1))
        z = jnp.maximum(y, tau)
        dtype = canonicalize_dtype(scale, bias, tau, dtype=self.dtype)
        return jnp.asarray(z, dtype)


class ResNet20(nn.Module):
    """
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """
    conv: nn.Module = partial(
        nn.Conv, use_bias=True,
        kernel_init=jax.nn.initializers.he_normal(),
        bias_init=jax.nn.initializers.zeros)
    norm: nn.Module = FilterResponseNorm
    relu: Callable = jax.nn.silu
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):

        y = self.conv(features=16, kernel_size=(3, 3), strides=(1, 1))(x)
        y = self.norm()(y)
        y = self.relu(y)

        for layer_idx, num_block in enumerate([3, 3, 3]):
            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)

            for _s_idx, s in enumerate(_strides, start=1):
                _channel = 16 * (2 ** layer_idx)
                residual = y

                y = self.conv(
                    features=_channel, kernel_size=(3, 3), strides=(s, s))(y)
                y = self.norm()(y)
                y = self.relu(y)

                y = self.conv(
                    features=_channel, kernel_size=(3, 3), strides=(1, 1))(y)
                y = self.norm()(y)

                # Note that we use projection shortcut here
                if residual.shape != y.shape:
                    residual = self.conv(
                        features=y.shape[-1],
                        kernel_size=(1, 1), strides=(s, s))(residual)
                    residual = self.norm()(residual)

                y = self.relu(y + residual)

        y = jnp.mean(y, axis=(1, 2))
        y = nn.Dense(self.num_classes)(y)

        return y


class PXResNet20(nn.Module):
    conv: nn.Module = partial(
        nn.Conv, use_bias=True,
        kernel_init=jax.nn.initializers.he_normal(),
        bias_init=jax.nn.initializers.zeros)
    norm: nn.Module = FilterResponseNorm
    relu: Callable = jax.nn.silu
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, **kwargs):

        get_perm = kwargs.get('get_perm', False)
        perm = dict()
        count = 0
        op = dict()
        layer_count = dict(Conv=0, FilterResponseNorm=0, Dense=0)
        def add_perm(cnt, ch):
            name = f'perm_{cnt}'
            if get_perm:
                perm[name] = jnp.eye(ch)
                return name, cnt + 1
            return name, cnt

        def add_op(perm_name, right=None, left=None):
            if get_perm:
                a = op.get(perm_name)
                if a is None:
                    op[perm_name] = dict(right=[], left=[])
                if right is not None:
                    right_name = f'{right}_{layer_count[right]}'
                    op[perm_name]['right'].append(right_name)
                    layer_count[right] += 1
                if left is not None:
                    left_name = f'{left}_{layer_count[left]}'
                    op[perm_name]['left'].append(left_name)

        y = self.conv(features=16, kernel_size=(3, 3), strides=(1, 1))(x)
        p, count = add_perm(count, y.shape[-1])
        add_op(p, right='Conv')
        y = self.norm()(y)
        p, count = add_perm(count, y.shape[-1])
        add_op(p, right='FilterResponseNorm')
        y = self.relu(y)

        for layer_idx, num_block in enumerate([3, 3, 3]):
            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)

            for _s_idx, s in enumerate(_strides, start=1):
                _channel = 16 * (2 ** layer_idx)
                residual = y
                p, count = add_perm(count, y.shape[-1])
                add_op(p, left='Conv')

                y = self.conv(
                    features=_channel, kernel_size=(3, 3), strides=(s, s))(y)
                p, count = add_perm(count, y.shape[-1])
                add_op(p, right='Conv')
                y = self.norm()(y)
                p, count = add_perm(count, y.shape[-1])
                add_op(p, right='FilterResponseNorm')
                y = self.relu(y)

                p, count = add_perm(count, y.shape[-1])
                add_op(p, left='Conv')
                y = self.conv(
                    features=_channel, kernel_size=(3, 3), strides=(1, 1))(y)
                p, count = add_perm(count, y.shape[-1])
                add_op(p, right='Conv')
                y = self.norm()(y)
                p, count = add_perm(count, y.shape[-1])
                add_op(p, right='FilterResponseNorm')

                if residual.shape != y.shape:
                    p, count = add_perm(count, residual.shape[-1])
                    add_op(p, left='Conv')
                    residual = self.conv(
                        features=y.shape[-1],
                        kernel_size=(1, 1), strides=(s, s))(residual)
                    p, count = add_perm(count, residual.shape[-1])
                    add_op(p, right='Conv')
                    residual = self.norm()(residual)
                    p, count = add_perm(count, residual.shape[-1])
                    add_op(p, right='FilterResponseNorm')

                y = self.relu(y + residual)

        y = jnp.mean(y, axis=(1, 2))
        p, count = add_perm(count, y.shape[-1])
        add_op(p, left='Dense')
        y = nn.Dense(self.num_classes)(y)

        if get_perm:
            return perm, op

        return y


def permute_params_apply(permute_params, op, model_params):
    permute_model_params = copy.deepcopy(model_params)

    for i, (perm_name, P) in enumerate(permute_params.items()):
        right_target = op[perm_name]["right"]
        left_target = op[perm_name]["left"]

        for right in right_target:
            permute_model_params[right] = targeting(
                "right", right, permute_model_params[right], P)

        for left in left_target:
            permute_model_params[left] = targeting(
                "left", left, permute_model_params[left], P)

    return permute_model_params


def targeting(dir, layer, param, P):

    if isinstance(P, (list, tuple)):
        Q = P[0]
        for p in P[1:]:
            Q = Q @ p
        P = Q

    if "Dense" in layer:
        if dir == "right":
            param["kernel"] = param["kernel"] @ P
            param["bias"] = param["bias"].T @ P
        else:
            param["kernel"] = P.T @ param["kernel"]

    elif "Conv" in layer:
        if dir == "right":
            param["kernel"] = jnp.einsum("abij,jk ->abik", param["kernel"], P)
            param['bias'] = param['bias'].T @ P
        else:
            param["kernel"] = jnp.einsum("ki,abij->abkj", P.T, param["kernel"])

    elif "FilterResponseNorm" in layer:
        if dir == "right":
            param["scale"] = param["scale"].T @ P
            param["bias"] = param["bias"].T @ P
            param["threshold"] = param["threshold"].T @ P
        else:
            raise Exception("No left operation for FilterResponseNorm")

    else:
        raise NotImplementedError

    return param