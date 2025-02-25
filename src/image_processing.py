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

from typing import List
from abc import ABCMeta, abstractmethod

import jax
import jax.numpy as jnp


class Transform(metaclass=ABCMeta):
    """Base class for transformations."""

    @abstractmethod
    def __call__(self, rng, image):
        """Apply the transform on an image."""


class TransformChain(Transform):
    """Chain multiple transformations."""

    def __init__(self, transforms: List[Transform], prob: float = 1.0):
        """Apply transforms with the given probability."""
        self.transforms = transforms
        self.prob = prob

    def __call__(self, rng, image):
        jmage = image
        _rngs = jax.random.split(rng, len(self.transforms))
        for _transform, _rng in zip(self.transforms, _rngs):
            jmage = _transform(_rng, jmage)
        return jnp.where(jax.random.bernoulli(rng, self.prob), jmage, image)


class RandomHFlipTransform(Transform):
    """RandomHFlipTransform"""

    def __init__(self, prob=0.5):
        """Flips an image horizontally with the given probability."""
        self.prob = prob

    def __call__(self, rng, image):
        jmage = jnp.flip(image, axis=1)
        return jnp.where(
            jax.random.bernoulli(rng, self.prob), jmage, image)


class RandomCropTransform(Transform):
    """RandomCropTransform"""

    def __init__(self, size, padding):
        """Crops an image with the given size and padding."""
        self.size = size
        self.padding = padding

    def __call__(self, rng, image):
        rngs = jax.random.split(rng, 2)
        pad_width = (
            (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        image = jnp.pad(
            image, pad_width=pad_width, mode='constant', constant_values=0)
        h0 = jax.random.randint(
            rngs[0], shape=(1,), minval=0, maxval=2*self.padding+1)[0]
        w0 = jax.random.randint(
            rngs[1], shape=(1,), minval=0, maxval=2*self.padding+1)[0]
        image = jax.lax.dynamic_slice(
            image, start_indices=(h0, w0, 0),
            slice_sizes=(self.size, self.size, image.shape[2]))
        return image