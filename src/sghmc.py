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

from typing import (
    Any, Callable, Mapping, NamedTuple, Iterable, Optional, Tuple, Union)

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


PRNGKey = Array
PRNGKeyLike = ArrayLike

Pytree = Union[
    Array, Iterable["Pytree"], Mapping[Any, "Pytree"]]
PytreeLike = Union[
    ArrayLike, Iterable["PytreeLike"], Mapping[Any, "PytreeLike"]]

Param = PytreeLike
Batch = PytreeLike


class SGHMCState(NamedTuple):
    """State including position and momentum."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum: Pytree


def randn_like(rng_key: PRNGKeyLike, pytree: PytreeLike) -> Pytree:
    """
    Returns a pytree with the same structure as `pytree` that is filled with
    random numbers from a normal distribution with mean 0 and variance 1.
    """
    tree = jax.tree_util.tree_structure(pytree)
    keys = jax.tree_util.tree_unflatten(
        tree, jax.random.split(rng_key, tree.num_leaves))
    return jax.tree_util.tree_map(
        lambda p, k: jax.random.normal(k, p.shape, p.dtype), pytree, keys)


def step(
        state: SGHMCState,
        batch: Batch,
        energy_fn: Callable[[Param, Batch], Any],
        step_size: float,
        friction: Union[float, PytreeLike] = None,
        momentum_decay: PytreeLike = None,
        momentum_stdev: float = 1.0,
        gradient_noise: float = 0.0,
        temperature: float = 1.0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
    ) -> Tuple[Any, SGHMCState]:
    """
    Stochastic Gradient Hamiltonian Monte Carlo
    https://arxiv.org/abs/1402.4102

    Args:
        state: Current state.
        batch: It will be send to `energy_fn`.
        energy_fn: Energy function to be differentiated. It should take
            `state.position` and `batch` and return the posterior energy value
            as well as auxiliary information.
        step_size: Step size, denoted by $\\epsilon$ in the paper. Note that
            `step_size**2 * train_size` corresponds to the learning rate in the
            conventional MomentumSGD implementation.
        friction: Friction coefficient, denoted by $CM^{-1}$ in the paper.
        momentum_decay: Momentum decay coefficient, denoted by $\\alpha$ in the
            paper. Note that `(1 - momentum_decay)` corresponds to the momentum
            coefficient in the convnetional MomentumSGD implementation.
        momentum_stdev: Standard deviation of momentum target distribution.
        gradient_noise: Gradient noise coefficient for non-tempered posterior.
        temperature: Temperature of joint distribution for posterior tempering.
        has_aux: It indicates whether the `energy_fn` returns a pair, with the
            first element as the main output of the energy function for
            differentiation and the second element as optional auxiliary data.
        axis_name: `gradients` will be averaged across replicas if specified.

    Returns:
        Auxiliary data and updated state.
    """
    if friction is None and momentum_decay is None:
        raise AssertionError(
            'Either friction or momentum_decay must be specified.')
    if momentum_decay is None:
        if isinstance(friction, float):
            friction = jax.tree_util.tree_map(
                lambda _: friction, state.position)
        momentum_decay = jax.tree_util.tree_map(
            lambda f: step_size * f, friction)

    aux, gradient = jax.value_and_grad(
        energy_fn, argnums=0, has_aux=has_aux)(state.position, batch)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)

    noise = randn_like(state.rng_key, state.position)
    momentum = jax.tree_util.tree_map(
        lambda m, g, n, md:
            m * (1. - md)
            + g * step_size
            + n * jnp.sqrt(
                2. * md * momentum_stdev**2 * temperature
                - gradient_noise * step_size**2 * temperature**2),
        state.momentum, gradient, noise, momentum_decay)
    position = jax.tree_util.tree_map(
        lambda p, m: p - m * step_size / momentum_stdev**2,
        state.position, momentum)

    return aux, SGHMCState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position, momentum=momentum)