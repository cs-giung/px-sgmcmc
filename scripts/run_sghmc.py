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

import math
import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
sys.path.append('./')

import flax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from scripts.default import get_args
from src import image_processing, sghmc
from src.model import ResNet20
from src.utils import load, save


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--data_root', default='./data/', type=str,
        help='root directory containing dataset files')
    parser.add_argument(
        '--data_name', default='cifar10', type=str,
        help='dataset name (default: cifar10)')
    parser.add_argument(
        '--data_augmentation', default='none', type=str,
        help='apply data augmentation during training (default: none)')

    parser.add_argument(
        '--num_samples', default=100, type=int,
        help='the number of samples (default: 100)')
    parser.add_argument(
        '--num_updates', default=5000, type=int,
        help='the number of updates for each sample (default: 5000)')
    parser.add_argument(
        '--num_batch', default=256, type=int,
        help='the number of instances in mini-batch (default: 256)')

    parser.add_argument(
        '--step_size', default=1e-05, type=float,
        help='base step size (default: 1e-05)')
    parser.add_argument(
        '--step_size_min', default=None, type=float,
        help='decay step size if specified (default: None)')

    parser.add_argument(
        '--posterior_temperature', default=1.0, type=float,
        help='temperature for posterior tempering (default: 1.0)')
    parser.add_argument(
        '--prior_variance', default=0.05, type=float,
        help='variance of zero-mean Gaussian prior (default: 0.05)')
    parser.add_argument(
        '--friction', default=100.0, type=float,
        help='friction coefficient (default: 100.0)')

    args, print_fn, time_stamp = get_args(
        parser, exist_ok=False, dot_log_file=False,
        libraries=(flax, jax, jaxlib))

    if args.step_size_min is None:
        args.step_size_min = args.step_size

    # ----------------------------------------------------------------------- #
    # Data
    # ----------------------------------------------------------------------- #
    shard_shape = (jax.local_device_count(), -1)
    input_shape = (32, 32, 3)
    num_classes = {
        'cifar10': 10, 'cifar100': 100, 'tin32': 200}[args.data_name]
    trn_dataset_size = {
        'cifar10': 40960, 'cifar100': 40960, 'tin32': 81920}[args.data_name]
    trn_steps_per_epoch = math.floor(trn_dataset_size / args.num_batch)

    trn_images = np.load(os.path.join(
        args.data_root, args.data_name, 'train_images.npy'))[:trn_dataset_size]
    trn_labels = np.load(os.path.join(
        args.data_root, args.data_name, 'train_labels.npy'))[:trn_dataset_size]
    val_images = np.load(os.path.join(
        args.data_root, args.data_name, 'train_images.npy'))[trn_dataset_size:]
    val_labels = np.load(os.path.join(
        args.data_root, args.data_name, 'train_labels.npy'))[trn_dataset_size:]

    if args.data_augmentation == 'none':
        trn_augmentation = None
    elif args.data_augmentation == 'simple':
        trn_augmentation = jax.jit(jax.vmap(image_processing.TransformChain([
            image_processing.RandomCropTransform(size=32, padding=4),
            image_processing.RandomHFlipTransform(prob=0.5)])))
    else:
        raise NotImplementedError(
            f'Unknown args.data_augmentation={args.data_augmentation}')

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #
    pixel_m = np.array([0.49, 0.48, 0.44])
    pixel_s = np.array([0.2, 0.2, 0.2])
    model = ResNet20(num_classes=num_classes)

    init_position = model.init(
        jax.random.PRNGKey(args.seed), jnp.ones((1,) + input_shape))['params']

    # ----------------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------------- #
    def softmax(logits):
        """Computes softmax of logits."""
        return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    def compute_err(logits, labels):
        """Computes classification error."""
        return np.mean(np.not_equal(np.argmax(logits, axis=-1), labels))

    def compute_nll(logits, labels):
        """Computes categorical negative log-likelihood."""
        return np.mean(np.negative(
            np.log(softmax(logits)[np.arange(labels.shape[0]), labels])))

    def get_metrics(device_metrics):
        """Get metrics."""
        return jax.tree_util.tree_map(
            lambda *args: np.stack(args), *jax.device_get(
                jax.tree_util.tree_map(lambda x: x[0], device_metrics)))

    def forward_fn(params, inputs):
        """Computes categorical logits."""
        inputs = inputs / 255.0
        inputs = inputs - pixel_m[None, None, None]
        inputs = inputs / pixel_s[None, None, None]
        logits = model.apply({'params': params}, inputs)
        return logits

    p_forward_fn = jax.pmap(forward_fn)

    def make_predictions(replicated_params, images):
        """Returns logits and labels for val split."""
        dataset_size = images.shape[0]
        steps_per_epoch = math.ceil(dataset_size / args.num_batch)

        images = np.concatenate([images, np.zeros((
            args.num_batch * steps_per_epoch - dataset_size,
            *images.shape[1:]), dtype=images.dtype)])
        _queue = np.arange(images.shape[0]).reshape(-1, args.num_batch)

        logits = np.concatenate([
            jax.device_put(p_forward_fn(
                replicated_params, images[batch_index].reshape(
                    shard_shape + input_shape)).reshape(args.num_batch, -1),
                jax.devices('cpu')[0]) for batch_index in _queue])
        logits = logits[:dataset_size]

        return logits

    def energy_fn(param, batch):
        """Computes unnormalized posterior energy."""
        logits = forward_fn(param, batch['inputs'])
        target = batch['target']

        neg_log_likelihood = jnp.negative(trn_dataset_size * jnp.mean(
            jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)))
        neg_log_prior = 0.5 * sum(
            jnp.sum(e**2) / args.prior_variance
            for e in jax.tree_util.tree_leaves(param))
        posterior_energy = neg_log_likelihood + neg_log_prior

        aux = OrderedDict({
            'posterior_energy': posterior_energy,
            'neg_log_likelihood': neg_log_likelihood,
            'neg_log_prior': neg_log_prior})

        return posterior_energy, aux

    @partial(jax.pmap, axis_name='batch')
    def update_fn(state, batch, step_size):
        """Updates state."""
        aux, state = sghmc.step(
            state=state,
            batch=batch,
            energy_fn=energy_fn,
            step_size=step_size,
            friction=args.friction,
            temperature=args.posterior_temperature,
            has_aux=True, axis_name='batch')
        aux[1]['step_size'] = step_size
        return aux, state

    init_momentum = \
        jax.tree_util.tree_map(jnp.zeros_like, init_position)

    state = sghmc.SGHMCState(
        step=0, rng_key=jax.random.PRNGKey(args.seed),
        position=init_position, momentum=init_momentum)
    state = jax.device_put_replicated(state, jax.local_devices())

    batch_rng = jax.random.PRNGKey(args.seed)
    batch_queue = np.asarray(
        jax.random.permutation(batch_rng, trn_dataset_size))

    ens_trn_ps = np.zeros((trn_images.shape[0], num_classes))
    ens_trn_ls = np.zeros((trn_images.shape[0], num_classes))
    ens_trn_ls_nlls = []

    ens_val_ps = np.zeros((val_images.shape[0], num_classes))
    ens_val_ls = np.zeros((val_images.shape[0], num_classes))
    ens_val_ls_nlls = []

    if args.save:
        sample_idx = 0 
        save(
            os.path.join(args.save, f'{sample_idx:06d}'),
            jax.tree_util.tree_map(lambda e: e[0], state))

    for sample_idx in range(1, args.num_samples + 1):
        metrics = []
        for update_idx in range(1, args.num_updates + 1):

            if batch_queue.shape[0] <= args.num_batch:
                batch_rng = jax.random.split(batch_rng)[0]
                batch_queue = np.concatenate((batch_queue,
                    jax.random.permutation(batch_rng, trn_dataset_size)))
            batch_index = batch_queue[:args.num_batch]
            batch_queue = batch_queue[args.num_batch:]

            batch = {
                'inputs': trn_images[batch_index],
                'target': jax.nn.one_hot(trn_labels[batch_index], num_classes)}
            if trn_augmentation:
                batch['inputs'] = (trn_augmentation(
                    jax.random.split(
                        jax.random.PRNGKey(update_idx), args.num_batch
                    ), batch['inputs'] / 255.) * 255.).astype(trn_images.dtype)
            batch = jax.tree_util.tree_map(
                lambda e: e.reshape(shard_shape + e.shape[1:]), batch)

            step_size = jax.device_put_replicated(
                args.step_size_min + (0.5 + 0.5 * np.cos(
                    (update_idx - 1) / args.num_updates * np.pi)
                ) * (args.step_size - args.step_size_min),
                jax.local_devices())
            aux, state = update_fn(state, batch, step_size)
            metrics.append(aux[1])

            if update_idx == 1 or update_idx % 1000 == 0:
                summarized = {
                    f'trn/{k}': float(v) for k, v in jax.tree_util.tree_map(
                        lambda e: e.mean(), get_metrics(metrics)).items()}

                if update_idx != 1:
                    metrics = []

                summarized['norm'] = float(jnp.sqrt(sum(
                    jnp.sum(e**2) for e in jax.tree_util.tree_leaves(
                        jax.tree_util.tree_map(
                            lambda e: e[0], state.position)))))

                logits = make_predictions(state.position, trn_images)
                summarized['trn/err'] = compute_err(logits, trn_labels)
                summarized['trn/nll'] = compute_nll(logits, trn_labels)

                logits = make_predictions(state.position, val_images)
                summarized['val/err'] = compute_err(logits, val_labels)
                summarized['val/nll'] = compute_nll(logits, val_labels)

                print_fn(
                    f'[Sample {sample_idx:6d}/{args.num_samples:6d}] '
                    f'[Update {update_idx:6d}/{args.num_updates:6d}] '
                    + ', '.join(
                        f'{k}: {v:.3e}' for k, v in summarized.items()))

                if jnp.isnan(summarized['trn/posterior_energy']):
                    break

        if jnp.isnan(summarized['trn/posterior_energy']):
            break

        summarized = {}

        summarized['norm'] = float(jnp.sqrt(sum(
            jnp.sum(e**2) for e in jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda e: e[0], state.position)))))

        logits = make_predictions(state.position, trn_images)
        summarized['trn/err'] = compute_err(logits, trn_labels)
        summarized['trn/nll'] = compute_nll(logits, trn_labels)

        ens_trn_ps = (
            ens_trn_ps * (sample_idx - 1) + softmax(logits)) / sample_idx
        summarized['trn/ens_err'] = compute_err(np.log(ens_trn_ps), trn_labels)
        summarized['trn/ens_nll'] = compute_nll(np.log(ens_trn_ps), trn_labels)

        ens_trn_ls = (ens_trn_ls * (sample_idx - 1) + logits) / sample_idx
        ens_trn_ls_nlls.append(summarized['trn/nll'])
        summarized['trn/ens_amb'] = \
            np.mean(ens_trn_ls_nlls) - compute_nll(ens_trn_ls, trn_labels)

        logits = make_predictions(state.position, val_images)
        summarized['val/err'] = compute_err(logits, val_labels)
        summarized['val/nll'] = compute_nll(logits, val_labels)

        ens_val_ps = (
            ens_val_ps * (sample_idx - 1) + softmax(logits)) / sample_idx
        summarized['val/ens_err'] = compute_err(np.log(ens_val_ps), val_labels)
        summarized['val/ens_nll'] = compute_nll(np.log(ens_val_ps), val_labels)

        ens_val_ls = (ens_val_ls * (sample_idx - 1) + logits) / sample_idx
        ens_val_ls_nlls.append(summarized['val/nll'])
        summarized['val/ens_amb'] = \
            np.mean(ens_val_ls_nlls) - compute_nll(ens_val_ls, val_labels)

        print_fn(
            f'[Sample {sample_idx:6d}/{args.num_samples:6d}] '
            + ', '.join(f'{k}: {v:.3e}' for k, v in summarized.items()))

        if args.save:
            save(
                os.path.join(args.save, f'{sample_idx:06d}'),
                jax.tree_util.tree_map(lambda e: e[0], state))