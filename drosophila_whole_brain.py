# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import itertools
import jax
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Union, Sequence, Dict, Optional

import brainevent
import brainstate
import brainunit as u


class Population(brainstate.nn.Neuron):
    """
    A population of neurons with leaky integrate-and-fire dynamics.

    The dynamics of the neurons are given by the following equations::

       dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
       dg/dt = -g / tau               : volt (unless refractory)


    Args:
      size: The number of neurons in the population.
      v_rest: The resting potential of the neurons.
      v_reset: The reset potential of the neurons after a spike.
      v_th: The threshold potential of the neurons for spiking.
      tau_m: The membrane time constant of the neurons.
      tau_syn: The synaptic time constant of the neurons.
      tau_ref: The refractory period of the neurons.
      spk_fun: The spike function of the neurons.
      name: The name of the population.

    """

    def __init__(
        self,
        size: brainstate.typing.Size,
        v_rest: u.Quantity = -52 * u.mV,  # resting potential
        v_reset: u.Quantity = -52 * u.mV,  # reset potential after spike
        v_th: u.Quantity = -45 * u.mV,  # potential threshold for spiking
        tau_m: u.Quantity = 20 * u.ms,  # membrane time constant
        # Jürgensen et al https://doi.org/10.1088/2634-4386/ac3ba6
        tau_syn: u.Quantity = 5 * u.ms,  # synaptic time constant
        # Lazar et al https://doi.org/10.7554/eLife.62362
        tau_ref: u.Quantity = 2.2 * u.ms,  # refractory period
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),  # spike function
        name: str = None,
    ):
        super().__init__(size, name=name)

        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.tau_m = tau_m
        self.tau_syn = tau_syn
        self.tau_ref = u.math.full(self.varshape, tau_ref)
        self.spk_fun = spk_fun

    def init_state(self, batch_size=None):
        self.v = brainstate.ShortTermState(brainstate.init.param(brainstate.init.Constant(self.v_rest), self.varshape, batch_size))
        self.g = brainstate.ShortTermState(brainstate.init.param(brainstate.init.Constant(0.0 * u.mV), self.varshape, batch_size))
        self.t_ref = brainstate.ShortTermState(brainstate.init.param(brainstate.init.Constant(-1e7 * u.ms), self.varshape, batch_size))
        self.spike_count = brainstate.ShortTermState(brainstate.init.param(brainstate.init.Constant(0.), self.varshape, batch_size))

    def reset_state(self, batch_size=None):
        self.v.value = u.math.ones(self.varshape) * self.v_rest
        self.g.value = u.math.zeros(self.varshape) * u.mV
        self.t_ref.value = u.math.full(self.varshape, -1e7 * u.ms)

    def get_spike(self, v=None):
        v = self.v.value if v is None else v
        return self.spk_fun((v - self.v_th) / (20. * u.mV))

    def update(self, x):
        t = brainstate.environ.get('t')

        # numerical integration
        dv = lambda v, t, g: (self.v_rest - v + g) / self.tau_m
        dg = lambda g, t: -g / self.tau_syn
        v = brainstate.nn.exp_euler_step(dv, self.v.value, t, self.g.value)
        g = brainstate.nn.exp_euler_step(dg, self.g.value, t)
        g += x  # external input current
        v = self.sum_delta_inputs(v)

        # refractory period
        ref = (t - self.t_ref.value) <= self.tau_ref
        v = u.math.where(ref, self.v.value, v)
        g = u.math.where(ref, self.g.value, g)

        # spikes
        spk = self.get_spike(v)
        self.spike_count.value += spk

        # update states
        self.v.value = spk * (self.v_reset - v) + v
        self.g.value = g - spk * g
        self.t_ref.value = u.math.where(spk, t, self.t_ref.value)
        return spk


class Network(brainstate.nn.Module):
    def __init__(
        self,
        path_neu: Union[Path, str],
        path_syn: Union[Path, str],
        neuron_to_excite: Sequence[Dict] = (),
        neuron_to_inhibit: Sequence[Dict] = (),
        w_syn: u.Quantity = 0.275 * u.mV,
        neuron_params: Dict = None,
    ):
        super().__init__()

        self.path_neu = Path(path_neu)
        self.path_syn = Path(path_syn)

        # neuron ids
        flywire_ids = pd.read_csv(self.path_neu, index_col=0)
        self.n_neuron = len(flywire_ids)
        self._flyid2i = {f: i for i, f in enumerate(flywire_ids.index)}
        self._i2flyid = {i: f for i, f in enumerate(flywire_ids.index)}

        # neurons ids to excite
        self.neuron_to_excite = tuple([
            {
                'indices': (
                    u.math.asarray(exc['indices'])
                    if 'indices' in exc else
                    np.asarray([self._flyid2i[id_] for id_ in exc['ids']])
                ),  # neuron ids to excite
                'rate': exc['rate'],  # Poisson rate
                'w_syn': w_syn.clone() * 250,  # synaptic weight, 250 is the scaling factor for Poisson synapse
            }
            for exc in neuron_to_excite
        ])
        # neuron ids to inhibit
        neuron_to_inhibit = [
            (
                inh['indices']
                if 'indices' in inh else
                [self._flyid2i[i] for i in inh['ids']]
            )
            for inh in neuron_to_inhibit
        ]
        self.neuron_mask = jax.numpy.ones(self.n_neuron, dtype=bool)
        if len(neuron_to_inhibit) > 0:
            for indices in neuron_to_inhibit:
                self.neuron_mask = self.neuron_mask.at[indices].set(False)

        # neuronal and synaptic dynamics
        neuron_params = neuron_params or {}
        self.pop = Population(size=self.n_neuron, **neuron_params)

        # delay for changes in post-synaptic neuron
        # Paul et al 2015 doi: https://doi.org/10.3389/fncel.2015.00029
        self.delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct(self.pop.varshape, brainstate.environ.dftype()),
            entries={'D': 1.8 * u.ms}
        )

        # synapses: CSR connectivity matrix
        flywire_conns = pd.read_parquet(self.path_syn)
        i_pre = flywire_conns.loc[:, 'Presynaptic_Index'].values
        i_post = flywire_conns.loc[:, 'Postsynaptic_Index'].values
        weight = flywire_conns.loc[:, 'Excitatory x Connectivity'].values
        sort_indices = np.argsort(i_pre)
        i_pre = i_pre[sort_indices]
        i_post = i_post[sort_indices]
        weight = weight[sort_indices]

        values, counts = np.unique(i_pre, return_counts=True)
        indptr = np.zeros(self.n_neuron + 1, dtype=int)
        indptr[values + 1] = counts
        indptr = np.cumsum(indptr)
        indices = i_post

        self.conn = brainstate.nn.SparseLinear(
            brainevent.CSR(
                (weight * w_syn, indices, indptr),
                shape=(self.n_neuron, self.n_neuron)
            ),
            b_init=None,
        )

        # Poisson input
        for exc in self.neuron_to_excite:
            self.pop.tau_ref[exc['indices']] = 0. * u.ms  # set refractory period to 0.5 ms

    def update(self, *args, **kwargs):
        # excite neurons
        for exc in self.neuron_to_excite:
            brainstate.nn.poisson_input(
                freq=exc['rate'],
                num_input=1,
                weight=exc['w_syn'],
                target=self.pop.v,
                indices=exc['indices'],
            )

        # delayed spikes
        pre_spk = self.delay.at('D')

        # inhibit neurons
        if len(self.neuron_mask):
            pre_spk = pre_spk * self.neuron_mask

        # compute recurrent connections and update neurons
        spk = self.pop(self.conn(brainevent.EventArray(pre_spk)))

        # update delay spikes
        self.delay.update(spk)
        return spk

    def step_run(self, i, ret_val: str = 'none'):
        with brainstate.environ.context(t=i * brainstate.environ.get_dt(), i=i):
            spk = self.update()
            if ret_val == 'spike':
                return spk
            elif ret_val == 'voltage':
                return self.pop.v.value
            elif ret_val == 'conductance':
                return self.pop.g.value
            elif ret_val == 'spike_count':
                return self.pop.spike_count.value
            elif ret_val == 'none':
                return None
            else:
                raise ValueError('ret_val must be "spike", "voltage", "conductance", or "spike_count"')


def run_one_exp(
    neurons_to_excite: Sequence[dict],
    neuron_to_inhibit: Sequence = (),
    duration: brainstate.typing.ArrayLike = 1000 * u.ms,
    dt: brainstate.typing.ArrayLike = 0.1 * u.ms,
    pbar: int = 1000,
):
    with brainstate.environ.context(dt=dt):
        indices = np.arange(int(duration / brainstate.environ.get_dt()))

        assert isinstance(neurons_to_excite, (tuple, list)), 'neurons_to_excite must be a list or tuple'
        for exc in neurons_to_excite:
            assert isinstance(exc, dict), 'neurons_to_excite must be a list of dictionaries'
            assert 'ids' in exc or 'indices' in exc, 'neurons_to_excite must have a key "ids" or "indices"'
            assert 'rate' in exc, 'neurons_to_excite must have a key "rate"'
        assert isinstance(neuron_to_inhibit, (tuple, list)), 'neuron_to_inhibit must be a list or tuple'

        net = Network(
            path_neu='./2023_03_23_completeness_630_final.csv',
            path_syn='./2023_03_23_connectivity_630_final.parquet',
            neuron_to_excite=neurons_to_excite,
            neuron_to_inhibit=neuron_to_inhibit,
        )
        brainstate.nn.init_all_states(net)
        brainstate.compile.for_loop(
            net.step_run,
            indices,
            pbar=pbar
        )
        return net.pop.spike_count.value / duration.to_decimal(u.second)


def flywire_ids(neu_path: str | Path = './2023_03_23_completeness_630_final.csv'):
    df = pd.read_csv(neu_path, index_col=0)
    return np.asarray(df.index)


def ndarray_to_dataframe(
    arr: jax.Array | np.ndarray,
    dim_names: Sequence,
    axis_names: Optional[Sequence[str]] = None,
    column_axis: Optional[int] = None,
    row_axis: Optional[int] = None
):
    """
    Convert a multidimensional NumPy array to a two-dimensional Pandas DataFrame.

    Parameters:
    - arr (numpy.ndarray): Input multi-dimensional array.
    - dim_names (list of list of str): List of names for each dimension.
    - column_axis (int): Specifies which dimension to use as DataFrame columns.
    - row_axis (int): Specifies which dimension to use as DataFrame rows, the remaining
        dimensions will be flattened into columns.

    Returns:
    - pd.DataFrame: Converted two-dimensional DataFrame.
    """

    # Validate input
    if not isinstance(arr, (np.ndarray, jax.Array)):
        raise TypeError("Input must be a NumPy array.")

    if not isinstance(dim_names, list) or len(dim_names) != arr.ndim:
        raise ValueError("dim_names must be a list with the same length "
                         "as the number of dimensions of the array.")
    for i, names in enumerate(dim_names):
        if len(names) != arr.shape[i]:
            raise ValueError(f"Length of dim_names[{i}] must be equal to "
                             f"the size of the corresponding dimension.")

    # Get indices for all dimensions
    axes = list(range(arr.ndim))

    if column_axis is None:
        assert row_axis is not None, 'row_axis must be specified if column_axis is None'

        if not isinstance(row_axis, int):
            if row_axis < 0:
                row_axis = row_axis + arr.ndim
            if not (0 <= row_axis < arr.ndim):
                raise ValueError("row_axis must be a valid dimension index.")

        # Separate row dimension and column dimensions
        row_dim = row_axis
        column_dims = axes[:row_dim] + axes[row_dim + 1:]

        # Get row labels
        row_labels = dim_names[row_dim]

        # Get labels for each column dimension
        column_dim_names = [dim_names[dim] for dim in column_dims]

        # Generate all combinations of column labels
        if column_dim_names:
            column_tuples = list(itertools.product(*column_dim_names))
            # Convert tuples to string labels, e.g., ('Feature_X', 'Measurement_I') -> 'Feature_X_Measurement_I'
            column_labels = ['_'.join(col) for col in column_tuples]
        else:
            # If there is only one dimension as rows, there are no columns
            column_labels = []

        # Rearrange array axes to move row axis to the first position,
        # keeping the order of the remaining dimensions
        reordered_axes = [row_dim] + column_dims
        arr_reordered = np.transpose(arr, axes=reordered_axes)

        # Get new shape
        new_shape = arr_reordered.shape
        num_rows = new_shape[0]
        num_cols = np.prod(new_shape[1:]).astype(int) if len(new_shape) > 1 else 1

        # Flatten all dimensions except the row dimension into columns
        arr_flat = arr_reordered.reshape(num_rows, num_cols)

        # If there are multiple column dimensions, generate combined
        # column labels; otherwise, use a single column label
        if len(column_dims) > 0:
            if len(column_dims) == 1:
                # Only one column dimension, directly use its labels
                column_labels = dim_names[column_dims[0]]
            else:
                # Multiple column dimensions, combine labels
                column_labels = [
                    '_'.join([dim_names[dim][i] for dim, i in zip(column_dims, idx)])
                    for idx in itertools.product(*[range(len(dim_names[dim])) for dim in column_dims])
                ]

        # Create row index
        index = pd.Index(row_labels, name=dim_names[row_dim][0] if len(dim_names[row_dim]) > 0 else f"dim_{row_dim}")

        # Create column index
        if column_labels:
            columns = column_labels
        else:
            # If there are no column labels, create a default single column
            columns = ['Value']
            arr_flat = arr_flat.flatten()

    else:
        assert row_axis is None, 'row_axis must be None if column_axis is specified'

        if not isinstance(column_axis, int):
            if column_axis < 0:
                column_axis = column_axis + arr.ndim
            if not (0 <= column_axis < arr.ndim):
                raise ValueError("column_axis must be a valid dimension index.")

        # Separate column dimension and row dimensions
        column_dim = column_axis
        row_dims = axes[:column_dim] + axes[column_dim + 1:]

        # Get column labels
        columns = dim_names[column_dim]

        # Get labels for each row dimension
        row_dim_names = [dim_names[dim] for dim in row_dims]

        # Generate all combinations of row labels
        row_tuples = list(itertools.product(*row_dim_names))

        # Rearrange array axes to move column axis to the last position
        reordered_axes = row_dims + [column_dim]
        arr_reordered = arr.transpose(reordered_axes)

        # Calculate number of rows and columns
        num_rows = arr_reordered.shape[:-1]
        num_cols = arr_reordered.shape[-1]

        # Flatten all dimensions except the column dimension
        arr_flat = arr_reordered.reshape(-1, num_cols)

        # Create row index, use MultiIndex if there are multiple row dimensions
        if len(row_dims) > 1:
            if axis_names is None:
                names = [f"dim_{dim}" for dim in row_dims]
            else:
                assert len(axis_names) == arr.ndim, 'axis_names must have the same length as the number of dimensions'
                names = [axis_names[dim] for dim in range(arr.ndim)]
            index = pd.MultiIndex.from_tuples(row_tuples, names=names)
        else:
            index = pd.Index(row_tuples, )

    # Create DataFrame
    df = pd.DataFrame(arr_flat, index=index, columns=columns)
    return df
