import jax.numpy as jnp
import flax.linen as nn
from jax.ops import segment_sum
from typing import Any, Callable, Dict
from mlff.nn.base.sub_module import BaseSubModule
import sys

Array = Any

class EnergySparse(BaseSubModule):
    prop_keys: Dict
    zmax: int = 118
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    learn_atomic_type_scales: bool = False
    learn_atomic_type_shifts: bool = False
    zbl_repulsion: bool = False
    zbl_repulsion_shift: float = 0.
    output_is_zero_at_init: bool = True
    output_convention: str = 'per_structure'
    module_name: str = 'energy'

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        #TODO: Update docstring
        Args:
            inputs (Dict):
                x (Array): Node features, (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, (num_nodes)
                batch_segments (Array): Batch segments, (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                graph_mask (Array): Graph mask, (num_graphs)
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)

        num_graphs = len(graph_mask)
        if self.learn_atomic_type_shifts:
            energy_offset = self.param(
                'energy_offset',
                nn.initializers.zeros_init(),
                (self.zmax + 1, )
            )[atomic_numbers]  # (num_nodes)
        else:
            energy_offset = jnp.zeros((1,), dtype=x.dtype)

        if self.learn_atomic_type_scales:
            atomic_scales = self.param(
                'atomic_scales',
                nn.initializers.ones_init(),
                (self.zmax + 1,)
            )[atomic_numbers]  # (num_nodes)
        else:
            atomic_scales = jnp.ones((1, ), dtype=x.dtype)

        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='energy_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            atomic_energy = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='energy_dense_final'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            atomic_energy = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='energy_dense_final'
            )(x).squeeze(axis=-1)  # (num_nodes)

        atomic_energy = atomic_energy * atomic_scales
        atomic_energy += energy_offset  # (num_nodes)

        atomic_energy = jnp.where(node_mask, atomic_energy, jnp.asarray(0., dtype=atomic_energy.dtype))  # (num_nodes)

        if self.zbl_repulsion:
            raise NotImplementedError('ZBL Repulsion for sparse model not implemented yet.')

        if self.output_convention == 'per_structure':
            energy = segment_sum(
                atomic_energy,
                segment_ids=batch_segments,
                num_segments=num_graphs
            )  # (num_graphs)
            energy = jnp.where(graph_mask, energy, jnp.asarray(0., dtype=energy.dtype))

            return dict(energy=energy)
        elif self.output_convention == 'per_atom':
            energy = atomic_energy  # (num_nodes)

            return dict(energy=energy)
        else:
            raise ValueError(
                f'{self.output_convention} is invalid argument for attribute `output_convention`.'
            )

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'zmax': self.zmax,
                                   'output_is_zero_at_init': self.output_is_zero_at_init,
                                   'output_convention': self.output_convention,
                                   'zbl_repulsion': self.zbl_repulsion,
                                   'zbl_repulsion_shift': self.zbl_repulsion_shift,
                                   'prop_keys': self.prop_keys}
                }

class DipoleSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'dipole'

    def setup(self):
        # self.partial_charge_key = self.prop_keys.get('partial_charge')
        # self.atomic_type_key = self.prop_keys.get('atomic_type')
        # self.total_charge_key = self.prop_keys.get('total_charge')

        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()        

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        #TODO: Update docstring
        Predict partial charges, from atom-wise features `x`, atomic types `atomic_numbers` and the total charge of the system `total_charge`.

        Args:
            inputs (Dict):
                x (Array): Node features, (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, (num_nodes)
                total_charge (Array): Total charge, shape: (1)
                batch_segments (Array): Batch segments, (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                graph_mask (Array): Graph mask, (num_graphs)
            *args ():
            **kwargs ():

        #Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n,1)

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        positions = inputs['positions'] # (num_nodes, 3)
        total_charge = inputs['total_charge'] # (num_graphs) #TODO: read total_charge from loaded graph
        center_of_mass = inputs['center_of_mass'] # (num_graphs, 3)

        num_graphs = len(graph_mask)
        num_nodes = len(node_mask)

        #q_ - element-dependent bias
        q_ = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)
        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='charge_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final'
            )(x).squeeze(axis=-1)  # (num_nodes)

        #mask x_q from padding graph
        x_q = jnp.where(node_mask, x_ + q_, jnp.asarray(0., dtype=x_.dtype))  # (num_nodes)
        # x_q = safe_scale(x_ + q_, scale=point_mask)  # shape: (n)

        total_charge_predicted = segment_sum(
            x_q,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs)

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts = True, size=num_graphs)

        charge_conservation = (1 / number_of_atoms_in_molecule) * (total_charge - total_charge_predicted)
        partial_charges = x_q + jnp.repeat(charge_conservation, number_of_atoms_in_molecule, total_repeat_length = num_nodes)   # shape: (num_nodes)
        
        #Shift origin to center of mass to get consistent dipole moment for charged molecules
        center_of_mass_expanded = jnp.repeat(center_of_mass, number_of_atoms_in_molecule, axis = 0, total_repeat_length = num_nodes) # shape: (num_nodes, 3)
        positions_shifted = positions - center_of_mass_expanded

        #mu = positions * charges / (1e-11 / c / e)  # [num_nodes, 3]
        mu_i = positions_shifted * partial_charges[:, None] #(512,3) * (512,)

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)
        dipole = jnp.linalg.norm(dipole, axis = 1)  # (num_graphs)
        dipole = jnp.where(graph_mask, dipole, jnp.asarray(0., dtype=dipole.dtype))

        return dict(dipole=dipole)
    
    # def reset_output_convention(self, output_convention):
    #     self.output_convention = output_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'output_is_zero_at_init': self.output_is_zero_at_init,
                                   'prop_keys': self.prop_keys}
                                   #'zmax': self.zmax,
                                #    'output_convention': self.output_convention,
                                #    'zbl_repulsion': self.zbl_repulsion,
                                #    'zbl_repulsion_shift': self.zbl_repulsion_shift,           
                }    
    
class HirshfeldSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name = 'hirshfeld_ratios'

    def setup(self):
        # self.atomic_type_key = self.prop_keys.get(pn.atomic_type)
        # self.hirshfeld_volume_ratio_key = self.prop_keys.get(pn.hirshfeld_volume_ratio)

        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal() 

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        #TODO: Update docstring
        Predict Hirshfeld volumes from atom-wise features `x` and atomic types `z`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
            *args ():
            **kwargs ():

        Returns: Dictionary of form {'v_eff': Array}, where Array are the predicted Hirshfeld ratios

        """
        # point_mask = inputs['point_mask']
        x = inputs['x']  # (num_nodes, num_features)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)

        F = x.shape[-1]

        v_shift = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)
        q = nn.Embed(num_embeddings=100, features=int(F/2))(atomic_numbers)  # shape: (n,F/2)
        
        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='hirshfeld_ratios_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            k = nn.Dense(
                int(F/2),
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(y) # (num_nodes)
        else:
            k = nn.Dense(
                int(F/2),
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(x) # (num_nodes)

        qk = (q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1) 
        q_x_k = jnp.where(node_mask, qk, jnp.asarray(0., dtype=k.dtype))

        v_eff = v_shift + q_x_k  # shape: (n)
        v_eff = jnp.where(node_mask, v_eff, jnp.asarray(0., dtype=v_eff.dtype))

        return dict(hirshfeld_ratios=v_eff)
    
class DipoleVecSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'dipole_vec'

    def setup(self):
        # self.partial_charge_key = self.prop_keys.get('partial_charge')
        # self.atomic_type_key = self.prop_keys.get('atomic_type')
        # self.total_charge_key = self.prop_keys.get('total_charge')

        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()        

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        #TODO: Update docstring
        Predict partial charges, from atom-wise features `x`, atomic types `atomic_numbers` and the total charge of the system `total_charge`.

        Args:
            inputs (Dict):
                x (Array): Node features, (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, (num_nodes)
                total_charge (Array): Total charge, shape: (1)
                batch_segments (Array): Batch segments, (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                graph_mask (Array): Graph mask, (num_graphs)
            *args ():
            **kwargs ():

        #Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n,1)

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        graph_mask_expanded = inputs['graph_mask_expanded']
        positions = inputs['positions'] # (num_nodes, 3)
        total_charge = inputs['total_charge'] # (num_graphs)
        center_of_mass = inputs['center_of_mass'] # (num_graphs, 3)

        num_graphs = len(graph_mask)
        num_nodes = len(node_mask)

        #q_ - element-dependent bias
        q_ = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)
        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='charge_dense_regression_vec'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final_vec'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final_vec'
            )(x).squeeze(axis=-1)  # (num_nodes)

        #mask x_q from padding graph
        x_q = jnp.where(node_mask, x_ + q_, jnp.asarray(0., dtype=x_.dtype))  # (num_nodes)
        # x_q = safe_scale(x_ + q_, scale=point_mask)  # shape: (n)

        total_charge_predicted = segment_sum(
            x_q,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs)

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts = True, size=num_graphs)

        charge_conservation = (1 / number_of_atoms_in_molecule) * (total_charge - total_charge_predicted)
        partial_charges = x_q + jnp.repeat(charge_conservation, number_of_atoms_in_molecule, total_repeat_length = num_nodes)   # shape: (num_nodes)
        #TODO: Check whether partial charges make sense

        #Shift origin to center of mass to get consistent dipole moment for charged molecules
        center_of_mass_expanded = jnp.repeat(center_of_mass, number_of_atoms_in_molecule, axis = 0, total_repeat_length = num_nodes) # shape: (num_nodes, 3)
        positions_shifted = positions - center_of_mass_expanded
        
        #mu = positions * charges / (1e-11 / c / e)  # [num_nodes, 3]
        mu_i = positions_shifted * partial_charges[:, None] #(512,3) * (512,)

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)

        dipole_vec = jnp.where(graph_mask_expanded, dipole, jnp.asarray(0., dtype=dipole.dtype))

        return dict(dipole_vec=dipole_vec)