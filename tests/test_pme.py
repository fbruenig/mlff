from mlff.md.calculator_sparse import matrix_to_voigt
import numpy as np
import numpy.testing as npt
from ase.io import read
from so3lr import So3lrCalculator
import jax
from jax import numpy as jnp

from so3lr import So3lrPotential, to_jax_md
import jax_md

from functools import partial

import pytest
from ase import Atoms
from ase.io import read

import math
import os


print(f'Jax backend: {jax.default_backend()}')

jax.config.update('jax_enable_x64', True)
DTYPE = jnp.float64


def generate_orthogonal_transformations():
    # Generate rotation matrix along x-axis
    def rot_x(phi):
        rot = np.zeros((3, 3), dtype=DTYPE)
        rot[0, 0] = rot[1, 1] = math.cos(phi)
        rot[0, 1] = -math.sin(phi)
        rot[1, 0] = math.sin(phi)
        rot[2, 2] = 1.0

        return rot

    # Generate rotation matrix along z-axis
    def rot_z(theta):
        rot = np.zeros((3, 3), dtype=DTYPE)
        rot[0, 0] = rot[2, 2] = math.cos(theta)
        rot[0, 2] = math.sin(theta)
        rot[2, 0] = -math.sin(theta)
        rot[1, 1] = 1.0

        return rot

    # Generate a few rotation matrices
    # rot_1 = rot_z(0.987654)
    # rot_2 = rot_z(1.23456) @ rot_x(0.82321)
    rot_1 = rot_z(1.0)
    rot_2 = rot_z(1.0) @ rot_x(1.0)
    rot_none = np.eye(3, dtype=DTYPE)
    # transformations = [rot_1, rot_2]
    transformations = [rot_none]

    # make sure that the generated transformations are indeed orthogonal
    for q in transformations:
        id = np.eye(3, dtype=DTYPE)
        id_2 = q.T @ q
        np.testing.assert_allclose(id, id_2, atol=1e-15, rtol=1e-15)

    return transformations


@pytest.mark.parametrize("sr_cutoff", [5.54, 6.01])
@pytest.mark.parametrize("frame_index", [0, 1, 2])
@pytest.mark.parametrize("scaling_factor", [0.4325, 1.3353610])
@pytest.mark.parametrize("ortho", generate_orthogonal_transformations())
@pytest.mark.parametrize("calc_name", ["ewald", "pme"])
@pytest.mark.parametrize("backend", ["ase", "jax-md"])
@pytest.mark.parametrize("fractional_coordinates", [True, False])
def test_random_structure(
    sr_cutoff, frame_index, scaling_factor, ortho, calc_name, backend, fractional_coordinates,
):
    """
    Verify that energy, forces and stress agree with GROMACS.

    Structures consisting of 4 Na and 4 Cl atoms placed randomly in cubic cells of
    varying sizes.

    GROMACS values are computed with SPME and parameters as defined in the manual:
    https://manual.gromacs.org/documentation/current/user-guide/mdp-options.html#ewald
    """
    if fractional_coordinates == True and backend == "ase":
        pytest.skip()

    # coulombtype = PME fourierspacing = 0.01 ; 1/nm
    # pme_order = 8
    # rcoulomb = 0.3  ; nm
    from jaxpme import prefactors
    from jaxpme.kspace import get_kgrid_mesh, get_kgrid_ewald
    import jaxpme

    package_path = os.path.dirname(os.path.abspath(jaxpme.__file__))
    struc_path = package_path+"/../tests/reference_structures/"
    frame = read(os.path.join(
        struc_path, "coulomb_test_frames.xyz"), frame_index)

    energy_target = jnp.array(
        frame.get_potential_energy(), dtype=DTYPE) / scaling_factor
    forces_target = jnp.array(
        frame.get_forces(), dtype=DTYPE) / scaling_factor**2
    stress_target = (
        jnp.array(frame.get_stress(
            voigt=False, include_ideal_gas=False), dtype=DTYPE)
        / scaling_factor
    )
    stress_target *= 2.0  # convert from GROMACS "virial"

    positions = scaling_factor * \
        (jnp.array(frame.positions, dtype=DTYPE) @ ortho)

    cell = scaling_factor * \
        jnp.array(np.array(frame.cell), dtype=DTYPE) @ ortho
    charges = jnp.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=DTYPE)
    sr_cutoff = scaling_factor * sr_cutoff
    smearing = sr_cutoff / 6.0

    atoms = Atoms(positions=positions, cell=cell, pbc=True)

    print("Atoms:", atoms)

    if calc_name == "ewald":

        rtol_e = 2e-5
        rtol_f = 3.5e-2

        k_spacing = jnp.array([smearing / 2])
        atoms.info.update(k_grid=get_kgrid_ewald(
            jnp.array(atoms.get_cell()), k_spacing))
        atoms.info.update(k_smearing=jnp.array([smearing]))

        coulomb_kspace_do_ewald = True

    elif calc_name == "pme":

        rtol_e = 4.5e-3
        rtol_f = 5.0e-3

        k_spacing = jnp.array([smearing / 8])
        atoms.info.update(k_grid=get_kgrid_mesh(
            jnp.array(atoms.get_cell()), k_spacing))
        atoms.info.update(k_smearing=jnp.array([smearing]))

        coulomb_kspace_do_ewald = False

    if backend == "ase":
        atoms.calc = So3lrCalculator(
            lr_cutoff=sr_cutoff,
            calculate_stress=True,
            dtype=np.float64,
            coulomb_kspace_do_ewald=coulomb_kspace_do_ewald,
        )

        volume = jnp.abs(
            jnp.dot(jnp.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]))
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        # ase stress = 1/V * virial
        stress = atoms.get_stress() * volume
    elif backend == "jax-md":
        species = jnp.array(atoms.get_atomic_numbers())
        masses = jnp.array(atoms.get_masses())
        num_atoms = len(species)
        volume = jnp.abs(
            jnp.dot(jnp.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]))
        box = jnp.diag(np.array(atoms.get_cell()))

        fractional_coordinates = True
        displacement, shift = jax_md.space.periodic_general(
            box=box, fractional_coordinates=fractional_coordinates)
        positions_init = jnp.array(atoms.get_positions(), dtype=DTYPE)
        if fractional_coordinates:
            positions_init = positions_init / box

        neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
            potential=So3lrPotential(
                dtype=DTYPE,  # or float32 for single precision
                coulomb_kspace_do_ewald=coulomb_kspace_do_ewald,
            ),
            displacement_or_metric=displacement,
            shift_fn=shift,
            box_size=box,
            species=species,
            capacity_multiplier=1.25,
            buffer_size_multiplier_sr=1.25,
            buffer_size_multiplier_lr=1.25,
            minimum_cell_size_multiplier_sr=1.0,
            # Cell list partitioning can only be applied if there is a simulation box.
            disable_cell_list=False,
            fractional_coordinates=fractional_coordinates,
        )

        # Energy function.
        energy_fn = jax.jit(partial(energy_fn, has_aux=False))
        force_fn = jax.jit(jax_md.quantity.force(energy_fn))
        # energy_fn = partial(energy_fn,has_aux=False)
        # force_fn = jax_md.quantity.force(energy_fn)

        # Initialize the short and long-range neighbor lists.
        nbrs = neighbor_fn.allocate(
            positions_init,
            box=box
        )
        nbrs_lr = neighbor_fn_lr.allocate(
            positions_init,
            box=box
        )

        energy = energy_fn(
            positions_init,
            nbrs.idx,
            nbrs_lr.idx,
            box=box,
            species=species,
            masses=masses,
            k_grid=atoms.info["k_grid"],
            k_smearing=atoms.info["k_smearing"],
        )
        forces = force_fn(
            positions_init,
            nbrs.idx,
            nbrs_lr.idx,
            box=box,
            species=species,
            masses=masses,
            k_grid=atoms.info["k_grid"],
            k_smearing=atoms.info["k_smearing"],
        )

        stress = jax_md.quantity.stress(energy_fn,
                                        position=positions_init,
                                        neighbor=nbrs.idx,
                                        neighbor_lr=nbrs_lr.idx,
                                        box=box,
                                        species=species,
                                        masses=masses,
                                        k_grid=atoms.info["k_grid"],
                                        k_smearing=atoms.info["k_smearing"])
        # jax-md stress = -1/V * virial
        stress = matrix_to_voigt(-stress) * volume

    np.testing.assert_allclose(energy, energy_target, atol=0.0, rtol=rtol_e)
    np.testing.assert_allclose(
        forces, forces_target @ ortho, atol=0.0, rtol=rtol_f)

    stress_target = jnp.einsum("ab,aA,bB->AB", stress_target, ortho, ortho)
    stress_target = matrix_to_voigt(stress_target)
    np.testing.assert_allclose(stress, stress_target, atol=0.01, rtol=0.01)
