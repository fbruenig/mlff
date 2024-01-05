from ase import units
import jax
from mlff.nn import SO3kratesSparse
from mlff.training.run_sparse import run_training_sparse
from mlff.utils import training_utils
from mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse
from mlff.data import AseDataLoaderSparse
from mlff.data import transformations
import numpy as np
import optax
from pathlib import Path
import pkg_resources

import portpicker


def test_run_training_sparse():
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

    filename = 'test_data/data_set.xyz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = AseDataLoaderSparse(input_file=f)
    all_data = loader.load_all(cutoff=4.)

    num_train = 30
    num_valid = 20

    energy_unit = units.kcal / units.mol
    length_unit = units.Angstrom

    numpy_rng = np.random.RandomState(0)
    numpy_rng.shuffle(all_data)

    energy_mean = transformations.calculate_energy_mean(all_data[:num_train]) * energy_unit
    num_nodes = transformations.calculate_average_number_of_nodes(all_data[:num_train])
    energy_shifts = {a: energy_mean / num_nodes for a in range(119)}

    training_data = transformations.subtract_atomic_energy_shifts(
        transformations.unit_conversion(
            all_data[:num_train],
            energy_unit=energy_unit,
            length_unit=length_unit
        ),
        atomic_energy_shifts=energy_shifts
    )

    validation_data = transformations.subtract_atomic_energy_shifts(
        transformations.unit_conversion(
            all_data[num_train:(num_train + num_valid)],
            energy_unit=energy_unit,
            length_unit=length_unit
        ),
        atomic_energy_shifts=energy_shifts
    )

    opt = optax.adam(learning_rate=1e-3)

    so3k = SO3kratesSparse(
        num_features=32,
        num_features_head=8,
        num_heads=2,
        degrees=[1, 2],
    )

    loss_fn = training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(so3k),
        weights={'energy': 0.001, 'forces': 0.999}
    )

    workdir = Path('_test_run_training_sparse').resolve().absolute()
    workdir.mkdir(exist_ok=True)

    run_training_sparse(
        model=so3k,
        optimizer=opt,
        loss_fn=loss_fn,
        graph_to_batch_fn=training_utils.graph_to_batch_fn,
        batch_max_num_edges=500,
        batch_max_num_nodes=50,
        batch_max_num_graphs=2,
        training_data=list(training_data),
        validation_data=list(validation_data),
        ckpt_dir=workdir / 'checkpoints',
        allow_restart=False
    )


def test_remove_dirs():
    try:
        import shutil
        shutil.rmtree('_test_run_training_sparse')
        shutil.rmtree('wandb')
    except FileNotFoundError:
        pass

