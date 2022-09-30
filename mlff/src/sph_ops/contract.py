import os
import pkg_resources
import jax
import jax.numpy as jnp
import numpy as np
import itertools as it
from functools import partial
from typing import (Callable, Sequence)
from mlff.src.masking.mask import safe_mask

# CG_PATH = os.path.abspath('cgmatrix.npz')
indx_fn = lambda x: int((x+1)**2) if x >= 0 else 0


def load_cgmatrix():
    stream = pkg_resources.resource_stream(__name__, 'cgmatrix.npz')
    return np.load(stream)['cg']


def init_clebsch_gordan_matrix(degrees, l_out_max=None):
    """
    Initialize the Clebsch-Gordan matrix (coefficients for the Clebsch-Gordan expansion of spherical basis functions)
    for given ``degrees`` and a maximal output order ``l_out_max`` up to which the given all_degrees shall be
    expanded. Minimal output order is ``min(degrees)``.

    Args:
        degrees (List): Sequence of degrees l. The lowest order can be chosen freely. However, it should
            be noted that all following all_degrees must be exactly one order larger than the following one. E.g.
            [0,1,2,3] or [1,2,3] are valid but [0,1,3] or [0,2] are not.
        l_out_max (int): Maximal output order. Can be both, smaller or larger than maximal order in degrees.
            Defaults to the maximum value of the passed degrees.

    Returns: Clebsch-Gordan matrix,
        shape: (``(l_out_max+1)**2, (l_out_max+1)**2 - (l_in_min)**2, (l_out_max+1)**2 - (l_in_min)**2``)

    """
    if l_out_max is None:
        _l_out_max = max(degrees)
    else:
        _l_out_max = l_out_max

    l_in_max = max(degrees)
    l_in_min = min(degrees)

    offset_corr = indx_fn(l_in_min - 1)
    _cg = load_cgmatrix()
    return _cg[offset_corr:indx_fn(_l_out_max), offset_corr:indx_fn(l_in_max), offset_corr:indx_fn(l_in_max)]


def init_expansion_fn(degrees: Sequence[int], cg: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Initialize function that returns the expansion in CG coefficients for all possible combinations for given
    ``degrees`` up to arbitrary order ``l3``, which is determined by the order along the 0-th axis of CG matrix,
    stores in ``cg``. The returned function takes the outer product of different orders of the spherical basis to the
    respective expansions in arbitrary order ``l3``.

    Args:
        degrees (List): Sequence of harmonic orders. The lowest order can be chosen freely. However, it should
            be noted that all following orders must be exactly one order larger than the following one. E.g.
            [0,1,2,3] or [1,2,3] are valid but [0,1,3] or [0,2] are not.
        cg (Array): Matrix that contains the CG coefficients for given `degrees`. They can be constructed
            using the function `ramses.utils.geometric.init_clebsch_gordan_matrix`.
            shape: (``(l_out_max+1)**2, (l_out_max+1)**2 - (l_in_min)**2, (l_out_max+1)**2 - (l_in_min)**2``)

    Returns: Expansion function, which takes the outer product of spherical basis functions to the correct expansion
        for arbitrary order l3. Illustrated in more detail in the example below.

    Examples:
        >>> from ramses.utils.geometric.spherical_harmonics import init_sph_fn
        >>> from mlff.src. import coordinates_to_distance_vectors_normalized
        >>> # Calculate the normalized distance vectors given some points in Euclidean space.
        >>> n = 4
        >>> p0 = generate_points(n=n, d_min=1, d_max=3)
        >>> rij = coordinates_to_distance_vectors_normalized(p0)  # shape: (n,n,3)
        >>> l_out_max=3
        >>> all_degrees = [0,1,2]
        >>> # Construct SPH coordinates in gamma.
        >>> alpha = (jnp.eye(rij.shape[-2]) == 0)[..., None]  # shape: (n,n,1)
        >>> sph_harms = []
        >>> gamma = []
        >>> sph_fns = [init_sph_fn(l) for l in all_degrees]
        >>> for sph_fn in sph_fns:
            >>> sph_ij = sph_fn(rij)  # shape: (n,n,2l+1)
            >>> gamma += [(sph_ij*alpha).sum(axis=-2)]
        >>> gamma = jnp.concatenate(gamma, axis=-1)  # shape: (n,m_tot)
        >>> # Take outer product of SPH coordinates.
        >>> gammaxgamma = jnp.einsum('nm, nl -> nml', gamma, gamma)[:, None, :, :]  # shape: (n,1,m_in,m_in)
        >>> # Initialize the Clebsch-Gordan matrix.
        >>> cg = init_clebsch_gordan_matrix(degrees=all_degrees, l_out_max=l_out_max)
        >>> # shape: (m_out,m_in,m_in)
        >>> # Initialize the expansion_fn
        >>> expansion_fn = init_expansion_fn(degrees=all_degrees, cg=cg)
        >>> J = expansion_fn(gammaxgamma)
        >>> J.shape
        (4,16,3,3)
        >>> # One can now obtain the expansion of e.g. order l1=1 and l2=2 in the basis functions of order l3=3 as
        >>> l1 = 1
        >>> l2 = 2
        >>> l1l2_l3 = J[:, 9:, l1, l2]
        >>> l1l2_l3.shape
        (4,7)
        >>> l1 = 0
        >>> l2 = 2
        >>> # or similar the expansion of e.g. order l1=0 and l2=2 in the basis functions of order l3=2 as
        >>> l1l2_l3 = J[:, 4:9, l1, l2]
        >>> l1l2_l3.shape
        (4,5)
        >>> # the SelfMix layer without parametrization from PhisNet can then be written as. Since we now that l2 > l1,
        >>> # we just sum over the upper tri-diagonal part of the matrix.
        >>> gamma_contracted = jnp.triu(J, k=1).sum(axis=(-2, -1))  # shape: (n, m_out)
    """
    segment_ids = jnp.array(
        [y for y in it.chain(*[[n] * int(2 * degrees[n] + 1) for n in range(len(degrees))])])
    num_segments = len(degrees)
    _segment_sum = partial(jax.ops.segment_sum, segment_ids=segment_ids, num_segments=num_segments)
    _v_segment_sum_l1 = jax.vmap(jax.vmap(jax.vmap(_segment_sum)))
    _v_segment_sum_l2 = jax.vmap(jax.vmap(_segment_sum))
    _expansion_fn = jax.jit(lambda x: _v_segment_sum_l1(_v_segment_sum_l2(x * cg)))
    return _expansion_fn
