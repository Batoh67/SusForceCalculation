import numpy as np
import pytest

from ForceCalculationStatic import Joint, TwoPointLink, Wishbone, Axle
from ForceCalculationStatic import calculate_suspension_forces,build_unit_moment_matrix

@pytest.fixture
def simple_axle():
    # Very simple symmetric geometry
    upper = Wishbone(
        Joint(-200, 400, 300),
        Joint(200, 400, 300),
        Joint(0, 600, 200),
    )

    lower = Wishbone(
        Joint(-300, 400, 0),
        Joint(300, 400, 0),
        Joint(0, 600, 0),
    )

    pushrod = TwoPointLink(
        Joint(0, 600, 200),
        Joint(0, 400, 500),
    )

    tierod = TwoPointLink(
        Joint(-400, 600, 0),
        Joint(-200, 400, 0),
    )

    return Axle(upper, lower, pushrod, tierod)

def test_static_equilibrium(simple_axle):
    contact_patch = np.array([50, 600.0, 0.0])

    contact_forces = np.array([
        [0.0, -3000.0, 4000.0],
        [-10, -3000.0, 4000.0]
    ])  # (n_cases, 3)


    FOut, FIn = calculate_suspension_forces(simple_axle,contact_patch,contact_forces)

    # print(FOut)
    # Build unit vectors for links
    A = build_unit_moment_matrix(simple_axle)


    unit_vectors = A[:3, :].T  # (6, 3)

    # Rebuild link force vectors
    link_forces = FOut * unit_vectors[None, :, :]


    # ----- FORCE EQUILIBRIUM -----
    total_link_force = link_forces.sum(axis=1)
    # print(total_link_force)
    total_force = total_link_force + contact_forces
    print(total_force)
    assert np.allclose(total_force, 0.0, atol=1e-6)

    # ----- MOMENT EQUILIBRIUM -----
    link_moments = np.cross(
        np.array([
            simple_axle.lower_wishbone.front.as_array(),
            simple_axle.lower.rear.as_array(),
            simple_axle.upper.front.as_array(),
            simple_axle.upper.rear.as_array(),
            simple_axle.pushrod.inside.as_array(),
            simple_axle.tierod.inside.as_array(),
        ]),
        unit_vectors
    )

    total_link_moment = (link_moments[None, :, :] * FOut[:, :, None]).sum(axis=1)
    contact_moment = np.cross(contact_patch, contact_forces)

    assert np.allclose(total_link_moment + contact_moment, 0.0, atol=1e-6)

def test_full_rank(simple_axle):
    A = build_unit_moment_matrix(simple_axle)
    rank = np.linalg.matrix_rank(A)
    if rank < A.shape[0]:
        raise ValueError(
            f"Equilibrium matrix is rank-deficient (rank={rank})"
        )