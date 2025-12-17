# Static Suspension Force Calculator

This repository contains a Python-based static force solver for vehicle suspension systems.
Given suspension geometry generated in Lotus Suspension Analysis and contact patch forces, 
it computes internal forces in suspension members (wishbones, pushrods, and tierods) using static equilibrium of forces and moments.

# Usage Example

    import numpy as np

    from ForceCalculationStatic import (
        SuspensionGeometry,
        StaticSuspensionForces
    )

    #Load suspension geometry

    suspension = SuspensionGeometry("example.shk")

    #Define contact patch positions (mm) [x,y,z]

    front_contact_patch = np.array([0, 700, -200]) 

    rear_contact_patch = np.array([1600, 700, -200])

    #Define applied forces (N) [x,y,z]
    
    front_forces = np.array([
        [0, -1000, 1000],
        [1000, 0, 1000],
    ])

    rear_forces = np.array([
        [0, -2000, 2000],
        [1000, -2000, 2000],
    ])

    #Run static force calculation
    
    StaticSuspensionForces(
        front_contact_patch,
        front_forces,
        rear_contact_patch,
        rear_forces,
        suspension
    )

    #Print results
    suspension.print_all_geometry()
    suspension.print_all_forces()

# Output

For each axle and load case, the solver outputs:

Suspension Geometry

Axial forces in suspension members [N]

X, Y, Z force components[N]

Convention:

Positive force = compression
Negative force = tension

# Testing

Tests are written using pytest and validate:

Static force equilibrium

Static moment equilibrium

Full-rank equilibrium matrix

Run Tests

    pytest Test.py
