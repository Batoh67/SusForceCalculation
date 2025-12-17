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
    from Export import export_to_excel

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

    # Run static force calculation
    SForces = StaticSuspensionForces(
        front_contact_patch,
        front_forces,
        rear_contact_patch,
        rear_forces,
        suspension
    )
    
    # Print results
    suspension.print_all_geometry()
    suspension.print_all_forces()
    
    #Export forces to .xlsx
    labels = ["LW Front Link", "LW Rear Link","UW Front Link",
                "UW Rear Link", "Pushrod", "Tierod"]
    export_to_excel(SForces.front_FOut, SForces.rear_FOut,
                    row_labels=labels, filename="usage.xlsx")

# Output

For each axle and load case, the solver outputs:

Suspension Geometry

Axial forces in suspension members [N]

X, Y, Z force components[N]

Convention: Positive force = compression
Negative force = tension

.xlsx file
# Testing

Tests are written using pytest and validate:

Static force equilibrium

Static moment equilibrium

Full-rank equilibrium matrix

Run Tests

    pytest Test.py

# Dependencies

Python 3.9+

NumPy

Pytest (for testing)

pandas (for exporting to excel)
# Limitations

Static analysis only (no dynamics or compliance)

Assumes rigid links

Fixed file format for geometry input