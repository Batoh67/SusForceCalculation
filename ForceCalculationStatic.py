import numpy as np
from Export import export_to_excel

# ==========Geometry===========
class Joint:
    """Represents a 3D joint/point in suspension geometry."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Joint(x={self.x}, y={self.y}, z={self.z})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def get_coordinates(self) -> tuple[float, float, float]:
        """Return coordinates as a tuple."""
        return self.x, self.y, self.z


class TwoPointLink:  # Pushrod, TieRod
    """Represents a link between two joints in suspension geometry."""

    def __init__(self, outside_joint: Joint, inside_joint: Joint):
        self.inside_joint = inside_joint
        self.outside_joint = outside_joint

        self.unit_moment_vector = unit_moment_vector(self.inside_joint, self.outside_joint)

        self.link_force = None

        self.comp_forces = None

    def print_joints(self, name: str = "TwoPointLink") -> None:
        """Print all joint coordinates."""
        print(f"{name} Joint Coordinates:")
        print(f"  Inside Joint: {self.inside_joint}")
        print(f"  Outside Joint:  {self.outside_joint}\n")
        print(f"  Unit Moment Vector: {self.unit_moment_vector}\n")

    def force(self, force: np.ndarray) -> None:

        self.link_force = force.squeeze()

        self.comp_forces = self.link_force[:, None] * self.unit_moment_vector[None, 0:3]

    def print_forces(self, name: str = "TwoPointLink") -> None:
        """Print link forces."""
        print(f"{name} Forces:")
        fx, fy, fz = self.comp_forces.T
        print(f"Link Force {self.link_force} [N]\n")
        print(f"Link Force X {fx} [N]")
        print(f"Link Force Y {fy} [N]")
        print(f"Link Force Z {fz} [N]\n")


class Wishbone:
    """Represents a wishbone made from three joints."""

    def __init__(self, front_joint: Joint, rear_joint: Joint, outer_joint: Joint):
        self.front_joint = front_joint
        self.rear_joint = rear_joint
        self.outer_joint = outer_joint

        self.front_unit_moment_vector = unit_moment_vector(self.front_joint,
                                                           self.outer_joint)

        self.rear_unit_moment_vector = unit_moment_vector(self.rear_joint,
                                                          self.outer_joint)

        self.front_link_force = None
        self.rear_link_force = None

        self.front_comp_forces = None
        self.rear_comp_forces = None

    def force(self, front_force: np.ndarray, rear_force: np.ndarray) -> None:
        if (self.front_unit_moment_vector is None
                or self.rear_unit_moment_vector is None):
            raise RuntimeError("Unit Moment Vector not built")

        self.front_link_force = front_force.squeeze()
        self.rear_link_force = rear_force.squeeze()

        self.front_comp_forces = (self.front_link_force[:, None]
                                  * self.front_unit_moment_vector[None, 0:3])

        self.rear_comp_forces = (self.rear_link_force[:, None]
                                  * self.rear_unit_moment_vector[None, 0:3])

    def __repr__(self) -> str:
        return (f"Wishbone(front={self.front_joint}, "
                f"rear={self.rear_joint}, outer={self.outer_joint})")

    def print_joints(self, name: str = "Wishbone") -> None:
        """Print all joint coordinates."""
        print(f"{name} Joint Coordinates:")
        print(f"  Front Joint: {self.front_joint}")
        print(f"  Rear Joint:  {self.rear_joint}")
        print(f"  Outer Joint: {self.outer_joint}\n")
        print(f"  Front Unit Moment Vector: {self.front_unit_moment_vector}")
        print(f"  Rear Unit Moment Vector: {self.rear_unit_moment_vector}\n")

    def print_forces(self, name: str = "Wishbone") -> None:
        """Print all link forces."""
        print(f"{name} Forces:")
        print(f"  Front Link Force {self.front_link_force} [N]\n")
        ffx, ffy, ffz = self.front_comp_forces.T
        print(f"  Front Link Force X {ffx} [N]")
        print(f"  Front Link Force Y {ffy} [N]")
        print(f"  Front Link Force Z {ffz} [N]\n")

        print(f"  Rear Link Force {self.rear_link_force} [N]\n")
        rfx, rfy, rfz = self.rear_comp_forces.T
        print(f"  Rear Link Force X {rfx} [N]")
        print(f"  Rear Link Force Y {rfy} [N]")
        print(f"  Rear Link Force Z {rfz} [N]\n")

    def get_all_joints(self) -> tuple[Joint, Joint, Joint]:
        """Return all joints as a tuple."""
        return self.front_joint, self.rear_joint, self.outer_joint


class Upright:
    def __init__(self, upper_outer_joint: Joint, lower_outer_joint: Joint):
        self.upper_outer_joint = Joint(upper_outer_joint.x,
                                       upper_outer_joint.y,
                                       upper_outer_joint.z)

        self.lower_outer_joint = Joint(lower_outer_joint.x,
                                       lower_outer_joint.y,
                                       lower_outer_joint.z)

    def print_joints(self, name: str = "Wishbone") -> None:
        """Print all joint coordinates."""
        print(f"{name} Joint Coordinates:")
        print(f"  Upper Joint: {self.upper_outer_joint}")
        print(f"  Lower Joint:  {self.lower_outer_joint}\n")


class Axle:
    """Represents front/rear suspension geometry."""

    def __init__(self, upper_wishbone: Wishbone, lower_wishbone: Wishbone,
                 pushrod: TwoPointLink, tierod: TwoPointLink,
                 name: str = "Axle"):
        self.upper_wishbone = upper_wishbone
        self.lower_wishbone = lower_wishbone
        self.pushrod = pushrod
        self.tierod = tierod
        self.upright = Upright(upper_wishbone.outer_joint, lower_wishbone.outer_joint)
        self.name = name

    def print_geometry(self) -> None:
        """Print complete front suspension geometry."""
        print(f"=== {self.name} Suspension Geometry ===")
        self.upper_wishbone.print_joints("Upper Wishbone")
        self.lower_wishbone.print_joints("Lower Wishbone")
        self.upright.print_joints("Upright")
        self.pushrod.print_joints("Pushrod")
        self.tierod.print_joints("Tierod")

    def print_forces(self) -> None:
        """Print complete front suspension forces."""
        print(f"=== {self.name} Suspension Forces ===")
        self.upper_wishbone.print_forces("Upper Wishbone")
        self.lower_wishbone.print_forces("Lower Wishbone")
        self.pushrod.print_forces("Pushrod")
        self.tierod.print_forces("Tierod")

def unit_moment_vector(p1: Joint, p2: Joint) -> np.ndarray:
    """Create a unit moment vector from two points."""
    p1 = np.array([p1.x, p1.y, p1.z])
    p2 = np.array([p2.x, p2.y, p2.z])

    vec = p2 - p1

    magnitude = np.linalg.norm(vec)
    unit_vec = vec / magnitude

    r = p1
    moment_vec = np.cross(r, unit_vec)

    combined = np.hstack((unit_vec, moment_vec))
    return combined

# ==========Main Pipeline===========
class SuspensionGeometry:
    """Loads and builds suspension geometry."""
    def __init__(self, file):
        self.file_path = file
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            self.lines = f.read().splitlines()

        frontend_start_line, rearend_start_line = self._find_suspension_line_numbers()

        def load_geometry(start_line: int) \
                -> tuple[Wishbone, Wishbone, TwoPointLink, TwoPointLink]:

            LAYOUT = {
                "lower_wishbone": 3,
                "upper_wishbone": 3,
                "pushrod": 2,
                "tierod": 2,
            }

            # Parse all joints in one pass
            joints = {}
            current = start_line + 1

            for name, count in LAYOUT.items():
                joints[name] = [
                    self._parse_joint_from_line(current + i)
                    for i in range(count)
                ]
                current += count

            # Build components
            upper_wishbone = Wishbone(*joints["upper_wishbone"])
            lower_wishbone = Wishbone(*joints["lower_wishbone"])
            pushrod = TwoPointLink(*joints["pushrod"])
            tierod = TwoPointLink(*joints["tierod"])

            return upper_wishbone, lower_wishbone, pushrod, tierod

        # Build front suspension
        geometry = load_geometry(frontend_start_line)
        self.front = Axle(*geometry, "Front")

        # Build rear suspension
        geometry = load_geometry(rearend_start_line)
        self.rear = Axle(*geometry, "Rear")

    def _find_suspension_line_numbers(self):
        """Find line numbers for FRONT and REAR suspension sections."""
        frontend_line_number = None
        rearend_line_number = None

        for line_number, line in enumerate(self.lines, start=1):
            if "FRONT SUSPENSION" in line:
                frontend_line_number = line_number
            if "REAR SUSPENSION" in line:
                rearend_line_number = line_number

        return frontend_line_number, rearend_line_number

    def _parse_joint_from_line(self, line_number: int) -> Joint:
        """Parse x, y, z coordinates from a specific line and create a Joint."""
        coords = self.lines[line_number]
        x = float(coords[10:25])
        y = float(coords[38:52])
        z = float(coords[64:77])
        return Joint(x, y, z)

    def print_all_geometry(self) -> None:
        """Print geometry suspension information."""
        self.front.print_geometry()
        self.rear.print_geometry()

    def print_all_forces(self) -> None:
        """Print force suspension information."""
        print("========Forces========")
        print("=== Positive number means that the link is under compression ===")
        print("")
        self.front.print_forces()
        self.rear.print_forces()


class StaticSuspensionForces:
    """Based on suspension geometry and input position and forces,
       calculates forces in suspension members."""
    def __init__(self,front_contact_patch, front_contact_patch_force,
                 rear_contact_patch, rear_contact_patch_force, suspension):

        self.front_contact_patch = front_contact_patch
        self.front_contact_patch_force = front_contact_patch_force
        self.suspension = suspension

        self.front_FOut, self.front_FIn = (
            calculate_suspension_forces(axle_obj = getattr(self.suspension, "front"),
                                        contact_patch = self.front_contact_patch,
                                        contact_patch_force = self.front_contact_patch_force))



        self.rear_contact_patch = rear_contact_patch 
        self.rear_contact_patch_force = rear_contact_patch_force

        self.rear_FOut, self.rear_FIn = (
            calculate_suspension_forces(axle_obj = getattr(self.suspension, "rear"),
                                        contact_patch = self.rear_contact_patch,
                                        contact_patch_force = self.rear_contact_patch_force))



        def save_forces_to_members(axle_obj,FOut) -> None:
            axle_obj.lower_wishbone.force(FOut[:, 0], FOut[:, 1])
            axle_obj.upper_wishbone.force(FOut[:, 2], FOut[:, 3])
            axle_obj.pushrod.force(FOut[:, 4])
            axle_obj.tierod.force(FOut[:, 5])

        save_forces_to_members(getattr(self.suspension, "front"), self.front_FOut)
        save_forces_to_members(getattr(self.suspension, "rear"), self.rear_FOut)

def calculate_suspension_forces(axle_obj: Axle,
                                contact_patch: np.ndarray ,
                                contact_patch_force: np.ndarray
                                ) -> tuple[np.ndarray,np.ndarray]:

    # Build unit moment matrix
    A_base = build_unit_moment_matrix(axle_obj)

    # Determine number of force cases from input
    n_cases = contact_patch_force.shape[0]

    # Dynamically replicate A to match number of force cases
    A = np.tile(A_base[np.newaxis, :, :], (n_cases, 1, 1))

    def calculate_input_moment():
        """Calculate input moment around origin."""
        Fx_vec = np.stack([contact_patch_force[:, 0],
                           np.zeros_like(contact_patch_force[:, 0]),
                           np.zeros_like(contact_patch_force[:, 0])], axis=1)

        Fy_vec = np.stack([np.zeros_like(contact_patch_force[:, 1]),
                           contact_patch_force[:, 1],
                           np.zeros_like(contact_patch_force[:, 1])], axis=1)

        Fz_vec = np.stack([np.zeros_like(contact_patch_force[:, 2]),
                           np.zeros_like(contact_patch_force[:, 2]),
                           contact_patch_force[:, 2]], axis=1)

        MFx = np.cross(contact_patch, Fx_vec)
        MFy = np.cross(contact_patch, Fy_vec)
        MFz = np.cross(contact_patch, Fz_vec)

        M_total = MFx + MFy + MFz
        Mx, My, Mz = M_total[:, 0], M_total[:, 1], M_total[:, 2]
        return Mx, My, Mz

    Mx, My, Mz = calculate_input_moment()

    FIn = np.array([-contact_patch_force[:, 0],
                    -contact_patch_force[:, 1],
                    -contact_patch_force[:, 2], -Mx, -My, -Mz]).T
    FIn = FIn[..., np.newaxis]

    FOut = np.linalg.solve(A, FIn)
    return FOut,FIn

def build_unit_moment_matrix(axle_obj: Axle) -> np.ndarray:

    A_base = np.stack((axle_obj.lower_wishbone.front_unit_moment_vector,
                       axle_obj.lower_wishbone.rear_unit_moment_vector,
                       axle_obj.upper_wishbone.front_unit_moment_vector,
                       axle_obj.upper_wishbone.rear_unit_moment_vector,
                       axle_obj.pushrod.unit_moment_vector,
                       axle_obj.tierod.unit_moment_vector), axis=1)
    return A_base


if __name__ == "__main__":

    # Path to lotus geometry save
    file_path = "example.shk"

    suspension = SuspensionGeometry(file_path)

    # Set the front contact patch position (origin of input forces)(mm)[x,y,z]
    front_contact_patch = np.array([0, 700, -200])
    # Set front input forces (N) [x,y,z]
    front_contact_patch_force = np.array(
                [[0, -1000, 1000], [1000, 0, 1000], [-1000, 0, 1000], [0, 1000, 0]])  # [0,-1560,1874]

    # Set the rear contact patch position (origin of input forces)(mm)[x,y,z]
    rear_contact_patch = np.array([1600, 700, -200])
    # Set rear input forces (N) [x,y,z]
    rear_contact_patch_force = np.array(
                [[0, -1000, 1000], [1000, 0, 1000], [-1000, 0, 1000]])

    # Calculate the forces (static)
    SForces = StaticSuspensionForces(front_contact_patch, front_contact_patch_force,
                     rear_contact_patch, rear_contact_patch_force,suspension)

    # Print information
    suspension.print_all_geometry()
    suspension.print_all_forces()

    #Export forces to .csv
    labels = ["LW Front Link", "LW Rear Link","UW Front Link",
              "UW Rear Link", "Pushrod", "Tierod"]
    export_to_excel(SForces.front_FOut,SForces.rear_FOut,row_labels=labels)