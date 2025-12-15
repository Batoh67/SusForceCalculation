from typing import Tuple
import numpy as np


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

    def get_coordinates(self) -> Tuple[float, float, float]:
        """Return coordinates as a tuple."""
        return self.x, self.y, self.z


class TwoPointLink:  # Pushrod, TieRod
    def __init__(self, inside_joint: Joint, outside_joint: Joint):
        self.inside_joint = inside_joint
        self.outside_joint = outside_joint

        self.unit_moment_vector = None

        self.link_force = None

        self.link_force_x = None
        self.link_force_y = None
        self.link_force_z = None

    def build_unit_moment_vector(self):
        def unit_moment_vector(p1, p2):
            p1 = np.array([p1.x, p1.y, p1.z])
            p2 = np.array([p2.x, p2.y, p2.z])

            vec = p2 - p1

            magnitude = np.linalg.norm(vec)
            unit_vec = vec / magnitude

            r = p1
            moment_vec = np.cross(r, unit_vec)

            combined = np.hstack((unit_vec, moment_vec))
            return combined

        self.unit_moment_vector = unit_moment_vector(self.inside_joint, self.outside_joint)

    def print_joints(self, name: str = "Wishbone") -> None:
        """Print all joint coordinates."""
        print(f"{name} Joint Coordinates:")
        print(f"  Inside Joint: {self.inside_joint}")
        print(f"  Outside Joint:  {self.outside_joint}")
        print(f"  Unit Moment Vector: {self.unit_moment_vector}")

    def force(self, force):
        self.link_force = force.squeeze()
        # self.link_force_x = self.link_force * self.vec[0]
        # self.link_force_y = self.link_force * self.vec[1]
        # self.link_force_z = self.link_force * self.vec[2]

    def print_forces(self, name: str) -> None:
        """Print all joint coordinates."""
        print(f"{name} Forces:")
        print(f"Link Force {self.link_force} [N]")
        # print(f"Link Force X {self.link_force_x} [N]")
        # print(f"Link Force Y {self.link_force_y} [N]")
        # print(f"Link Force Z {self.link_force_z} [N]")


class Wishbone:
    """Represents a wishbone with three joints."""

    def __init__(self, front_joint: Joint, rear_joint: Joint, outer_joint: Joint):
        self.front_joint = front_joint
        self.rear_joint = rear_joint
        self.outer_joint = outer_joint

        self.front_unit_moment_vector = None
        self.rear_unit_moment_vector = None

        self.front_link_force = None
        self.rear_link_force = None

        self.front_link_force_x = None
        self.front_link_force_y = None
        self.front_link_force_z = None

        self.rear_link_force_x = None
        self.rear_link_force_y = None
        self.rear_link_force_z = None

    def build_unit_moment_vector(self):
        def unit_moment_vector(p1, p2):
            p1 = np.array([p1.x, p1.y, p1.z])
            p2 = np.array([p2.x, p2.y, p2.z])

            self.vec = p2 - p1

            magnitude = np.linalg.norm(self.vec)
            unit_vec = self.vec / magnitude

            r = p1
            moment_vec = np.cross(r, unit_vec)

            combined = np.hstack((unit_vec, moment_vec))
            return combined

        self.front_unit_moment_vector = unit_moment_vector(self.front_joint, self.outer_joint)
        self.rear_unit_moment_vector = unit_moment_vector(self.rear_joint, self.outer_joint)

    def force(self, front_force, rear_force):
        self.front_link_force = front_force.squeeze()
        self.rear_link_force = rear_force.squeeze()

        self.front_link_force_x = self.front_link_force * self.front_unit_moment_vector[0]
        self.front_link_force_y = self.front_link_force * self.front_unit_moment_vector[1]
        self.front_link_force_z = self.front_link_force * self.front_unit_moment_vector[2]

        self.rear_link_force_x = self.rear_link_force * self.rear_unit_moment_vector[0]
        self.rear_link_force_y = self.rear_link_force * self.rear_unit_moment_vector[1]
        self.rear_link_force_z = self.rear_link_force * self.rear_unit_moment_vector[2]

    def __repr__(self) -> str:
        return (f"Wishbone(front={self.front_joint}, "
                f"rear={self.rear_joint}, outer={self.outer_joint})")

    def print_joints(self, name: str = "Wishbone") -> None:
        """Print all joint coordinates."""
        print(f"{name} Joint Coordinates:")
        print(f"  Front Joint: {self.front_joint}")
        print(f"  Rear Joint:  {self.rear_joint}")
        print(f"  Outer Joint: {self.outer_joint}")
        print(f"  Front Unit Moment Vector: {self.front_unit_moment_vector}")
        print(f"  Rear Unit Moment Vector: {self.rear_unit_moment_vector}")

    def print_forces(self, name: str = "Wishbone") -> None:
        """Print all joint coordinates."""
        print(f"{name} Forces:")
        print(f"  Front Link Force {self.front_link_force} [N]")
        print(f"  Front Link Force X {self.front_link_force_x} [N]")
        print(f"  Front Link Force Y {self.front_link_force_y} [N]")
        print(f"  Front Link Force Z {self.front_link_force_z} [N]")
        print(f"  Rear Link Force {self.rear_link_force} [N]")
        print(f"  Rear Link Force X {self.rear_link_force_x} [N]")
        print(f"  Rear Link Force Y {self.rear_link_force_y} [N]")
        print(f"  Rear Link Force Z {self.rear_link_force_z} [N]")

    def get_all_joints(self) -> Tuple[Joint, Joint, Joint]:
        """Return all joints as a tuple."""
        return (self.front_joint, self.rear_joint, self.outer_joint)


class Upright:
    def __init__(self, upper_outer_joint: Joint, lower_outer_joint: Joint):
        self.upper_outer_joint = Joint(upper_outer_joint.x, upper_outer_joint.y, upper_outer_joint.z)
        self.lower_outer_joint = Joint(lower_outer_joint.x, lower_outer_joint.y, lower_outer_joint.z)

    def print_joints(self, name: str = "Wishbone") -> None:
        """Print all joint coordinates."""
        print(f"{name} Joint Coordinates:")
        print(f"  Upper Joint: {self.upper_outer_joint}")
        print(f"  Lower Joint:  {self.lower_outer_joint}")


class Front:
    """Represents front suspension geometry."""

    def __init__(self, upper_wishbone: Wishbone, lower_wishbone: Wishbone, pushrod: TwoPointLink, tierod: TwoPointLink,
                 end):
        self.upper_wishbone = upper_wishbone
        self.lower_wishbone = lower_wishbone
        self.pushrod = pushrod
        self.tierod = tierod
        # upright = Upright(upper_wishbone.outer_joint,lower_wishbone.outer_joint)
        self.upright = Upright(upper_wishbone.outer_joint, lower_wishbone.outer_joint)
        self.end = end

    def print_geometry(self) -> None:
        """Print complete front suspension geometry."""
        print("=== Front Suspension Geometry ===")
        self.upper_wishbone.print_joints("Upper Wishbone")
        print()
        self.lower_wishbone.print_joints("Lower Wishbone")
        print()
        self.upright.print_joints("Upright")
        print()
        self.pushrod.print_joints("Pushrod")
        print()
        self.tierod.print_joints("Tierod")

    def print_forces(self) -> None:
        """Print complete front suspension geometry."""
        print(f"=== {self.end} Suspension Forces ===")
        self.upper_wishbone.print_forces("Upper Wishbone")
        print()
        self.lower_wishbone.print_forces("Lower Wishbone")
        print()
        self.pushrod.print_forces("Pushrod")
        print()
        self.tierod.print_forces("Tierod")


# ==========Main Pipeline===========
class SuspensionGeometry:

    def __init__(self, file):
        self.file_path = file
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            self.lines = f.read().splitlines()

        frontend_start_line, rearend_start_line = self._find_suspension_line_numbers()

        def load_geometry(start_line):

            # # Parse lower wishbone joints
            # lower_front_joint = self._parse_joint_from_line(line_number + 1)
            # lower_rear_joint = self._parse_joint_from_line(line_number + 2)
            # lower_outer_joint = self._parse_joint_from_line(line_number + 3)
            # lower_wishbone = Wishbone(lower_front_joint, lower_rear_joint, lower_outer_joint)
            #
            # # Parse upper wishbone joints
            # upper_front_joint = self._parse_joint_from_line(line_number + 4)
            # upper_rear_joint = self._parse_joint_from_line(line_number + 5)
            # upper_outer_joint = self._parse_joint_from_line(line_number + 6)
            # upper_wishbone = Wishbone(upper_front_joint, upper_rear_joint, upper_outer_joint)
            #
            # lower_joint = self._parse_joint_from_line(line_number + 7)
            # upper_joint = self._parse_joint_from_line(line_number + 8)
            # pushrod = TwoPointLink(upper_joint, lower_joint)
            #
            # outer_joint = self._parse_joint_from_line(line_number + 9)
            # inside_joint = self._parse_joint_from_line(line_number + 10)
            # tierod = TwoPointLink(inside_joint, outer_joint)

            # Number of joints each component needs
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
            upper_wishbone = Wishbone(*joints["lower_wishbone"])
            lower_wishbone = Wishbone(*joints["upper_wishbone"])
            pushrod = TwoPointLink(*joints["pushrod"])
            tierod = TwoPointLink(*joints["tierod"])

            return upper_wishbone, lower_wishbone, pushrod, tierod
        # Build front suspension
        geometry = load_geometry(frontend_start_line)
        self.front = Front(*geometry, "Front")


        # Build front suspension
        geometry = load_geometry(rearend_start_line)
        self.rear = Front(*geometry, "Rear")

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
        """Print complete suspension information."""
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

    def __init__(self):
        self.calculate_front_suspension_forces()

    def calculate_front_suspension_forces(self):
        # Front
        contact_patch = np.array([0, 600, -200])

        # +x: points to the rear of vehicle | +y: points to the right of vehicle | +z: points upwards
        contact_patch_force = np.array(
            [[0, -3421.4, 4109.7], [2504.5, 0, 3710.3], [-1534.6, 0, 2273.4], [0, 1000, 0]])  # [0,-1560,1874]

        suspension.front.upper_wishbone.build_unit_moment_vector()
        suspension.front.lower_wishbone.build_unit_moment_vector()
        suspension.front.pushrod.build_unit_moment_vector()
        suspension.front.tierod.build_unit_moment_vector()

        # Build base coefficient matrix (6x6)
        A_base = np.stack((suspension.front.lower_wishbone.front_unit_moment_vector,
                           suspension.front.lower_wishbone.rear_unit_moment_vector,
                           suspension.front.upper_wishbone.front_unit_moment_vector,
                           suspension.front.upper_wishbone.rear_unit_moment_vector,
                           suspension.front.pushrod.unit_moment_vector,
                           suspension.front.tierod.unit_moment_vector), axis=1)

        # Determine number of force cases from input
        n_cases = contact_patch_force.shape[0]

        # Dynamically replicate A to match number of force cases
        A = np.tile(A_base[np.newaxis, :, :], (n_cases, 1, 1))

        def calculate_input_moment():
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
        self.FIn = FIn[..., np.newaxis]

        FOut = np.linalg.solve(A, self.FIn)
        self.FOut = FOut.squeeze()

        def save_forces_to_members():
            suspension.front.lower_wishbone.force(FOut[:, 0], FOut[:, 1])
            suspension.front.upper_wishbone.force(FOut[:, 2], FOut[:, 3])
            suspension.front.pushrod.force(FOut[:, 4])
            suspension.front.tierod.force(FOut[:, 5])

        save_forces_to_members()

file_path = "C:/Users/pc/Downloads/HAFO24_v19_DECOUPLE_acc_steering (1).shk"
suspension = SuspensionGeometry(file_path)
suspension.print_all_geometry()