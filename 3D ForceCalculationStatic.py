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

class TwoPointLink: # Pushrod, TieRod
    def __init__(self,inside_joint: Joint,outside_joint: Joint):
        self.inside_joint = inside_joint
        self.outside_joint = outside_joint

        self.unit_moment_vector = None

        self.link_force = None

        self.link_force_x = None
        self.link_force_y = None
        self.link_force_z = None


aaaaaaaaa=7