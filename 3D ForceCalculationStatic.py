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