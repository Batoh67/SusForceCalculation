# ==========Geometry===========
class Joint:
    """Represents a 3D joint/point in suspension geometry."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z