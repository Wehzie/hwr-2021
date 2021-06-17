from numpy import ndarray


class BoundingBox:
    """Simple representation of a rectangular form."""

    def __init__(self, x: int, y: int, w: int, h: int):
        """Initialize the bounding box."""
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h

    def image_slice(self, img: ndarray) -> ndarray:
        """Get the slice of an image enclosed by this bounding box."""
        return img[self.y : self.y + self.h, self.x : self.x + self.w]

    def __str__(self) -> str:
        """Get the string representation of the bounding box."""
        return f"x:{self.x}, y:{self.y}, w:{self.w}, h:{self.h}"
