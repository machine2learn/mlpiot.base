from typing import Optional, TYPE_CHECKING

from .image_pb2 import Image


if TYPE_CHECKING:
    import numpy


class ImageWithHelpers(object):

    def __init__(self, proto_image: Image):
        self.proto_image = proto_image
        self._numpy_image = None  # type: Optional[numpy.ndarray]

    def set_numpy_image(self, numpy_image: 'numpy.ndarray'):
        self._numpy_image = numpy_image

    def image_as_numpy_array(self) -> 'numpy.ndarray':
        # TODO: load from proto_image if needed
        return self._numpy_image
