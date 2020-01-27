from typing import Optional

from .image_pb2 import Image


import numpy as np


class ImageWithHelpers(object):

    def __init__(self,
                 proto_image: Image,
                 numpy_image: np.ndarray = None):
        self.proto_image = proto_image
        self._numpy_image = numpy_image  # type: Optional[np.ndarray]

    def image_as_numpy_array(self) -> 'np.ndarray':
        if self._numpy_image is None:
            self._numpy_image = np.frombuffer(
                self.proto_image.data, dtype=np.uint8)
            self._numpy_image = self._numpy_image.reshape((
                self.proto_image.height,
                self.proto_image.width,
                self.proto_image.channels))
        return self._numpy_image
