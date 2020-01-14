# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from mlpiot.proto.google_color_pb2 import (
    Color as mlpiot___proto___google_color_pb2___Color,
)

from mlpiot.proto.google_geometry_pb2 import (
    BoundingPoly as mlpiot___proto___google_geometry_pb2___BoundingPoly,
)

from mlpiot.proto.google_timestamp_pb2 import (
    Timestamp as mlpiot___proto___google_timestamp_pb2___Timestamp,
)

from mlpiot.proto.image_pb2 import (
    Image as mlpiot___proto___image_pb2___Image,
)

from typing import (
    Iterable as typing___Iterable,
    Optional as typing___Optional,
    Text as typing___Text,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


class SceneDescriptorMetadata(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    name = ... # type: typing___Text
    version = ... # type: builtin___int
    payload = ... # type: typing___Text

    def __init__(self,
        *,
        name : typing___Optional[typing___Text] = None,
        version : typing___Optional[builtin___int] = None,
        payload : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> SceneDescriptorMetadata: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"name",u"payload",u"version"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"name",b"name",u"payload",b"payload",u"version",b"version"]) -> None: ...

class ObjectInScene(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    id = ... # type: typing___Text
    class_name = ... # type: typing___Text
    class_icon_url = ... # type: typing___Text
    confidence = ... # type: builtin___float
    score = ... # type: builtin___float

    @property
    def bounding_box(self) -> mlpiot___proto___google_geometry_pb2___BoundingPoly: ...

    @property
    def color(self) -> mlpiot___proto___google_color_pb2___Color: ...

    def __init__(self,
        *,
        id : typing___Optional[typing___Text] = None,
        class_name : typing___Optional[typing___Text] = None,
        class_icon_url : typing___Optional[typing___Text] = None,
        bounding_box : typing___Optional[mlpiot___proto___google_geometry_pb2___BoundingPoly] = None,
        confidence : typing___Optional[builtin___float] = None,
        score : typing___Optional[builtin___float] = None,
        color : typing___Optional[mlpiot___proto___google_color_pb2___Color] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> ObjectInScene: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"bounding_box",u"color"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"bounding_box",u"class_icon_url",u"class_name",u"color",u"confidence",u"id",u"score"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"bounding_box",b"bounding_box",u"color",b"color"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"bounding_box",b"bounding_box",u"class_icon_url",b"class_icon_url",u"class_name",b"class_name",u"color",b"color",u"confidence",b"confidence",u"id",b"id",u"score",b"score"]) -> None: ...

class SceneDescription(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    cycle_id = ... # type: builtin___int
    input_image_id = ... # type: builtin___int

    @property
    def timestamp(self) -> mlpiot___proto___google_timestamp_pb2___Timestamp: ...

    @property
    def metadata(self) -> SceneDescriptorMetadata: ...

    @property
    def objects(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[ObjectInScene]: ...

    @property
    def annotated_image(self) -> mlpiot___proto___image_pb2___Image: ...

    def __init__(self,
        *,
        cycle_id : typing___Optional[builtin___int] = None,
        timestamp : typing___Optional[mlpiot___proto___google_timestamp_pb2___Timestamp] = None,
        metadata : typing___Optional[SceneDescriptorMetadata] = None,
        objects : typing___Optional[typing___Iterable[ObjectInScene]] = None,
        annotated_image : typing___Optional[mlpiot___proto___image_pb2___Image] = None,
        input_image_id : typing___Optional[builtin___int] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> SceneDescription: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"annotated_image",u"metadata",u"timestamp"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"annotated_image",u"cycle_id",u"input_image_id",u"metadata",u"objects",u"timestamp"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"annotated_image",b"annotated_image",u"metadata",b"metadata",u"timestamp",b"timestamp"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"annotated_image",b"annotated_image",u"cycle_id",b"cycle_id",u"input_image_id",b"input_image_id",u"metadata",b"metadata",u"objects",b"objects",u"timestamp",b"timestamp"]) -> None: ...
