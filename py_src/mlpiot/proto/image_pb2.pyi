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

from mlpiot.proto.google_timestamp_pb2 import (
    Timestamp as mlpiot___proto___google_timestamp_pb2___Timestamp,
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


class Image(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    height = ... # type: builtin___int
    width = ... # type: builtin___int
    channels = ... # type: builtin___int
    format = ... # type: typing___Text
    data = ... # type: builtin___bytes
    url = ... # type: typing___Text

    @property
    def timestamp(self) -> mlpiot___proto___google_timestamp_pb2___Timestamp: ...

    def __init__(self,
        *,
        timestamp : typing___Optional[mlpiot___proto___google_timestamp_pb2___Timestamp] = None,
        height : typing___Optional[builtin___int] = None,
        width : typing___Optional[builtin___int] = None,
        channels : typing___Optional[builtin___int] = None,
        format : typing___Optional[typing___Text] = None,
        data : typing___Optional[builtin___bytes] = None,
        url : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> Image: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"content_oneof",u"data",u"timestamp",u"url"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"channels",u"content_oneof",u"data",u"format",u"height",u"timestamp",u"url",u"width"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"content_oneof",b"content_oneof",u"data",b"data",u"timestamp",b"timestamp",u"url",b"url"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"channels",b"channels",u"content_oneof",b"content_oneof",u"data",b"data",u"format",b"format",u"height",b"height",u"timestamp",b"timestamp",u"url",b"url",u"width",b"width"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions___Literal[u"content_oneof",b"content_oneof"]) -> typing_extensions___Literal["data","url"]: ...

class ImageArray(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def timestamp(self) -> mlpiot___proto___google_timestamp_pb2___Timestamp: ...

    @property
    def objects(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[Image]: ...

    def __init__(self,
        *,
        timestamp : typing___Optional[mlpiot___proto___google_timestamp_pb2___Timestamp] = None,
        objects : typing___Optional[typing___Iterable[Image]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> ImageArray: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"timestamp"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"objects",u"timestamp"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"timestamp",b"timestamp"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"objects",b"objects",u"timestamp",b"timestamp"]) -> None: ...
