# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Optional as typing___Optional,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


class Timestamp(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    seconds = ... # type: builtin___int
    nanos = ... # type: builtin___int

    def __init__(self,
        *,
        seconds : typing___Optional[builtin___int] = None,
        nanos : typing___Optional[builtin___int] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> Timestamp: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"nanos",u"seconds"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"nanos",b"nanos",u"seconds",b"seconds"]) -> None: ...
