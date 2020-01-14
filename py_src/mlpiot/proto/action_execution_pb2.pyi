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

from mlpiot.proto.event_pb2 import (
    Event as mlpiot___proto___event_pb2___Event,
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


class ActionExecutorMetadata(google___protobuf___message___Message):
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
    def FromString(cls, s: builtin___bytes) -> ActionExecutorMetadata: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"name",u"payload",u"version"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"name",b"name",u"payload",b"payload",u"version",b"version"]) -> None: ...

class ActionExecution(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    cycle_id = ... # type: builtin___int

    @property
    def timestamp(self) -> mlpiot___proto___google_timestamp_pb2___Timestamp: ...

    @property
    def metadata(self) -> ActionExecutorMetadata: ...

    @property
    def action_execution_events(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[mlpiot___proto___event_pb2___Event]: ...

    def __init__(self,
        *,
        cycle_id : typing___Optional[builtin___int] = None,
        timestamp : typing___Optional[mlpiot___proto___google_timestamp_pb2___Timestamp] = None,
        metadata : typing___Optional[ActionExecutorMetadata] = None,
        action_execution_events : typing___Optional[typing___Iterable[mlpiot___proto___event_pb2___Event]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> ActionExecution: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"metadata",u"timestamp"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"action_execution_events",u"cycle_id",u"metadata",u"timestamp"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"metadata",b"metadata",u"timestamp",b"timestamp"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"action_execution_events",b"action_execution_events",u"cycle_id",b"cycle_id",u"metadata",b"metadata",u"timestamp",b"timestamp"]) -> None: ...
