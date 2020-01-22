# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlpiot/proto/action_execution.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlpiot.proto import google_timestamp_pb2 as mlpiot_dot_proto_dot_google__timestamp__pb2
from mlpiot.proto import event_pb2 as mlpiot_dot_proto_dot_event__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlpiot/proto/action_execution.proto',
  package='mlpiot.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n#mlpiot/proto/action_execution.proto\x12\x0cmlpiot.proto\x1a#mlpiot/proto/google_timestamp.proto\x1a\x18mlpiot/proto/event.proto\"H\n\x16\x41\x63tionExecutorMetadata\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\x05\x12\x0f\n\x07payload\x18\x03 \x01(\t\"\xab\x01\n\x0f\x41\x63tionExecution\x12*\n\ttimestamp\x18\x01 \x01(\x0b\x32\x17.mlpiot.proto.Timestamp\x12\x36\n\x08metadata\x18\x02 \x01(\x0b\x32$.mlpiot.proto.ActionExecutorMetadata\x12\x34\n\x17\x61\x63tion_execution_events\x18\x03 \x03(\x0b\x32\x13.mlpiot.proto.Eventb\x06proto3'
  ,
  dependencies=[mlpiot_dot_proto_dot_google__timestamp__pb2.DESCRIPTOR,mlpiot_dot_proto_dot_event__pb2.DESCRIPTOR,])




_ACTIONEXECUTORMETADATA = _descriptor.Descriptor(
  name='ActionExecutorMetadata',
  full_name='mlpiot.proto.ActionExecutorMetadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlpiot.proto.ActionExecutorMetadata.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='mlpiot.proto.ActionExecutorMetadata.version', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='mlpiot.proto.ActionExecutorMetadata.payload', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=116,
  serialized_end=188,
)


_ACTIONEXECUTION = _descriptor.Descriptor(
  name='ActionExecution',
  full_name='mlpiot.proto.ActionExecution',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='mlpiot.proto.ActionExecution.timestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='metadata', full_name='mlpiot.proto.ActionExecution.metadata', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action_execution_events', full_name='mlpiot.proto.ActionExecution.action_execution_events', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=191,
  serialized_end=362,
)

_ACTIONEXECUTION.fields_by_name['timestamp'].message_type = mlpiot_dot_proto_dot_google__timestamp__pb2._TIMESTAMP
_ACTIONEXECUTION.fields_by_name['metadata'].message_type = _ACTIONEXECUTORMETADATA
_ACTIONEXECUTION.fields_by_name['action_execution_events'].message_type = mlpiot_dot_proto_dot_event__pb2._EVENT
DESCRIPTOR.message_types_by_name['ActionExecutorMetadata'] = _ACTIONEXECUTORMETADATA
DESCRIPTOR.message_types_by_name['ActionExecution'] = _ACTIONEXECUTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ActionExecutorMetadata = _reflection.GeneratedProtocolMessageType('ActionExecutorMetadata', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONEXECUTORMETADATA,
  '__module__' : 'mlpiot.proto.action_execution_pb2'
  # @@protoc_insertion_point(class_scope:mlpiot.proto.ActionExecutorMetadata)
  })
_sym_db.RegisterMessage(ActionExecutorMetadata)

ActionExecution = _reflection.GeneratedProtocolMessageType('ActionExecution', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONEXECUTION,
  '__module__' : 'mlpiot.proto.action_execution_pb2'
  # @@protoc_insertion_point(class_scope:mlpiot.proto.ActionExecution)
  })
_sym_db.RegisterMessage(ActionExecution)


# @@protoc_insertion_point(module_scope)
