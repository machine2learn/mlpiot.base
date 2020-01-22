# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlpiot/proto/image.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlpiot.proto import google_timestamp_pb2 as mlpiot_dot_proto_dot_google__timestamp__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlpiot/proto/image.proto',
  package='mlpiot.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x18mlpiot/proto/image.proto\x12\x0cmlpiot.proto\x1a#mlpiot/proto/google_timestamp.proto\"\xa4\x01\n\x05Image\x12*\n\ttimestamp\x18\x01 \x01(\x0b\x32\x17.mlpiot.proto.Timestamp\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x10\n\x08\x63hannels\x18\x04 \x01(\x05\x12\x0e\n\x06\x66ormat\x18\x05 \x01(\t\x12\x0e\n\x04\x64\x61ta\x18\x06 \x01(\x0cH\x00\x12\r\n\x03url\x18\x07 \x01(\tH\x00\x42\x0f\n\rcontent_oneofb\x06proto3'
  ,
  dependencies=[mlpiot_dot_proto_dot_google__timestamp__pb2.DESCRIPTOR,])




_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='mlpiot.proto.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='mlpiot.proto.Image.timestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='mlpiot.proto.Image.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='mlpiot.proto.Image.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channels', full_name='mlpiot.proto.Image.channels', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='format', full_name='mlpiot.proto.Image.format', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='mlpiot.proto.Image.data', index=5,
      number=6, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='url', full_name='mlpiot.proto.Image.url', index=6,
      number=7, type=9, cpp_type=9, label=1,
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
    _descriptor.OneofDescriptor(
      name='content_oneof', full_name='mlpiot.proto.Image.content_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=80,
  serialized_end=244,
)

_IMAGE.fields_by_name['timestamp'].message_type = mlpiot_dot_proto_dot_google__timestamp__pb2._TIMESTAMP
_IMAGE.oneofs_by_name['content_oneof'].fields.append(
  _IMAGE.fields_by_name['data'])
_IMAGE.fields_by_name['data'].containing_oneof = _IMAGE.oneofs_by_name['content_oneof']
_IMAGE.oneofs_by_name['content_oneof'].fields.append(
  _IMAGE.fields_by_name['url'])
_IMAGE.fields_by_name['url'].containing_oneof = _IMAGE.oneofs_by_name['content_oneof']
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'mlpiot.proto.image_pb2'
  # @@protoc_insertion_point(class_scope:mlpiot.proto.Image)
  })
_sym_db.RegisterMessage(Image)


# @@protoc_insertion_point(module_scope)
