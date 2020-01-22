# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlpiot/proto/scene_description.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlpiot.proto import google_color_pb2 as mlpiot_dot_proto_dot_google__color__pb2
from mlpiot.proto import google_geometry_pb2 as mlpiot_dot_proto_dot_google__geometry__pb2
from mlpiot.proto import google_timestamp_pb2 as mlpiot_dot_proto_dot_google__timestamp__pb2
from mlpiot.proto import image_pb2 as mlpiot_dot_proto_dot_image__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlpiot/proto/scene_description.proto',
  package='mlpiot.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n$mlpiot/proto/scene_description.proto\x12\x0cmlpiot.proto\x1a\x1fmlpiot/proto/google_color.proto\x1a\"mlpiot/proto/google_geometry.proto\x1a#mlpiot/proto/google_timestamp.proto\x1a\x18mlpiot/proto/image.proto\"I\n\x17SceneDescriptorMetadata\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\x05\x12\x0f\n\x07payload\x18\x03 \x01(\t\"\xc5\x01\n\rObjectInScene\x12\x12\n\nclass_name\x18\x01 \x01(\t\x12\x16\n\x0e\x63lass_icon_url\x18\x02 \x01(\t\x12\x30\n\x0c\x62ounding_box\x18\x03 \x01(\x0b\x32\x1a.mlpiot.proto.BoundingPoly\x12\x12\n\nconfidence\x18\x04 \x01(\x02\x12\r\n\x05score\x18\x05 \x01(\x02\x12\"\n\x05\x63olor\x18\x06 \x01(\x0b\x32\x13.mlpiot.proto.Color\x12\x0f\n\x07payload\x18\x07 \x01(\t\"\xeb\x01\n\x10SceneDescription\x12*\n\ttimestamp\x18\x01 \x01(\x0b\x32\x17.mlpiot.proto.Timestamp\x12\x37\n\x08metadata\x18\x02 \x01(\x0b\x32%.mlpiot.proto.SceneDescriptorMetadata\x12,\n\x07objects\x18\x03 \x03(\x0b\x32\x1b.mlpiot.proto.ObjectInScene\x12,\n\x0f\x61nnotated_image\x18\x04 \x01(\x0b\x32\x13.mlpiot.proto.Image\x12\x16\n\x0einput_image_id\x18\x05 \x01(\x03\x62\x06proto3'
  ,
  dependencies=[mlpiot_dot_proto_dot_google__color__pb2.DESCRIPTOR,mlpiot_dot_proto_dot_google__geometry__pb2.DESCRIPTOR,mlpiot_dot_proto_dot_google__timestamp__pb2.DESCRIPTOR,mlpiot_dot_proto_dot_image__pb2.DESCRIPTOR,])




_SCENEDESCRIPTORMETADATA = _descriptor.Descriptor(
  name='SceneDescriptorMetadata',
  full_name='mlpiot.proto.SceneDescriptorMetadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlpiot.proto.SceneDescriptorMetadata.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='mlpiot.proto.SceneDescriptorMetadata.version', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='mlpiot.proto.SceneDescriptorMetadata.payload', index=2,
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
  serialized_start=186,
  serialized_end=259,
)


_OBJECTINSCENE = _descriptor.Descriptor(
  name='ObjectInScene',
  full_name='mlpiot.proto.ObjectInScene',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='class_name', full_name='mlpiot.proto.ObjectInScene.class_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class_icon_url', full_name='mlpiot.proto.ObjectInScene.class_icon_url', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bounding_box', full_name='mlpiot.proto.ObjectInScene.bounding_box', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='mlpiot.proto.ObjectInScene.confidence', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score', full_name='mlpiot.proto.ObjectInScene.score', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='color', full_name='mlpiot.proto.ObjectInScene.color', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='mlpiot.proto.ObjectInScene.payload', index=6,
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
  ],
  serialized_start=262,
  serialized_end=459,
)


_SCENEDESCRIPTION = _descriptor.Descriptor(
  name='SceneDescription',
  full_name='mlpiot.proto.SceneDescription',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='mlpiot.proto.SceneDescription.timestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='metadata', full_name='mlpiot.proto.SceneDescription.metadata', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='objects', full_name='mlpiot.proto.SceneDescription.objects', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='annotated_image', full_name='mlpiot.proto.SceneDescription.annotated_image', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_image_id', full_name='mlpiot.proto.SceneDescription.input_image_id', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=462,
  serialized_end=697,
)

_OBJECTINSCENE.fields_by_name['bounding_box'].message_type = mlpiot_dot_proto_dot_google__geometry__pb2._BOUNDINGPOLY
_OBJECTINSCENE.fields_by_name['color'].message_type = mlpiot_dot_proto_dot_google__color__pb2._COLOR
_SCENEDESCRIPTION.fields_by_name['timestamp'].message_type = mlpiot_dot_proto_dot_google__timestamp__pb2._TIMESTAMP
_SCENEDESCRIPTION.fields_by_name['metadata'].message_type = _SCENEDESCRIPTORMETADATA
_SCENEDESCRIPTION.fields_by_name['objects'].message_type = _OBJECTINSCENE
_SCENEDESCRIPTION.fields_by_name['annotated_image'].message_type = mlpiot_dot_proto_dot_image__pb2._IMAGE
DESCRIPTOR.message_types_by_name['SceneDescriptorMetadata'] = _SCENEDESCRIPTORMETADATA
DESCRIPTOR.message_types_by_name['ObjectInScene'] = _OBJECTINSCENE
DESCRIPTOR.message_types_by_name['SceneDescription'] = _SCENEDESCRIPTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SceneDescriptorMetadata = _reflection.GeneratedProtocolMessageType('SceneDescriptorMetadata', (_message.Message,), {
  'DESCRIPTOR' : _SCENEDESCRIPTORMETADATA,
  '__module__' : 'mlpiot.proto.scene_description_pb2'
  # @@protoc_insertion_point(class_scope:mlpiot.proto.SceneDescriptorMetadata)
  })
_sym_db.RegisterMessage(SceneDescriptorMetadata)

ObjectInScene = _reflection.GeneratedProtocolMessageType('ObjectInScene', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTINSCENE,
  '__module__' : 'mlpiot.proto.scene_description_pb2'
  # @@protoc_insertion_point(class_scope:mlpiot.proto.ObjectInScene)
  })
_sym_db.RegisterMessage(ObjectInScene)

SceneDescription = _reflection.GeneratedProtocolMessageType('SceneDescription', (_message.Message,), {
  'DESCRIPTOR' : _SCENEDESCRIPTION,
  '__module__' : 'mlpiot.proto.scene_description_pb2'
  # @@protoc_insertion_point(class_scope:mlpiot.proto.SceneDescription)
  })
_sym_db.RegisterMessage(SceneDescription)


# @@protoc_insertion_point(module_scope)
