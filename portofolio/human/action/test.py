import sys
import json
import requests
import logging
import numpy as np
import grpc
import io
from grpc import aio 
import asyncio
import aiohttp 
import time
from PIL import Image
import ast

"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\x12 org.pytorch.serve.grpc.inference\x1a\x1bgoogle/protobuf/empty.proto\"\xbd\x01\n\x12PredictionsRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x15\n\rmodel_version\x18\x02 \x01(\t\x12N\n\x05input\x18\x03 \x03(\x0b\x32?.org.pytorch.serve.grpc.inference.PredictionsRequest.InputEntry\x1a,\n\nInputEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\"(\n\x12PredictionResponse\x12\x12\n\nprediction\x18\x01 \x01(\x0c\"*\n\x18TorchServeHealthResponse\x12\x0e\n\x06health\x18\x01 \x01(\t2\xf1\x01\n\x14InferenceAPIsService\x12\\\n\x04Ping\x12\x16.google.protobuf.Empty\x1a:.org.pytorch.serve.grpc.inference.TorchServeHealthResponse\"\x00\x12{\n\x0bPredictions\x12\x34.org.pytorch.serve.grpc.inference.PredictionsRequest\x1a\x34.org.pytorch.serve.grpc.inference.PredictionResponse\"\x00\x42\x02P\x01\x62\x06proto3')



_PREDICTIONSREQUEST = DESCRIPTOR.message_types_by_name['PredictionsRequest']
_PREDICTIONSREQUEST_INPUTENTRY = _PREDICTIONSREQUEST.nested_types_by_name['InputEntry']
_PREDICTIONRESPONSE = DESCRIPTOR.message_types_by_name['PredictionResponse']
_TORCHSERVEHEALTHRESPONSE = DESCRIPTOR.message_types_by_name['TorchServeHealthResponse']
PredictionsRequest = _reflection.GeneratedProtocolMessageType('PredictionsRequest', (_message.Message,), {

  'InputEntry' : _reflection.GeneratedProtocolMessageType('InputEntry', (_message.Message,), {
    'DESCRIPTOR' : _PREDICTIONSREQUEST_INPUTENTRY,
    '__module__' : 'inference_pb2'
    # @@protoc_insertion_point(class_scope:org.pytorch.serve.grpc.inference.PredictionsRequest.InputEntry)
    })
  ,
  'DESCRIPTOR' : _PREDICTIONSREQUEST,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:org.pytorch.serve.grpc.inference.PredictionsRequest)
  })
_sym_db.RegisterMessage(PredictionsRequest)
_sym_db.RegisterMessage(PredictionsRequest.InputEntry)

PredictionResponse = _reflection.GeneratedProtocolMessageType('PredictionResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONRESPONSE,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:org.pytorch.serve.grpc.inference.PredictionResponse)
  })
_sym_db.RegisterMessage(PredictionResponse)

TorchServeHealthResponse = _reflection.GeneratedProtocolMessageType('TorchServeHealthResponse', (_message.Message,), {
  'DESCRIPTOR' : _TORCHSERVEHEALTHRESPONSE,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:org.pytorch.serve.grpc.inference.TorchServeHealthResponse)
  })
_sym_db.RegisterMessage(TorchServeHealthResponse)

_INFERENCEAPISSERVICE = DESCRIPTOR.services_by_name['InferenceAPIsService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'P\001'
  _PREDICTIONSREQUEST_INPUTENTRY._options = None
  _PREDICTIONSREQUEST_INPUTENTRY._serialized_options = b'8\001'
  _PREDICTIONSREQUEST._serialized_start=83
  _PREDICTIONSREQUEST._serialized_end=272
  _PREDICTIONSREQUEST_INPUTENTRY._serialized_start=228
  _PREDICTIONSREQUEST_INPUTENTRY._serialized_end=272
  _PREDICTIONRESPONSE._serialized_start=274
  _PREDICTIONRESPONSE._serialized_end=314
  _TORCHSERVEHEALTHRESPONSE._serialized_start=316
  _TORCHSERVEHEALTHRESPONSE._serialized_end=358
  _INFERENCEAPISSERVICE._serialized_start=361
  _INFERENCEAPISSERVICE._serialized_end=602
# @@protoc_insertion_point(module_scope)

import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

class InferenceAPIsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Ping = channel.unary_unary(
                '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Ping',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=TorchServeHealthResponse.FromString,
                )
        self.Predictions = channel.unary_unary(
                '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Predictions',
                request_serializer=PredictionsRequest.SerializeToString,
                response_deserializer=PredictionResponse.FromString,
                )

class InferenceAPIsServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Ping(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predictions(self, request, context):
        """Predictions entry point to get inference using default model version.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_InferenceAPIsServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Ping': grpc.unary_unary_rpc_method_handler(
                    servicer.Ping,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=TorchServeHealthResponse.SerializeToString,
            ),
            'Predictions': grpc.unary_unary_rpc_method_handler(
                    servicer.Predictions,
                    request_deserializer=PredictionsRequest.FromString,
                    response_serializer=PredictionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'org.pytorch.serve.grpc.inference.InferenceAPIsService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.

class InferenceAPIsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Ping',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            TorchServeHealthResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Predictions(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Predictions',
            PredictionsRequest.SerializeToString,
            PredictionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


data = np.random.random((1,314)).astype(np.float32)


input_data = {'data': data.tobytes()}

host = "0.0.0.0:5800"
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096*4096*3
channel = grpc.insecure_channel(host,options=[
                    ('grpc.max_send_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
stub = InferenceAPIsServiceStub(channel)
response =  stub.Predictions(
    PredictionsRequest(model_name="har",
                                    input=input_data))
prediction = response.prediction
print("to literal eval")
detections = np.frombuffer(prediction,dtype=np.float32)
print(detections,detections.shape)