import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import asyncio
import aiohttp 
import grpc
import inference_pb2
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc
from grpc import aio 
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from config import get_default_configuration
from coordinates import get_coordinates
from connections import get_connections
from estimators import estimate
from renderers import draw

BODY_CONFIG = get_default_configuration()

GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3  # Max LENGTH the GRPC should handle

#async def post_grpc(pid,stub,frame):
def post_grpc(pid,stub,frame):
	try:
		print(frame.shape)
		start_time = time.time()
		request = predict_pb2.PredictRequest()
		request.model_spec.name = "tfpose"
		request.model_spec.signature_name = 'serving_default'
		request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(frame))
		result = stub.Predict(request)
		paf_shape = tuple([i.size for i in list(result.outputs['output_11'].tensor_shape.dim)][1:])
		htm_shape = tuple([i.size for i in list(result.outputs['output_12'].tensor_shape.dim)][1:])
		pafs = np.array(result.outputs['output_11'].float_val,
			dtype=np.float32).reshape(paf_shape)
		heatmap = np.array(result.outputs['output_12'].float_val,
			dtype=np.float32).reshape(htm_shape)
		
		coordinates = get_coordinates(BODY_CONFIG, heatmap)
		connections = get_connections(BODY_CONFIG, coordinates, pafs)
		skeletons = estimate(BODY_CONFIG, connections)
		
		output = draw(BODY_CONFIG, frame[0], coordinates, skeletons, resize_fac=8)
		Image.fromarray(output.astype(np.uint8)).save("./test_pred.jpg")
		
		print(f"---{pid} done with time {time.time() - start_time}")

	except Exception as e:
		raise
		print(f'{pid} failed with {e}')


#async def remote_predict(fileName,port):
def remote_predict(fileName,port):
	host = f"localhost:{port}"
	if fileName.endswith(".jpg") or fileName.endswith(".jpeg") or fileName.endswith(".png"):
		
		img = cv2.imread(fileName)  # B,G,R order
		img = cv2.resize(img, (640,360))
		input_img = img[np.newaxis, :, :, [2, 1, 0]].astype(np.uint8)
		#input_img = np.load("/Users/jalalirs/Documents/code/falcoeye/falcoeye_analysis/data_0.npy")
		#input_img = input_img[np.newaxis, :, :, :].astype(np.uint8)


		#async with aio.insecure_channel(host) as channel:
		with grpc.insecure_channel(host,options=[('grpc.max_receive_message_length', 
			GRPC_MAX_RECEIVE_MESSAGE_LENGTH)]) as channel:
		#async with aio.secure_channel(host, grpc.ssl_channel_credentials(), options=options) as channel:
			stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
			#await post_grpc(pid=1,stub=stub,frame=data)
			post_grpc(pid=1,stub=stub,frame=input_img)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Predict single file with a tensorflow model')
	parser.add_argument('--model', type=str,default=None, help='Model name')
	parser.add_argument('--file', type=str,default=None, help='File name')
	parser.add_argument("--grpc", action="store_true")
	parser.add_argument('--port', type=int,default=5802, help='File name')
	args = parser.parse_args()
	if args.grpc:
		#asyncio.run(remote_predict(args.file,args.port))
		remote_predict(args.file,args.port)
	
	

	