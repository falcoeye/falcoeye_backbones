import logging
import os
import io
from PIL import Image
import importlib.util
import numpy as np
import base64
import pickle
logger = logging.getLogger(__name__)
from ts.torch_handler.base_handler import BaseHandler

class HAR(BaseHandler):
	"""
	A custom model handler implementation.
	"""
	def __init__(self):
		self.pca = None

	def initialize(self, context):
		"""
		Initialize model. This will be called during model loading time
		:param context: Initial context contains model server system properties.
		:return:
		"""
		properties = context.system_properties
		
		self.manifest = context.manifest
		model_dir = properties.get("model_dir")
		model_file = self.manifest["model"].get("modelFile", "")
		model_pt_path = None
		

		if "serializedFile" in self.manifest["model"]:
			serialized_file = self.manifest["model"]["serializedFile"]
			model_pt_path = os.path.join(model_dir, serialized_file)
			logger.info(f"serializedFile model path: {model_pt_path}")
		logger.info(f"Loading pickle file with {model_dir} {model_file} {model_pt_path}")
		self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
		
		self.initialized = True
	
	def _load_pickled_model(self, model_dir, model_file, model_pt_path):
		model_def_path = os.path.join(model_dir, model_file)
		if not os.path.isfile(model_def_path):
			raise RuntimeError("Missing the model.py file")

		logger.info(f"Loading {model_pt_path} ")
		with open(model_pt_path,"rb") as f:
			model = pickle.load(f)
		
		with open("./pca.pkl","rb") as f:
			self.pca = pickle.load(f)

		return model
	
		
	def preprocess(self, data):
		"""The preprocess function of MNIST program converts the input data to a float tensor

		Args:
			data (List): Input data from the request is in the form of a Tensor

		Returns:
			list : The preprocess function returns the input image as a list of float tensors.
		"""
		image = data[0].get("data") or data[0].get("body")
		image = np.frombuffer(image, dtype=np.float32) 
		image = image.reshape((-1,314))
		logger.info(f"feature shape {image.shape}")
		return image
	

	def inference(self, data, *args, **kwargs):
		result = self.model.predict_proba(self.pca.transform(data))
		return result
	
	def postprocess(self, inference_output):
		"""
		Return inference result.
		:param inference_output: list of inference output
		:return: list of predict results
		"""
		# Take output from network and post-process to desired format
		logger.info(f"result shape {inference_output.shape}")
		return [inference_output.astype(np.float32).tobytes()]