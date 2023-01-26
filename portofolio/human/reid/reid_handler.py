from ts.torch_handler.vision_handler import VisionHandler
from ts.utils.util import list_classes_from_module
from torchvision import datasets, transforms
import logging
import torch
import os
import io
from PIL import Image
import importlib.util
import numpy as np
import base64
logger = logging.getLogger(__name__)

data_meta = {}
data_meta['num_classes'] = 625
data_meta['mean'] = [0.4144, 0.3932, 0.3824]
data_meta['std'] = [0.1929, 0.1880, 0.1849]
data_meta['size'] = 256, 128

class Reid(VisionHandler):
	"""
	A custom model handler implementation.
	"""
	image_processing=transforms.Compose([
			transforms.Resize(data_meta['size']),
			transforms.ToTensor(),
			transforms.Normalize(data_meta['mean'], data_meta['std'])
		])
	def __init__(self):
		self._context = None
		self.initialized = False
		self.explain = False
		self.target = 0

	def initialize(self, context):
		"""
		Initialize model. This will be called during model loading time
		:param context: Initial context contains model server system properties.
		:return:
		"""
		properties = context.system_properties
		self.map_location = "cpu"
		self.device = torch.device(
			self.map_location
		)
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
		self.model.to(self.device)
		self.model.eval()
		self.initialized = True
	
	def _load_pickled_model(self, model_dir, model_file, model_pt_path):
		model_def_path = os.path.join(model_dir, model_file)
		if not os.path.isfile(model_def_path):
			raise RuntimeError("Missing the model.py file")

		module = importlib.import_module(model_file.split(".")[0])
		model_class_definitions = list_classes_from_module(module)
		for m in model_class_definitions:
			if m.__name__ == "WideResnet":
				model_class = m
				break
		logging.info(f"loading model class {model_class}")
		model = model_class()
		if model_pt_path:
			logger.info(f"Loading {model_pt_path} ")
			state_dict = torch.load(model_pt_path, map_location=self.device)['net_dict']
			model.load_state_dict(state_dict)
		return model
	
	def preprocess(self, data):
		"""The preprocess function of MNIST program converts the input data to a float tensor

		Args:
			data (List): Input data from the request is in the form of a Tensor

		Returns:
			list : The preprocess function returns the input image as a list of float tensors.
		"""
		images = []

		for row in data:
			# Compat layer: normally the envelope should just return the data
			# directly, but older versions of Torchserve didn't have envelope.
			image = row.get("data") or row.get("body")
			logger.info(f"Received image of type {type(image)}")
			if isinstance(image, str):
				# if the image is a string of bytesarray.
				image = base64.b64decode(image)

			# If the image is sent as bytesarray
			if isinstance(image, (bytearray, bytes)):
				image = Image.open(io.BytesIO(image))
				image = self.image_processing(image)
			else:
				# if the image is a list
				image = torch.FloatTensor(image)

			images.append(image)

		return torch.stack(images).float().to(self.device)
	
	def postprocess(self, inference_output):
		"""
		Return inference result.
		:param inference_output: list of inference output
		:return: list of predict results
		"""
		# Take output from network and post-process to desired format
		inf = inference_output.detach().cpu().numpy().astype(np.float32)
		
		return [inf.tobytes()]