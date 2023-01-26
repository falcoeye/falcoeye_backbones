from ts.torch_handler.vision_handler import VisionHandler
import torchvision.transforms as standard_transforms
from ts.utils.util import list_classes_from_module
import logging
import torch
import os
import io
import importlib.util
from PIL import Image
import numpy as np
import base64
logger = logging.getLogger(__name__)
from dataset_params import dataset_params

class BirdYolo(VisionHandler):
	"""
	A custom model handler implementation.
	"""
	image_processing=transform = standard_transforms.Compose([
		standard_transforms.ToTensor()
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
		self.dsp = dataset_params["bird_dataset"]
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
		model_class = None
		for m in model_class_definitions:
			if m.__name__ == "YoloNet":
				model_class = m
				break

		model = model_class(self.dsp)
		if model_pt_path:
			logger.info(f"Loading {model_pt_path} ")
			checkpoint = torch.load(model_pt_path, map_location=self.device)
			model.load_state_dict(checkpoint['state_dict'])
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
			img_raw = row.get("data") or row.get("body")
			logger.info(f"Received image of type {type(img_raw)}")
		
			image = Image.open(io.BytesIO(img_raw)).convert('RGB')
			width, height = image.size
			new_width = self.dsp["img_w"]
			new_height =  self.dsp["img_h"]
			image = image.resize((new_width, new_height), Image.ANTIALIAS)
			image = self.image_processing(image)
			images.append(image)
		images_tensor = torch.stack(images).to(self.device)
		logger.info(images_tensor.shape)
		return images_tensor
	
	def postprocess(self, outputs):
		"""
		Return inference result.
		:param inference_output: list of inference output
		:return: list of predict results
		"""
		# Take output from network and post-process to desired format
		logger.info(f"Postprocess: calculating response")
		response_tensor = outputs.detach().cpu().numpy()
		response = np.zeros((
			response_tensor.shape[0],
			response_tensor.shape[1],6))
		response[:,:,:4] = response_tensor[:,:,:4] 
		response[:,:,4] = response_tensor[:,:,4] 
		response[:,:,5] = np.argmax(response_tensor[:,:,5:],axis=2)
		response = response.astype(np.float32)
		return [response.tobytes()]