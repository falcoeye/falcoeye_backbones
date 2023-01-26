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
import torchvision.transforms as T
from timm.models import load_checkpoint
from bench import DetBenchPredict
logger = logging.getLogger(__name__)


class DetectWaste(VisionHandler):
	"""
	A custom model handler implementation.
	"""
	image_processing=transform = standard_transforms.Compose([
		T.Resize((768, 768)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
		model_class = None
		for m in model_class_definitions:
			if m.__name__ == "EfficientDet":
				model_class = m
				break

		model = model_class()
		model.reset_head(num_classes=1)
		load_checkpoint(model,model_pt_path)
		model = DetBenchPredict(model)
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
		response = outputs.detach().cpu().numpy().astype(np.float32)
		return [response.tolist()]