from ts.torch_handler.vision_handler import VisionHandler
import torchvision.transforms as standard_transforms
from ts.utils.util import list_classes_from_module

import io
import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from timm.models import load_checkpoint
from bench import DetBenchPredict
from model import EfficientDet
from model_config import get_efficientdet_config
from transform import ResizePad

import logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RoadDamage(VisionHandler):
	"""
	A custom model handler implementation.
	"""
	
	def __init__(self,model_backbone="tf_efficientdet_d0"):
		self.explain = False
		self.target = 0
		self.resizer = None
		self.tensorizer = T.ToTensor()
		self.normalizer = T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
		self.inputlayersize = None
		self.model_backbone = model_backbone

	def calculate_scale(self,width,height):
		img_scale_y = self.inputlayersize / height
		img_scale_x = self.inputlayersize / width
		img_scale = min(img_scale_y, img_scale_x)
		return img_scale

	def initialize(self,context):
		"""
		Initialize model. This will be called during model loading time
		:return:
		"""
		self.map_location = "cpu"
		self.device = torch.device(
			self.map_location
		)
		
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
			logging.info(f"serializedFile model path: {model_pt_path}")
		logging.info(f"Loading pickle file with {model_dir} {model_file} {model_pt_path}")
		
		self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
		self.model.to(self.device)
		self.model.eval()
		self.initialized = True
		
	def _load_pickled_model(self,model_dir, model_file, model_pt_path):
		config = get_efficientdet_config(self.model_backbone)
		
		model = EfficientDet(config)
		load_checkpoint(model,model_pt_path)
		self.inputlayersize = config["image_size"]
		self.resizer = ResizePad(self.inputlayersize)
		model = DetBenchPredict(model,config)
		return model
	
	def preprocess(self, data):
		"""The preprocess function of MNIST program converts the input data to a float tensor

		Args:
			data (List): Input data from the request is in the form of a Tensor

		Returns:
			list : The preprocess function returns the input image as a list of float tensors.
		"""
		images = []
		scales = []
		sizes = []
		for row in data:
			# Compat layer: normally the envelope should just return the data
			# directly, but older versions of Torchserve didn't have envelope.
			img_raw = row.get("data") or row.get("body")
			logging.info(f"Received image of type {type(img_raw)}")
			image = Image.open(io.BytesIO(img_raw)).convert('RGB')

			width, height = image.size

			scale = self.calculate_scale(width, height)
			img_tensor = self.normalizer(self.tensorizer(self.resizer(image,scale)))
			
			images.append(img_tensor)
			scales.append(1/scale) # to recover
			sizes.append([width,height])

		images_tensor = torch.stack(images).to(self.device)
		scales_tensor = torch.Tensor(scales).to(self.device)
		sizes_tensor = torch.Tensor(sizes).to(self.device)
			
		return images_tensor,scales_tensor,sizes_tensor
	
	def inference(self,data, *args, **kwargs):
		images_tensor,scales_tensor,sizes_tensor = data
		return self.model(images_tensor,scales_tensor,sizes_tensor)
		
	def postprocess(self, outputs):
		"""
		Return inference result.
		:param inference_output: list of inference output
		:return: list of predict results
		"""
		# Take output from network and post-process to desired format
		logging.info(f"Postprocess: calculating response")
		response = outputs.detach().cpu().numpy().astype(np.float32)
		return [response.tobytes()]
