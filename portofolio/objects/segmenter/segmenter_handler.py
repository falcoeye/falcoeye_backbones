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
import torchvision.transforms.functional as F
from utils import resize,sliding_window,merge_windows
import numpy as np


logger = logging.getLogger(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

STATS = {
	"vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}

class Segmenter(VisionHandler):
	"""
	A custom model handler implementation.
	"""

	def __init__(self):
		self._context = None
		self.initialized = False
		self.explain = False
		self.target = 0
		self._variant = None
		self._normalization = STATS["vit"]
		self._n_cls = 0

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
			if m.__name__ == "SegmenterLoader":
				model_class = m
				break

		modelLoader = model_class(model_pt_path)
		model,self._variant = modelLoader.load()
		self._window_size = self._variant["inference_kwargs"]["window_size"]
		self._window_stride = self._variant["inference_kwargs"]["window_stride"]
		self._n_cls = model.n_cls
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
		
			pil_im = Image.open(io.BytesIO(img_raw)).convert('RGB')
			im = F.pil_to_tensor(pil_im).float() / 255
			im = F.normalize(im, self._normalization["mean"], self._normalization["std"])
			im = im.to("cpu").unsqueeze(0)
			ori_shape = im.shape[2:4]
			im = resize(im, self._window_size)
			windows = sliding_window(im, False, self._window_size, self._window_stride)
			crops = torch.stack(windows.pop("crop"))[:, 0]
			seg_map = torch.zeros((self._n_cls, ori_shape[0], ori_shape[1]), device=self.map_location)
			images.append([windows,crops,seg_map,ori_shape])
		
		return images
	
	def inference(self, data, *args, **kwargs):
		logits = []
		for windows,crops,seg_map,ori_shape in data:
			B = len(crops)
			WB = 2 # bach size
			seg_maps = torch.zeros((B, self._n_cls, self._window_size, 
					self._window_size), device=self.map_location)
			with torch.no_grad():
				for i in range(0, B, WB):
					seg_maps[i : i + WB] = self.model.forward(crops[i : i + WB])
			windows["seg_maps"] = seg_maps
			im_seg_map = merge_windows(windows, self._window_size, ori_shape)
			seg_map = im_seg_map.argmax(0, keepdim=True)
			
			logits.append(seg_map.detach().cpu().numpy().astype(np.float32))
		
		return np.array(logits)
		
	def postprocess(self, outputs):
		"""
		Return inference result.
		:param inference_output: list of inference output
		:return: list of predict results
		"""
		# Take output from network and post-process to desired format
		logger.info(f"Postprocess: calculating response")
		return [outputs.tobytes()]