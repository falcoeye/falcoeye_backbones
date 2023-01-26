
import torchvision.transforms as standard_transforms
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
import argparse
from model import EfficientDet
import vis_utils 
logger = logging.getLogger(__name__)


class DetectWaste:
	"""
	A custom model handler implementation.
	"""
	image_processing=transform = standard_transforms.Compose([
		T.Resize((768, 768)),
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	def __init__(self):
		self.explain = False
		self.target = 0

	def initialize(self):
		"""
		Initialize model. This will be called during model loading time
		:return:
		"""
		self.map_location = "cpu"
		self.device = torch.device(
			self.map_location
		)
		
		self.model = self._load_pickled_model()
		self.model.to(self.device)
		self.model.eval()
		
		
	def _load_pickled_model(self):
		

		model = EfficientDet()
		model.reset_head(num_classes=1)
		load_checkpoint(model,args.checkpoint)
		model = DetBenchPredict(model)
		return model
	
	def predict(self,img):
		return self.model(img)
	
	
	def preprocess(self, data):
		"""The preprocess function of MNIST program converts the input data to a float tensor

		Args:
			data (List): Input data from the request is in the form of a Tensor

		Returns:
			list : The preprocess function returns the input image as a list of float tensors.
		"""
		images = [self.image_processing(Image.fromarray(data))]
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
		return response.reshape(-1,6)

class FileSource:
	def __init__(self, filename):
		self._filename = filename
		self._reader = None
		self.width = -1
		self.height = -1
		self.frames_per_second = -1
		self.num_frames = -1

	def open(self):
		self._reader = cv2.VideoCapture(self._filename)
		self.width = int(self._reader.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self._reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.frames_per_second = self._reader.get(cv2.CAP_PROP_FPS)
		self.num_frames = int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT))

	def read(self):
		return self._reader.read()

	def close(self):
		self._reader.release()
		self.width = -1
		self.height = -1
		self.frames_per_second = -1
		self.num_frames = -1

class FileSink:
	def __init__(self, filename):
		self._filename = filename
		self._writer = None

	def open(self, frames_per_second, width, height):
		self._writer = cv2.VideoWriter(
			self._filename,
			fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
			fps=30,
			frameSize=(width, height),
			isColor=True,
		)

	def sink(self, frame):

		self._writer.write(frame)

	def close(self):
		self._writer.release()


def get_args_parser():
	parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
	
	parser.add_argument('--file', default='',
						help='file to predict')
	parser.add_argument('--checkpoint', default='',
						help='path where the trained weights saved')
	
	return parser

def main_video(args, debug=False):
	model = DetectWaste()
	source = FileSource(args.file)
	sink = FileSink(args.file.replace(".mp4","_pred.mp4").replace(".mov","_pred.mov"))

	frames = []
	source.open()
	sink.open(source.frames_per_second,source.width,source.height)
	model.initialize()
	count = 0
	read, frame = source.read()
	while read:
		im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		predicted = mode.predict(im_rgb,True)
		sink.sink(predicted)
		im_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		read, frame = source.read()
		count += 1
	
	sink.close()
	source.close()

def main_image(args, debug=False):
	model = DetectWaste()
	
	source = np.asarray(Image.open(args.file))[:,:,:3]
	sink = args.file.replace(".jpg","_pred.jpg").replace(".png","_pred.jpg")
	_category_index = {1:"Litter"}
	
	model.initialize()
	img = model.preprocess(source)
	predicted = model.predict(img)
	detection = model.postprocess(predicted)
	bboxes = detection[..., [1,0,3,2]]

	print(detection)
	scores = detection[..., 4]
	classes = detection[..., 5].astype(int)
	label_id_offset = 0
	image_np_with_detections = source.copy()
	vis_utils.visualize_boxes_and_labels_on_image_array(
		image_np_with_detections,
		bboxes,
		classes,
		scores,
		_category_index,
		use_normalized_coordinates=False,
		max_boxes_to_draw=200,
		min_score_thresh=0.25,
		agnostic_mode=False,
	)

	Image.fromarray(image_np_with_detections).save(sink)



if __name__ == '__main__':
	parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	if args.file.endswith(".jpg") or args.file.endswith(".png"):
		main_image(args)
	else:
		main_video(args)