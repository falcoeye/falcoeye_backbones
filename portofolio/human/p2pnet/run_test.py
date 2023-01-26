import argparse
import datetime
import random
import time
from pathlib import Path
import cv2
import torch
import torchvision.transforms as standard_transforms
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2
from model import P2PNet
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def get_args_parser():
	parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
	
	# * Backbone
	parser.add_argument('--backbone', default='vgg16_bn', type=str,
						help="name of the convolutional backbone to use")

	parser.add_argument('--row', default=2, type=int,
						help="row number of anchor points")
	parser.add_argument('--line', default=2, type=int,
						help="line number of anchor points")

	parser.add_argument('--output_dir', default='./',
						help='path where to save')
	parser.add_argument('--file', default='',
						help='file to predict')
	parser.add_argument('--weight_path', default='',
						help='path where the trained weights saved')

	parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
	

	return parser


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



def main(args, debug=False):

	#os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

	print(args)
	device = torch.device('cpu')
	# get the P2PNet
	model = P2PNet()
	# move to GPU
	model.to(device)
	# load trained model
	if args.weight_path is not None:
		checkpoint = torch.load(args.weight_path, map_location='cpu')
		model.load_state_dict(checkpoint['model'])
	
	# convert to eval mode
	model.eval()
	print("Model loaded")
	# create the pre-processing transform
	transform = standard_transforms.Compose([
		standard_transforms.ToTensor(), 
		standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	source = FileSource(args.file)
	sink = FileSink(args.file.replace(".mp4","_pred.mp4").replace(".mov","_pred.mov"))

	frames = []
	source.open()
	sink.open(source.frames_per_second,384,256)
	
	# # set your image path here
	# img_path = args.file
	# # load the images
	# img_raw = Image.open(img_path).convert('RGB')
	read, frame = source.read()
	print(frame.shape)
	count = 0
	while read:
		print(f"{count}/{source.num_frames}")
		img_raw = Image.fromarray(frame[:,:,::-1])
		# round the size
		width, height = img_raw.size
		new_width = width // 128 * 128
		new_height = height // 128 * 128
		img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
		# pre-proccessing
		img = transform(img_raw)

		samples = torch.Tensor(img).unsqueeze(0)
		samples = samples.to(device)
		# run inference
		outputs = model(samples)
		outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

		outputs_points = outputs['pred_points'][0]


		threshold = 0.5
		# filter the predictions
		points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
		predict_cnt = int((outputs_scores > threshold).sum())

		outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

		outputs_points = outputs['pred_points'][0]
		

		# draw the predictions
		size = 2
		img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
		for p in points:
			img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
		sink.sink(img_to_draw)
		
		read, frame = source.read()
		count += 1
	
		
	sink.close()
	source.close()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	main(args)