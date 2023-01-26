import argparse
import cv2
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import vis_utils 
import json

def get_args_parser():
	parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
	
	parser.add_argument('--file', default='',
						help='file to predict')
	parser.add_argument('--model', default='',
						help='path where the trained weights saved')
	
	return parser

class TFFrozenDetector:
	def __init__(self, path):
		
		self._pb_file = f"{path}/"
		self._model = None
		self._category_index = None

	def load(self):
		self._model = tf.saved_model.load(self._pb_file)
		with open(f"{self._pb_file}label_map.json") as f:
			self._category_index = {int(k): v for k,v in json.load(f).items()}

	def predict(self, frame, actuate=True):

		input_tensor = np.expand_dims(frame, axis=0)
		detections = self._model(input_tensor)

		boxes, scores = (
			detections["detection_boxes"][0],
			detections["detection_scores"][0],
		)
		num_detections = int(detections.pop("num_detections"))

		selected_indices = (
			tf.image.non_max_suppression(boxes, scores, num_detections, 0.5)
			.numpy()
			.tolist()
		)

		detections = {
			key: value[0].numpy()[selected_indices] for key, value in detections.items()
		}

		detections["num_detections"] = num_detections

		if not actuate:
			return detections

		detections["detection_classes"] = detections["detection_classes"].astype(
			np.int64
		)
		label_id_offset = 0
		image_np_with_detections = frame.copy()
		vis_utils.visualize_boxes_and_labels_on_image_array(
			image_np_with_detections,
			detections["detection_boxes"],
			detections["detection_classes"],
			detections["detection_scores"],
			self._category_index,
			use_normalized_coordinates=True,
			max_boxes_to_draw=200,
			min_score_thresh=0.20,
			agnostic_mode=False,
		)

		return image_np_with_detections

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


def main_video(args, debug=False):
	model = TFFrozenDetector(args.model)
	source = FileSource(args.file)
	sink = FileSink(args.file.replace(".mp4","_pred.mp4").replace(".mov","_pred.mov"))

	frames = []
	source.open()
	sink.open(source.frames_per_second,source.width,source.height)
	model.load()
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
	model = TFFrozenDetector(args.model)
	source = np.asarray(Image.open(args.file))
	sink = args.file.replace(".jpg","_pred.jpg")
	model.load()
		
	predicted = model.predict(source,True)
	Image.fromarray(predicted).save(sink)


if __name__ == '__main__':
	parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	if args.file.endswith(".jpg"):
		main_image(args)
	else:
		main_video(args)