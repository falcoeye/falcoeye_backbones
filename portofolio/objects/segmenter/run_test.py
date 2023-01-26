
import click
from model import SegmenterLoader
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as F
from utils import resize,sliding_window,merge_windows
import torch
import numpy as np

basedir = os.path.abspath(os.path.dirname(__file__))
STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}
normalization = STATS["vit"]
def preprocess(data,n_cls,window_size,window_stride):
    
    images = [] 
    for i,filename in enumerate(data):
        print(f"Processing  {i}/{len(data)} {filename}")
        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        pil_im = Image.open(filename).copy()
        im = F.pil_to_tensor(pil_im).float() / 255
        im = F.normalize(im, normalization["mean"], normalization["std"])
        im = im.to("cpu").unsqueeze(0)
        ori_shape = im.shape[2:4]
        im = resize(im, window_size)
        windows = sliding_window(im, False, window_size, window_stride)
        crops = torch.stack(windows.pop("crop"))[:, 0]
        seg_map = torch.zeros((n_cls, ori_shape[0], ori_shape[1]), device="cpu")
        images.append([windows,crops,seg_map,ori_shape])
    return images

def inference(model,n_cls,window_size,data):
    logits = []
    for i,(windows,crops,seg_map,ori_shape) in enumerate(data):
        print(f"Inferring {i}/{len(data)}")
        B = len(crops)
        WB = 2 # bach size
        seg_maps = torch.zeros((B, n_cls, window_size, window_size), device="cpu")
        with torch.no_grad():
            for i in range(0, B, WB):
                seg_maps[i : i + WB] = model.forward(crops[i : i + WB])
        windows["seg_maps"] = seg_maps
        im_seg_map = merge_windows(windows, window_size, ori_shape)
        seg_map = im_seg_map.argmax(0)

        logits.append(seg_map.detach().cpu().numpy().astype(np.float32))
    return logits

@click.command()
@click.option("--input-dir", "-i", type=str, help="folder with input images")
@click.option("--output-dir", "-o", type=str, help="folder with output images")
def main(input_dir, output_dir):
    modelLoader = SegmenterLoader(f"{basedir}/checkpoints/checkpoint.pth")
    model,variant = modelLoader.load()
    window_size = variant["inference_kwargs"]["window_size"]
    window_stride = variant["inference_kwargs"]["window_stride"]
    n_cls = model.n_cls
    model.to("cpu")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    list_dir = list(input_dir.iterdir())
    images = preprocess(list_dir[:1],n_cls,window_size,window_stride)
    logits = inference(model,n_cls,window_size,images)
    print(logits[0].shape)


if __name__ == "__main__":
    main()