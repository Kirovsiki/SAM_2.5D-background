import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
import torch
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')

from segment_anything import SamPredictor, sam_model_registry
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points


class ClickPoints:
    def __init__(self, img_path):
        self.img_path = img_path
        self.coords = []

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        print(f'x = {ix}, y = {iy}')
        self.coords.append([ix, iy])

        if len(self.coords) == 7:  #
            plt.close()

    import numpy as np

    class PointPicker:
        def __init__(self, img):
            self.img = img
            self.coords = []

        def onclick(self, event):
            if len(self.coords) < 5:  # Limit to 5 points
                ix, iy = event.xdata, event.ydata
                print(f'click: ({ix}, {iy})')
                self.coords.append([ix, iy])

        def show(self):
            fig = plt.figure()
            cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
            plt.imshow(self.img)
            plt.show()  # This will block
            fig.canvas.mpl_disconnect(cid)
            return self.coords

    if __name__ == "__main__":
        img = np.random.rand(10, 10)
        point_picker = PointPicker(img)
        point_picker.show()
        print(point_picker.coords)

    def select_points(self):
        img = Image.open(self.img_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
        return self.coords


def predict_masks_with_sam(
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        model_type: str,
        ckpt_p: str,
        device="cuda"
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits


class Args:
    def __init__(self):
        self.input_img = "./input/5.jpg"
        self.dilate_kernel_size = 15
        self.output_dir = "./output"
        self.sam_model_type = "vit_h"
        self.sam_ckpt = "./pretrained_models/sam_vit_h_4b8939.pth"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Args()

    click_points = ClickPoints(args.input_img)
    coords = click_points.select_points()

    labels = [1] * len(coords)

    img = load_img_to_array(args.input_img)

    masks, _, _ = predict_masks_with_sam(
        img,
        coords,
        labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points_{idx}.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        save_array_to_img(mask, mask_p)

        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi, height/dpi))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), coords, labels, size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()


