import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )

def get_clicked_point(img_path):
    def onclick(event):
        global coords
        coords = [int(event.xdata), int(event.ydata)]
        print(f"Clicked coordinates: {coords}")
        plt.close()

    img = Image.open(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return coords
import matplotlib
matplotlib.use('TkAgg')

def get_multiple_clicked_points(img_path, max_points):
    def onclick(event):
        nonlocal coords_list
        if event.dblclick:
            plt.close()
        else:
            coords = [int(event.xdata), int(event.ydata)]
            print(f"Clicked coordinates: {coords}")
            coords_list.append(coords)
            if len(coords_list) >= max_points:
                plt.close()

    coords_list = []
    img = Image.open(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return coords_list


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.input_img = './input/6.jpg'
            self.coords_type = 'click'
            self.point_coords = [200, 450]  # 控制坐标点
            self.point_labels = [1]
            self.dilate_kernel_size = 15
            self.output_dir = './results'
            self.sam_model_type = 'vit_h'
            self.sam_ckpt = './pretrained_models/sam_vit_h_4b8939.pth'
            self.lama_config = './lama/configs/prediction/default.yaml'
            self.lama_ckpt = './pretrained_models/big-lama'
            self.max_points = 6


    args = Args()

    parser = argparse.ArgumentParser()
    setup_args(parser)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.coords_type == "click":
        latest_coords = get_multiple_clicked_points(args.input_img, args.max_points)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    img = load_img_to_array(args.input_img)

    point_labels = [1] * len(latest_coords)

    masks, _, _ = predict_masks_with_sam(
        img,
        latest_coords,
        point_labels,  # Use the updated point_labels list
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )

    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)
