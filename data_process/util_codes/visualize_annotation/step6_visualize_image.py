import argparse
import gc
import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm


def rle_to_mask(rle, img_width, img_height):
    rle_obj = {"counts": rle["counts"].encode("utf-8"), "size": [img_height, img_width]}
    return mask_util.decode(rle_obj)


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(1,))

    return mask


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def extract_and_save_masks(
    input_video_root,
    key,
    value,
    output_image_folder,
):
    # Load the and JSON data
    metadata = value["metadata"]
    annotation_data = value["annotation"]

    crop = metadata["crop"]
    class_names = annotation_data["mask_map"]
    frame_idx = annotation_data["ann_frame_data"]["ann_frame_idx"]
    video_path = os.path.join(input_video_root, metadata["path"])

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # decord
    try:
        vr = VideoReader(video_path)
        input_image = (
            vr.get_batch([int(frame_idx)]).asnumpy()[0][..., ::-1].astype(np.uint8)
        )
        s_x, e_x, s_y, e_y = crop
        input_image = input_image[s_y:e_y, s_x:e_x]
        del vr
    except Exception as e:
        print(f"load video {video_path} error: {e}")
        return
    gc.collect()

    img_width = input_image.shape[1]
    img_height = input_image.shape[0]
    mask_data = annotation_data["mask_annotation"][str(frame_idx)]
    for i, annotation_idx in enumerate(mask_data):
        class_name = class_names[f"{annotation_idx}"]["class_name"]
        mask_rle = mask_data[annotation_idx]
        mask = rle_to_mask(mask_rle, img_width, img_height)

        # refine mask
        # from scipy.ndimage import binary_dilation, binary_fill_holes
        # structure = np.ones((4, 4), dtype=np.uint8)
        # kernel = np.ones((4, 4), np.uint8)
        # mask = binary_dilation(mask, structure=structure).astype(np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=3)
        # mask = binary_fill_holes(mask).astype(float)
        # mask = refine_masks(torch.tensor(mask).unsqueeze(0).unsqueeze(0), True)[0]

        # Find the bounding box of the mask
        rows, cols = np.where(mask == 1)
        if len(rows) == 0 or len(cols) == 0:  # Skip if the mask is empty
            continue
        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)

        #################################################################
        # Adjust if the region goes out of bounds
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, img_width - 1)
        y_max = min(y_max, img_height - 1)

        # Crop the region from the original image and mask
        cropped_image = input_image[y_min : y_max + 1, x_min : x_max + 1]
        cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

        # Create a white background of the same size as the crop
        white_background = np.ones_like(cropped_image) * 255

        # Apply the mask to the cropped image
        white_background[cropped_mask == 1] = cropped_image[cropped_mask == 1]
        resized_image = white_background
        #################################################################

        # Convert the image to PIL format for CLIP processing
        pil_image = Image.fromarray(
            cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        ).convert("RGB")

        if True:
            os.makedirs(os.path.join(output_image_folder, key), exist_ok=True)
            crop_area = pil_image.size[0] * pil_image.size[1]
            original_area = img_height * img_width
            crop_ratio = crop_area / original_area
            temp_output_path = os.path.join(
                output_image_folder, key, f"{class_name}_{i}_ratio{crop_ratio:.4f}.png"
            )
            cv2.imwrite(temp_output_path, resized_image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse arguments for the video processing and model paths."
    )
    parser.add_argument(
        "--input_video_root",
        type=str,
        default="../../demo_result/step0/videos/dataset1",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--input_video_json",
        type=str,
        default="../../demo_result/step5/merge_final_json/dataset1.json",
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output_image_folder",
        type=str,
        default="./step6/images",
        help="Directory where output masks will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Assign paths
    input_video_root = args.input_video_root
    input_video_json = args.input_video_json
    output_image_folder = args.output_image_folder

    with open(input_video_json, "r") as f:
        all_data = json.load(f)
    data_items = list(all_data.items())

    # for json_file in tqdm(json_files):
    for key, data in tqdm(data_items, desc="Processing Files", total=len(data_items)):
        extract_and_save_masks(
            input_video_root,
            key,
            data,
            output_image_folder,
        )
