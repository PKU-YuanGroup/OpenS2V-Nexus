import sys

sys.path.append("util_codes/gme_model")
sys.path.append("util_codes/groundingsam2")
sys.path.append("util_codes/lama_with_maskdino/lama_with_refiner")

import argparse
import json
import os
from typing import Tuple

import cv2
import grounding_dino.groundingdino.datasets.transforms as T
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from grounding_dino.groundingdino.util.inference import load_model, predict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from diffusers.training_utils import free_memory


from typing import List

import torch.nn as nn
from omegaconf import OmegaConf
from scipy.ndimage import binary_dilation, binary_fill_holes
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection,
)

from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint
from simple_lama_inpainting.utils.util import prepare_img_and_mask


from gme_inference import GmeQwen2VL

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image_pil)
    image_transformed, _ = transform(image_pil, None)
    return image_np, image_transformed, image_pil


class ImageDataset(Dataset):
    def __init__(self, json_files, offload_images_folder, output_path):
        self.json_files = list(json_files.items())
        self.offload_images_folder = offload_images_folder
        self.output_path = output_path

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        try:
            key, value = self.json_files[idx]
            local_image_folder = os.path.join(self.offload_images_folder, key).replace(
                "_step5", ""
            )

            metadata = value["metadata"]
            texts = (
                ". ".join(
                    ". ".join(tags).strip()
                    for tags in value["word_tags"].values()
                    if tags
                ).lower()
                + "."
            )

            # load offload images
            image_files = sorted(
                [f for f in os.listdir(local_image_folder) if f.endswith(".png")],
                key=lambda x: int(os.path.splitext(x)[0]),
            )

            image_np_list = []
            image_transformed_list = []
            image_pil_list = []
            frame_indices = []
            for image_file in image_files:
                frame_indice = int(os.path.splitext(image_file)[0])
                image_path = os.path.join(local_image_folder, image_file)
                image_np, image_transformed, image_pil = load_image(image_path)
                image_np_list.append(image_np)
                image_transformed_list.append(image_transformed)
                image_pil_list.append(image_pil)
                frame_indices.append(frame_indice)

            return {
                "image_nps": image_np_list,
                "image_transformed_lists": image_transformed_list,
                "image_pils": image_pil_list,
                "frame_indices": frame_indices,
                "texts": texts,
                "metadata": metadata,
                "json_key": key,
                "raw_data": value,
            }
        except Exception:
            return None


def custom_collate_fn(batch):
    return [b for b in batch if b is not None]


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def rle_to_mask(rle, img_width, img_height):
    rle_obj = {"counts": rle["counts"].encode("utf-8"), "size": [img_height, img_width]}
    return mask_util.decode(rle_obj)


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(1,))
    return mask


def get_inpaint_model(base_name="LaMa"):
    predict_config = OmegaConf.load(os.path.join(base_name, "default.yaml"))
    predict_config.model.path = base_name
    predict_config.refiner.gpu_ids = "0"

    device = torch.device(predict_config.device)
    train_config_path = os.path.join(base_name, "big-lama/config.yaml")

    train_config = OmegaConf.load(train_config_path)
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    checkpoint_path = os.path.join(base_name, "big-lama/models/best.ckpt")

    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location="cpu"
    )
    model.freeze()
    model.to(device)
    return model, predict_config


@torch.no_grad
def run_aesthetic_laion(model, image):
    if not isinstance(image, list):
        image = [image]
    return model(image)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, aes_clip_path, aes_main_path):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(aes_clip_path)
        self.processor = CLIPProcessor.from_pretrained(aes_clip_path)

        self.mlp = MLP()
        state_dict = torch.load(
            aes_main_path, weights_only=True, map_location=torch.device("cpu")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip(**inputs)[0]
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def gme_aes_inpaint(
    json_all_data,
    tag_json_data,
    mask_datas,
    class_names,
    input_image,
    foreground_dir,
    json_key,
    gme_model=None,
    aes_model=None,
    inpaint_model=None,
    predict_config=None,
):
    # Extract relevant class names from tag_json_data for Inpaint
    if "pre_define" not in tag_json_data.keys():
        tag_json_data["pre_define"] = ["person", "human head"]
    valid_classes = set(
        tag_json_data["pre_define"] + tag_json_data["subject"]
        # + tag_json_data["object"]
    )

    input_image = np.array(input_image)
    if input_image.dtype != np.uint8:
        input_image = input_image.astype(np.uint8)
    input_image = input_image[..., ::-1]
    img_width = input_image.shape[1]
    img_height = input_image.shape[0]

    combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    images = []
    texts = []
    annotations = []
    for i, (mask, class_name) in enumerate(zip(mask_datas, class_names)):
        if class_name in valid_classes:
            combined_mask += mask.astype(np.uint8)

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

        # Add to batch processing lists
        images.append(pil_image)
        texts.append(f"{class_name}.")
        annotations.append((i, class_name, resized_image))

    with torch.no_grad():
        # For GME Score
        text_pooled_output = gme_model.get_text_embeddings(
            texts=texts,
            instruction="Find an image that matches the given text.",
            show_progress_bar=False,
        )
        image_pooled_output = gme_model.get_image_embeddings(
            images=images, is_query=False, show_progress_bar=False
        )
        gme_scores = (text_pooled_output * image_pooled_output).sum(-1)
        # For Aesthetic Score
        aes_scores = run_aesthetic_laion(aes_model, images)

    # Save result
    for idx, (i, class_name, resized_image) in enumerate(annotations):
        gme_score = float(gme_scores[idx])
        aes_score = float(aes_scores[idx].detach().item())

        json_all_data["annotation"]["ann_frame_data"]["annotations"][i]["gme_score"] = (
            gme_score
        )
        json_all_data["annotation"]["ann_frame_data"]["annotations"][i]["aes_score"] = (
            aes_score
        )

        # For visualization
        # if True:
        #     crop_height, crop_width = resized_image.shape[:2]
        #     original_area = img_height * img_width
        #     crop_area = crop_height * crop_width
        #     crop_ratio = crop_area / original_area

        #     temp_output_dir = f"output_step7/{json_key}"
        #     os.makedirs(temp_output_dir, exist_ok=True)
        #     temp_output_path = os.path.join(
        #         temp_output_dir,
        #         f"{class_name}_clip{gme_score:.3f}_ase{aes_score:.3f}_ratio{crop_ratio:.4f}.png",
        #     )
        #     cv2.imwrite(temp_output_path, resized_image)
        #     print(f"Saved {temp_output_path}, {gme_score}, {aes_score}")

    structure = np.ones((20, 20), dtype=np.uint8)
    kernel = np.ones((4, 4), np.uint8)
    try:
        combined_mask = binary_dilation(combined_mask, structure=structure).astype(
            np.uint8
        )
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=3)
        combined_mask = binary_fill_holes(combined_mask).astype(float)
    except Exception as e:
        print(e)
        pass

    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
    forground_image = input_image

    extracted_image = np.zeros_like(input_image)
    extracted_image[combined_mask == 0] = input_image[combined_mask == 0]
    extracted_image = extracted_image[..., ::-1]

    if combined_mask.sum() / (input_image.shape[0] * input_image.shape[1]) > 0.45:
        size = (768, 768)
    else:
        size = (input_image.shape[1], input_image.shape[0])
    input_mask_image = Image.fromarray(combined_mask * 255).convert("RGB").resize(size)
    input_image = Image.fromarray(extracted_image).convert("RGB").resize(size)

    # For visualization
    # input_mask_image.save(f"{output_json_folder}/extracted_image_mask_{json_key}.png")
    # input_image.save(f"{output_json_folder}/extracted_image_frame_{json_key}.png")

    # Lama with refine
    input_mask_image = input_mask_image.convert("L")
    img, masks = prepare_img_and_mask(input_image, input_mask_image, device="cpu")
    batch = {"image": img[0], "mask": masks[0][0][None, ...]}
    batch["unpad_to_size"] = [
        torch.tensor([batch["image"].shape[1]]),
        torch.tensor([batch["image"].shape[2]]),
    ]
    batch["image"] = torch.tensor(
        pad_img_to_modulo(batch["image"], predict_config.dataset.pad_out_to_modulo)
    )[None].to(predict_config.device)
    batch["mask"] = (
        torch.tensor(
            pad_img_to_modulo(batch["mask"], predict_config.dataset.pad_out_to_modulo)
        )[None]
        .float()
        .to(predict_config.device)
    )
    cur_res = refine_predict(batch, inpaint_model, **predict_config.refiner)
    cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    background_image = Image.fromarray(cur_res)

    background_image = background_image.resize((img_width, img_height))

    # Get Aes Similarity
    aes_score = run_aesthetic_laion(aes_model, [forground_image, background_image])

    # if aes_score > 5.0:
    if not os.path.exists(foreground_dir):
        os.makedirs(foreground_dir)
    final_image_path = os.path.join(foreground_dir, f"{json_key}_step7.png")
    background_image.save(final_image_path)

    json_all_data["annotation"]["ann_frame_data"]["foreground_ase_score"] = float(
        aes_score[0].detach().item()
    )
    json_all_data["annotation"]["ann_frame_data"]["background_ase_score"] = float(
        aes_score[1].detach().item()
    )

    return json_all_data


def inference_all(gd_model, sam_model, texts, image_np, image_transformed):
    """
    GroundingSAM inference
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
        boxes, confidences, class_names = predict(
            model=gd_model,
            image=image_transformed,
            caption=texts,
            box_threshold=0.25,
            text_threshold=0.25,
            remove_combined=True,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_np.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    """
    SAM inference
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_model.set_image(image_np)
        masks, scores, logits = sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    return input_boxes, class_names, confidences, masks, scores


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def post_process_all(
    input_boxes,
    class_names,
    scores,
    masks,
    h,
    w,
    raw_data,
    input_image,
    foreground_dir,
    json_key,
    inpaint_model,
    gme_model,
    aes_model,
    predict_config,
    ann_frame_idx=0,
):
    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    all_frames_results = {"ann_frame_data": {}, "mask_map": {}, "mask_annotation": {}}
    input_boxes = input_boxes.tolist()

    # Get Bbox
    all_frames_results["ann_frame_data"] = {
        "ann_frame_idx": int(ann_frame_idx),
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box,
            }
            for class_name, box in zip(class_names, input_boxes)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }

    # Get Mask Map
    all_frames_results["mask_map"] = {
        f"{int(i + 1)}": {"class_name": class_name}
        for i, class_name in enumerate(class_names)
    }

    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    scores = scores.tolist()
    # save the results in standard format
    for out_obj_id, (mask_rle, score) in enumerate(zip(mask_rles, scores)):
        if int(ann_frame_idx) not in all_frames_results["mask_annotation"]:
            all_frames_results["mask_annotation"][int(ann_frame_idx)] = {}
        all_frames_results["mask_annotation"][int(ann_frame_idx)][out_obj_id + 1] = {
            "size": mask_rle["size"],
            "counts": mask_rle["counts"],
            "score": score,
        }

    raw_data["annotation"] = all_frames_results

    raw_data = gme_aes_inpaint(
        json_all_data=raw_data,
        tag_json_data=raw_data["word_tags"],
        mask_datas=masks,
        class_names=class_names,
        input_image=input_image,
        foreground_dir=foreground_dir,
        json_key=json_key,
        gme_model=gme_model,
        aes_model=aes_model,
        inpaint_model=inpaint_model,
        predict_config=predict_config,
    )

    return raw_data


def parse_args():
    parser = argparse.ArgumentParser()

    # all
    parser.add_argument(
        "--input_video_json",
        default="demo_result/step4/merge_final_json/dataset1.json",
    )
    parser.add_argument(
        "--offload_images_folder",
        type=str,
        default="demo_result/step5/temp_offload_images",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        default="demo_result/step5/final_output/dataset1",
    )
    # sam2.1
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="OpenS2V-Weight/sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--sam_model_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    # grounding dino
    parser.add_argument(
        "--gd_config",
        default="util_codes/groundingsam2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        type=str,
    )
    parser.add_argument(
        "--gd_checkpoint",
        default="OpenS2V-Weight/groundingdino_swint_ogc.pth",
        type=str,
    )
    # gme model
    parser.add_argument(
        "--gme_score_model_path",
        type=str,
        default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    )
    # aesthetic predictor
    parser.add_argument(
        "--aes_clip_path",
        type=str,
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--aes_main_path",
        type=str,
        default="OpenS2V-Weight/aesthetic-model.pth",
    )
    # inpaint model
    parser.add_argument(
        "--inpaint_model_path",
        type=str,
        default="OpenS2V-Weight/LaMa",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--part", default=0, type=int)
    parser.add_argument("--total_part", default=1, type=int)
    return parser.parse_args()


def split_list(data, nums, part):
    items = list(data.items())
    size = len(items)
    part_size = size // nums
    remainder = size % nums

    start = part * part_size + min(part, remainder)
    end = start + part_size + (1 if part < remainder else 0)

    return dict(items[start:end])


if __name__ == "__main__":
    args = parse_args()

    device = "cuda"

    sam_checkpoint = args.sam_checkpoint
    sam_model_cfg = args.sam_model_cfg
    gd_checkpoint = args.gd_checkpoint
    gd_config = args.gd_config
    gme_score_model_path = args.gme_score_model_path
    aes_clip_path = args.aes_clip_path
    aes_main_path = args.aes_main_path

    offload_images_folder = args.offload_images_folder
    output_json_folder = args.output_json_folder
    input_video_json = args.input_video_json

    total_part = args.total_part
    part = args.part

    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder, exist_ok=True)

    with open(input_video_json, "r") as f:
        data = json.load(f)

    split_data = split_list(data, args.total_part, args.part)

    """
    Load GroundingDINO
    """
    grounding_model = load_model(
        model_config_path=gd_config, model_checkpoint_path=gd_checkpoint, device=device
    )

    """
    Load SAM2.1
    """
    sam_model = SAM2ImagePredictor(
        build_sam2(sam_model_cfg, sam_checkpoint, device=device)
    )

    """
    Load Gme Model
    """
    gme_model = GmeQwen2VL(gme_score_model_path, attn_model="flash_attention_2")

    """
    Load Ase Model
    """
    aes_model = AestheticScorer(
        dtype=torch.float32, aes_clip_path=aes_clip_path, aes_main_path=aes_main_path
    ).to(device)

    """
    Load Inpaint Model
    """
    inpaint_model, predict_config = get_inpaint_model(args.inpaint_model_path)
    inpaint_model.to(device)

    """
    Main Loop DataLoader
    """
    batch_size = args.batch_size
    dataset = ImageDataset(split_data, offload_images_folder, output_json_folder)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=None if args.num_workers == 0 else 2,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    assert batch_size == 1

    for batch in tqdm(dataloader, desc="Processing Files", total=len(dataloader)):
        if batch is None or len(batch) == 0:
            print("skipping bad sample!")
            continue

        batch = batch[0]
        json_key = batch["json_key"]
        data = batch["raw_data"]
        frame_indices = batch["frame_indices"]
        texts = batch["texts"]
        metadata = batch["metadata"]

        image_nps = batch["image_nps"]
        image_transformed_lists = batch["image_transformed_lists"]
        image_pils = batch["image_pils"]

        output_json_path = os.path.join(output_json_folder, f"{json_key}_step6.json")
        if os.path.exists(output_json_path):
            print("skipping!")
            continue

        foreground_dir = os.path.join(output_json_folder, "foreground")
        os.makedirs(foreground_dir, exist_ok=True)

        for image_np, image_transformed, frame_indice, image_pil in zip(
            image_nps, image_transformed_lists, frame_indices, image_pils
        ):
            h, w, _ = image_np.shape

            """
            Main Inference
            """
            try:
                input_boxes, class_names, confidences, masks, scores = inference_all(
                    gd_model=grounding_model,
                    sam_model=sam_model,
                    texts=texts,
                    image_np=image_np,
                    image_transformed=image_transformed,
                )
            except Exception as e:
                print(e)
                print("no available detect, continue")
                continue

            """
            Post-process and Save
            """
            assert batch_size == 1
            results = None
            try:
                results = post_process_all(
                    input_boxes=input_boxes,
                    class_names=class_names,
                    scores=scores,
                    masks=masks,
                    h=h,
                    w=w,
                    input_image=image_pil,
                    raw_data=data,
                    foreground_dir=foreground_dir,
                    json_key=json_key,
                    inpaint_model=inpaint_model,
                    gme_model=gme_model,
                    aes_model=aes_model,
                    predict_config=predict_config,
                    ann_frame_idx=frame_indice,
                )
            except Exception as e:
                results = None
                print(e)
                print("no available result, continue")
                continue

        # data["annotation"] = results
        if results is not None:
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=4)

        free_memory()
