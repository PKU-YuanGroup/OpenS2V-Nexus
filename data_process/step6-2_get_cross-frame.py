import os
import gc
import json
import argparse
from tqdm import tqdm
from PIL import Image
from pycocotools import mask as mask_util
import numpy as np

import sys

sys.path.append("util_codes/gme_model")
from gme_inference import GmeQwen2VL
from decord import VideoReader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def rle_to_mask(rle, img_width, img_height):
    rle_obj = {"counts": rle["counts"].encode("utf-8"), "size": [img_height, img_width]}
    return mask_util.decode(rle_obj)


def extract_video_frame(input_video_root, cur_data, cur_frame_idx):
    try:
        video_path = os.path.join(input_video_root, cur_data["metadata"]["path"])
        vr = VideoReader(video_path)
        frame = vr[int(cur_frame_idx)]
        s_x, e_x, s_y, e_y = cur_data["metadata"]["crop"]
        cur_image = frame.asnumpy()[s_y:e_y, s_x:e_x]
        cur_pil_image = Image.fromarray(cur_image)
        del vr
        gc.collect()
        return cur_pil_image
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None  # Return None if there was an error


def extract_subject_image(pil_image, mask_rle, img_width, img_height):
    img_array = np.array(pil_image)
    mask = rle_to_mask(mask_rle, img_width, img_height).astype(bool)
    subject_image_array = np.zeros_like(img_array)
    subject_image_array[mask] = img_array[mask]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_start, row_end = np.argmax(rows), len(rows) - np.argmax(rows[::-1]) - 1
    col_start, col_end = np.argmax(cols), len(cols) - np.argmax(cols[::-1]) - 1
    subject_image_array = subject_image_array[
        row_start : row_end + 1, col_start : col_end + 1
    ]
    return Image.fromarray(subject_image_array)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_cluster_json",
        default="demo_result/step6/cluster_videos.json",
    )
    parser.add_argument(
        "--input_video_json",
        default="demo_result/step5/merge_final_json/dataset1_cluster.json",
    )
    parser.add_argument(
        "--input_video_root",
        type=str,
        default="demo_result/step0/videos/dataset1",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        default="demo_result/step6/cross-frames-images/",
    )
    parser.add_argument(
        "--gme_score_model_path",
        type=str,
        default="/mnt/workspace/checkpoints/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_cluster_json = args.input_cluster_json
    input_video_root = args.input_video_root
    input_video_json = args.input_video_json
    output_json_folder = args.output_json_folder
    gme_score_model_path = args.gme_score_model_path

    os.makedirs(output_json_folder, exist_ok=True)

    text_similarity_threshold = 0.6
    imgae_similarity_threshold = 0.6

    with open(input_cluster_json, "r") as f:
        cluster_data = json.load(f)
    cluster_data_items = list(cluster_data.items())

    with open(input_video_json, "r") as f:
        all_data = json.load(f)

    gme_model = GmeQwen2VL(gme_score_model_path, attn_model="flash_attention_2")

    
    for key, data in tqdm(
        cluster_data_items, desc="Processing Files", total=len(cluster_data_items)
    ):
        temp_json_path = os.path.join(output_json_folder, f"{key}.json")
        if os.path.exists(temp_json_path):
            print(f"already process {key}")
            return
        temp_match_data = []
        for cur_idx in range(len(data)):
            cur_data = all_data[data[cur_idx]]
            cur_frame_idx = cur_data["annotation"]["ann_frame_data"]["ann_frame_idx"]
            cur_pil_image = extract_video_frame(
                input_video_root, cur_data, cur_frame_idx
            )
            cur_img_width = cur_pil_image.size[0]
            cur_img_height = cur_pil_image.size[1]

            for aft_idx in range(cur_idx + 1, len(data)):
                aft_data = all_data[data[aft_idx]]
                aft_frame_idx = aft_data["annotation"]["ann_frame_data"][
                    "ann_frame_idx"
                ]
                aft_pil_image = extract_video_frame(
                    input_video_root, aft_data, aft_frame_idx
                )
                aft_img_width = aft_pil_image.size[0]
                aft_img_height = aft_pil_image.size[1]

                cur_class_names = {
                    key: value["class_name"]
                    for key, value in cur_data["annotation"]["mask_map"].items()
                }
                aft_class_names = {
                    key: value["class_name"]
                    for key, value in aft_data["annotation"]["mask_map"].items()
                }
                cur_class_name_list = list(cur_class_names.values())
                aft_class_name_list = list(aft_class_names.values())
                e_text_cur = gme_model.get_text_embeddings(
                    texts=cur_class_name_list,
                    instruction="Find an image that matches the given text.",
                    show_progress_bar=False,
                )
                e_text_aft = gme_model.get_text_embeddings(
                    texts=aft_class_name_list,
                    instruction="Find an image that matches the given text.",
                    show_progress_bar=False,
                )

                cur_subject_images = []
                aft_subject_images = []
                for cur_id, cur_class_name in cur_class_names.items():
                    cur_mask_rle = cur_data["annotation"]["mask_annotation"][
                        f"{cur_frame_idx}"
                    ].get(str(cur_id), {})
                    cur_subject_image = extract_subject_image(
                        cur_pil_image, cur_mask_rle, cur_img_width, cur_img_height
                    )
                    cur_subject_images.append(cur_subject_image)
                for aft_id, aft_class_name in aft_class_names.items():
                    aft_mask_rle = aft_data["annotation"]["mask_annotation"][
                        f"{aft_frame_idx}"
                    ].get(str(aft_id), {})
                    aft_subject_image = extract_subject_image(
                        aft_pil_image, aft_mask_rle, aft_img_width, aft_img_height
                    )
                    aft_subject_images.append(aft_subject_image)
                e_image_cur = gme_model.get_image_embeddings(
                    images=cur_subject_images, is_query=False, show_progress_bar=False
                )
                e_image_aft = gme_model.get_image_embeddings(
                    images=aft_subject_images, is_query=False, show_progress_bar=False
                )

                for cur_id, cur_class_name in cur_class_names.items():
                    for aft_id, aft_class_name in aft_class_names.items():
                        text_similarity = (
                            e_text_cur[int(cur_id) - 1] * e_text_aft[int(aft_id) - 1]
                        ).sum(-1)

                        if text_similarity >= text_similarity_threshold:
                            print(
                                f"Object {cur_class_name} (ID: {cur_id}) in cur_data is similar to object {aft_class_name} (ID: {aft_id}) in aft_data"
                            )

                            image_similarity = (
                                e_image_cur[int(cur_id) - 1]
                                * e_image_aft[int(aft_id) - 1]
                            ).sum(-1)

                            if image_similarity > imgae_similarity_threshold:
                                temp_data = {
                                    "cur_id": data[cur_idx],
                                    "aft_id": data[aft_idx],
                                    "cur_class_name": cur_class_name,
                                    "aft_class_name": aft_class_name,
                                    "text_similarity": text_similarity.item(),
                                    "image_similarity": image_similarity.item(),
                                    "cur_frame_idx": cur_frame_idx,
                                    "aft_frame_idx": aft_frame_idx,
                                }
                                temp_match_data.append(temp_data)

        with open(temp_json_path, "w") as f:
            json.dump(temp_match_data, f, indent=2)

        print(f"Match data saved to {output_json_folder}")


if __name__ == "__main__":
    main()
