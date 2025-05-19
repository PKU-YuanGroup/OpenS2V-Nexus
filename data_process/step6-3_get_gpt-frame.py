import re
import os
import gc
import json
import random
import argparse
from tqdm import tqdm
from PIL import Image
import concurrent.futures
from openai import OpenAI
import base64
from tenacity import retry, stop_after_attempt, wait_exponential

from decord import VideoReader


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(prompt, image_path, model_name="gpt-image-1", api_key=None, base_url=None):
    client = OpenAI(api_key=api_key, base_url=base_url)
    result = client.images.edit(
        model=model_name,
        image=open(image_path, "rb"),
        prompt=prompt,
        size="auto",
    )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    return image_bytes


def process_tag_extraction(
    tag, save_folder, main_image_save_path, idx, api_key, base_url
):
    subject_image_save_path = os.path.join(save_folder, f"{idx}_{tag}.png")
    prompt = f"Extract the {tag} as a separate image based on the elements in this picture, realistic-style, only one element."

    image_bytes = call_gpt(
        prompt=prompt,
        image_path=main_image_save_path,
        api_key=api_key,
        base_url=base_url,
    )

    with open(subject_image_save_path, "wb") as f:
        f.write(image_bytes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video_json",
        default="demo_result/step5/merge_final_json/dataset1.json",
    )
    parser.add_argument(
        "--input_video_root",
        type=str,
        default="demo_result/step0/videos/dataset1",
    )
    parser.add_argument(
        "--output_image_folder",
        type=str,
        default="demo_result/step6/gpt-frames_images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
    )
    parser.add_argument("--max_image_num", type=int, default=4)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key
    base_url = args.base_url
    max_image_num = args.max_image_num
    num_workers = args.num_workers

    input_video_root = args.input_video_root
    input_video_json = args.input_video_json
    output_image_folder = args.output_image_folder

    with open(input_video_json, "r") as f:
        all_data = json.load(f)
    data_items = list(all_data.items())

    for key, data in tqdm(data_items, desc="Processing Files", total=len(data_items)):
        save_folder = os.path.join(output_image_folder, key)
        os.makedirs(save_folder, exist_ok=True)

        if (
            len([f for f in os.listdir(save_folder) if f.endswith(".png")])
            > max_image_num + 1
        ):
            print(
                f"Skipping {key} as it already has {max_image_num + 1} PNG images, exceeding the max limit."
            )
            continue

        main_image_save_path = os.path.join(save_folder, "main_image.png")

        metadata = data["metadata"]
        tag_json_data = data["word_tags"]
        annotation_data = data["annotation"]

        crop = metadata["crop"]
        bbox_data = annotation_data["ann_frame_data"]["annotations"]
        frame_idx = annotation_data["ann_frame_data"]["ann_frame_idx"]
        video_path = os.path.join(input_video_root, metadata["path"])

        try:
            vr = VideoReader(video_path)
            frame = vr[int(frame_idx)]
            s_x, e_x, s_y, e_y = crop
            input_image = frame.asnumpy()[s_y:e_y, s_x:e_x]
            # save the original image
            pil_image = Image.fromarray(input_image)
            pil_image.save(main_image_save_path)
        except Exception as e:
            print(f"{e} load video {video_path} error")
            continue
        del vr
        gc.collect()

        valid_classes = set(
            tag_json_data["background"]
            + tag_json_data["subject"]
            + tag_json_data["object"]
        )

        ori_width = input_image.shape[1]
        ori_height = input_image.shape[0]
        original_area = ori_height * ori_width

        high_quality_tags = []
        for temp_bbox_data in bbox_data:
            # gme_score = temp_bbox_data['gme_score']
            # aes_score = temp_bbox_data['aes_score']
            x1, y1, x2, y2 = temp_bbox_data["bbox"]
            bbox_area = (x2 - x1) * (y2 - y1)
            area_ratio = bbox_area / original_area

            # if gme_score >= 0.7 and aes_score >= 3.9 and area_ratio > 0.08 and temp_bbox_data['class_name'] in valid_classes:
            if area_ratio > 0.08 and temp_bbox_data["class_name"] in valid_classes:
                high_quality_tags.append(temp_bbox_data["class_name"])

        high_quality_tags = list(
            set(
                re.sub(r"\s*-\s*", "-", tag.strip()).replace(" ", "_")
                for tag in high_quality_tags
            )
        )
        random.shuffle(high_quality_tags)

        max_tags = min(random.randint(1, max_image_num), len(high_quality_tags))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for idx, temp_tag in enumerate(
                tqdm(high_quality_tags, desc="Processing tag image")
            ):
                if idx == max_tags:
                    break
                futures.append(
                    executor.submit(
                        process_tag_extraction,
                        temp_tag,
                        save_folder,
                        main_image_save_path,
                        idx,
                        api_key,
                        base_url,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                pass


if __name__ == "__main__":
    main()
