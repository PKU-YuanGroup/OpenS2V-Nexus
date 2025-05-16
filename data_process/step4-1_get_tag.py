import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


file_lock = Lock()

prompt_format = (
    "Given an image caption, please retrieve the entity words that indicate background, subject, and visually separable objects. "
    "[Definition of background] The background spaces that appear in most of the image area. "
    "[Definition of subject] Human or animal subjects that appear in the image. "
    "[Definition of object] Entities that are visually separable, tangible, and physically present in part of the image. "
    "Attention! All entity words need to strictly follow the rules below: "
    "1) The entity word is a singular or plural noun without any quantifier or descriptive phrase. "
    "2) The entity word must be an exact subset of the caption, including its characters, words, and symbols. (e.g, 'red top' better than 'top', 'martial arts uniforms' better than 'uniforms') "
    "3) Exclude any part of the body (e.g., 'hands', 'legs', 'feet', 'head'). "
    "4) Exclude abstract or non-physical concepts (e.g., 'facial expressions', 'gestures', 'stance'). "
    "5) Exclude actions or descriptions (e.g., 'adjusting', 'imitating'). "
    "6) Do not modify or interpret any part of the caption. "
    "Here is an example, follow this JSON format to output the results: "
    "Caption: A woman in a mask and coat, with long brown hair, shows a small green-capped bottle to the camera. "
    "Output: {'background': [''], 'subject': ['woman'], 'object': ['mask', 'coat', 'long brown hair', 'green-capped bottle']} "
    "Here is the input: "
    "Caption: {{{}}} "
    "Output: "
)

keywords_to_remove = [
    "person",
    "people",
    "human",
    "individual",
    "individuals",
    "woman",
    "man",
    "child",
    "children",
    "adult",
    "teenager",
    "infant",
    "youth",
    "elder",
    "boy",
    "girl",
    "male",
    "female",
]


def filter_keywords(items, keywords):
    return [item for item in items if item.lower() not in keywords]


def extract_data_from_response(response):
    response = response.replace('"', "'")

    background_pattern = r"'background'\s*:\s*\[(.*?)\]"
    subject_pattern = r"'subject'\s*:\s*\[(.*?)\]"
    object_pattern = r"'object'\s*:\s*\[(.*?)\]"

    background_match = re.search(background_pattern, response)
    subject_match = re.search(subject_pattern, response)
    object_match = re.search(object_pattern, response)

    result = {
        "pre_define": ["person", "human head"],
        "background": [],
        "subject": [],
        "object": [],
    }

    if background_match:
        background_items = [
            item.strip("' ") for item in background_match.group(1).split("', '")
        ]
        # result["background"] = filter_keywords(background_items, keywords_to_remove)
        result["background"] = background_items

    if subject_match:
        subject_items = [
            item.strip("' ") for item in subject_match.group(1).split("', '")
        ]
        # result["subject"] = filter_keywords(subject_items, keywords_to_remove)
        result["subject"] = subject_items

    if object_match:
        object_items = [
            item.strip("' ") for item in object_match.group(1).split("', '")
        ]
        # result["object"] = filter_keywords(object_items, keywords_to_remove)
        result["object"] = object_items

    if background_match and subject_match and object_match:
        flag = True
    else:
        flag = False

    return result, flag


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(prompt, model_name="deepseek-chat", api_key=None, base_url=None):
    client = OpenAI(api_key=api_key, base_url=base_url)
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content


def process_file(key, value, model_name, api_key, base_url, output_json_folder):
    temp_output_path = os.path.join(output_json_folder, key + "_step5.json")

    if os.path.exists(temp_output_path):
        print(f"{key} is already exists")
        return

    metadata = value["metadata"]

    if "face_cap_glm" in metadata.keys() and len(metadata["face_cap_glm"]) > 0:
        ori_prompt = metadata["face_cap_glm"]
    elif "face_cap_Aria" in metadata.keys() and len(metadata["face_cap_Aria"]) > 0:
        ori_prompt = metadata["face_cap_Aria"]
    elif "face_cap_qwen" in metadata.keys() and len(metadata["face_cap_qwen"]) > 0:
        ori_prompt = metadata["face_cap_qwen"]
    else:
        return

    input_prompt = (
        prompt_format.replace("{{{}}}", ori_prompt.strip())
        .replace("\n", "")
        .replace("\t", "")
        .strip()
    )

    response = call_gpt(
        input_prompt, model_name=model_name, api_key=api_key, base_url=base_url
    )

    try:
        response_data = json.loads(response)
    except json.JSONDecodeError:
        response_data, flag = extract_data_from_response(response)
        if not flag:
            print("The response fis not in JSON format, skipping.")
            return

    value["word_tags"] = response_data

    with open(temp_output_path, "w") as f:
        json.dump(value, f, indent=4)


def split_list(data, nums, part):
    items = list(data.items())
    size = len(items)
    part_size = size // nums
    remainder = size % nums

    start = part * part_size + min(part, remainder)
    end = start + part_size + (1 if part < remainder else 0)

    return dict(items[start:end])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process some input and output folders with model settings."
    )
    parser.add_argument(
        "--input_video_json",
        type=str,
        default="demo_result/step3/merge_final_json/dataset1.json",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        default="demo_result/step4/final_output/dataset1",
    )
    parser.add_argument(
        "--api_key", type=str, default="sk-3845a3e5f3d84ecd847c20b9eb543fe8"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com/v1")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total_part", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Config
    num_workers = args.num_workers
    input_video_json = args.input_video_json
    output_json_folder = args.output_json_folder

    model_name = args.model_name
    api_key = args.api_key
    base_url = args.base_url

    os.makedirs(output_json_folder, exist_ok=True)

    with open(input_video_json, "r") as f:
        data = json.load(f)

    split_data = split_list(data, args.total_part, args.part)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for key, value in split_data.items():
            futures.append(
                executor.submit(
                    process_file,
                    key,
                    value,
                    model_name,
                    api_key,
                    base_url,
                    output_json_folder,
                )
            )

        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing videos",
            unit="video",
        ):
            pass


if __name__ == "__main__":
    main()
