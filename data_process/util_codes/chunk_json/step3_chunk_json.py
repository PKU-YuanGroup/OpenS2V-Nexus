import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def should_include_item(item, resume_dir):
    key, value = item
    file_name = key + "_step4.json"
    file_path = os.path.join(resume_dir, file_name)
    if not os.path.exists(file_path):
        return key, value
    return None


def write_chunk_to_file(chunk, output_file):
    with open(output_file, "w") as f:
        json.dump(chunk, f, indent=2)
    return output_file


def split_json_file(
    input_video_json, output_chunk_json_folder, total_part, resume_dir, num_workers
):
    os.makedirs(output_chunk_json_folder, exist_ok=True)

    with open(input_video_json, "r") as f:
        json_data = json.load(f)

    total_items = len(json_data)
    print(f"{input_video_json} Total items before: {total_items}")

    if not isinstance(json_data, (list, dict)):
        raise ValueError("JSON data must be an array or object")

    data = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(
            lambda item: should_include_item(item, resume_dir), json_data.items()
        )
        for result in futures:
            if result:
                key, value = result
                if "face_cap_glm" in value["metadata"].keys():
                    if not value["metadata"]["face_cap_glm"]:
                        del value["metadata"]["face_cap_glm"]
                if "face_cap_qwen" in value["metadata"].keys():
                    del value["metadata"]["face_cap_qwen"]
                if "face_cap_Aria" in value["metadata"].keys():
                    del value["metadata"]["face_cap_Aria"]
                if "face_cap_aria" in value["metadata"].keys():
                    del value["metadata"]["face_cap_aria"]
                data[key] = value

    if isinstance(data, dict):
        items = list(data.items())
    else:
        items = data

    total_items = len(items)
    print(f"{input_video_json} Total items after: {total_items}")

    base_chunk_size = total_items // total_part
    remainder = total_items % total_part

    start = 0
    tasks = []
    base_name = os.path.splitext(os.path.basename(input_video_json))[0]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(total_part):
            if start < total_items:
                extra = 1 if i < remainder else 0
                end = start + base_chunk_size + extra
                chunk = items[start:end]
                start = end
            else:
                chunk = []

            if isinstance(data, dict):
                chunk = dict(chunk)

            output_file = os.path.join(
                output_chunk_json_folder, f"{base_name}_part{i + 1}.json"
            )
            tasks.append(executor.submit(write_chunk_to_file, chunk, output_file))

        for future in as_completed(tasks):
            future.result()


def main():
    parser = argparse.ArgumentParser(
        description="Split a large JSON file into smaller chunks."
    )
    parser.add_argument(
        "--input_video_json",
        type=str,
        default="0_demo_output/step2/merge_input_json/dataset1.json",
    )
    parser.add_argument(
        "--output_chunk_json_folder",
        type=str,
        default="0_demo_output/step4/chunk_input_json/dataset1",
    )
    parser.add_argument(
        "--resume_dir", type=str, default="0_demo_output/step4/final_output/dataset1"
    )
    parser.add_argument("--total_part", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.output_chunk_json_folder, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)

    split_json_file(
        args.input_video_json,
        args.output_chunk_json_folder,
        args.total_part,
        args.resume_dir,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
