import argparse
import concurrent.futures
import json
import os
from functools import partial

from tqdm import tqdm


def process_file(filename, folder_path, merged_data):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            key = filename.replace(".json", "")
            merged_data[key] = data
        except Exception as e:
            os.remove(file_path)
            print(f"File {filename} error: {e}")
            return None
    return True


def merge_json_files(folder_path, output_json_file, num_workers=8):
    merged_data = {}

    files = os.listdir(folder_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        process_func = partial(
            process_file, folder_path=folder_path, merged_data=merged_data
        )

        list(
            tqdm(
                executor.map(process_func, files), total=len(files), desc="merge jsons"
            )
        )

    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    with open(output_json_file, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merge complete, save to {output_json_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_folder",
        type=str,
        default="demo_result/step3/final_output/dataset1",
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        default="demo_result/step3/merge_final_json/dataset1.json",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    merge_json_files(args.input_json_folder, args.output_json_file, args.num_workers)


if __name__ == "__main__":
    main()
