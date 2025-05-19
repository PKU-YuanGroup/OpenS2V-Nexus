import argparse
import json
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_folder",
        type=str,
        default="demo_result/step5/merge_final_json",
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        default="demo_result/step6/cluster_videos.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_json_folder = args.input_json_folder
    output_json_file = args.output_json_file

    output_dir = os.path.dirname(output_json_file)
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_json_folder) if f.endswith(".json")]

    clustered_data = {}
    for json_file in tqdm(json_files, desc="Processing JSON files", unit="file"):
        input_file_path = os.path.join(input_json_folder, json_file)
        with open(input_file_path, "r") as f:
            data = json.load(f)

        for video_name in data:
            cluster_id = video_name.split("_segment")[0]
            if cluster_id == video_name:
                cluster_id = video_name.split("_part")[0]
            if cluster_id not in clustered_data:
                clustered_data[cluster_id] = set()
            clustered_data[cluster_id].add(video_name)

    clustered_data = {k: list(v) for k, v in clustered_data.items() if len(v) > 1}
    cluster_sizes = [len(videos) for videos in clustered_data.values()]
    average_cluster_size = (
        sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
    )

    with open(output_json_file, "w") as f:
        json.dump(clustered_data, f, indent=4)

    print(f"Clustered data saved to {output_json_file}")
    print(f"Average number of videos per cluster: {average_cluster_size:.2f}")
    print(f"Total number of clusters: {len(clustered_data)}")


if __name__ == "__main__":
    main()
