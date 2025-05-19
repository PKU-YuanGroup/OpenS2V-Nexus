import argparse
import json
import os

from decord import VideoReader, cpu
from PIL import Image, ImageOps
from tqdm import tqdm
import concurrent.futures


def process_json_file(key, value, input_video_root, output_image_folder):
    local_output_path = os.path.join(output_image_folder, key)
    if not os.path.exists(local_output_path):
        os.makedirs(local_output_path, exist_ok=True)

    try:
        metadata = value["metadata"]
        cut = metadata["face_cut"]
        s_x, e_x, s_y, e_y = metadata["crop"]
        video_path = os.path.join(input_video_root, metadata["path"])
        frame_indices = [cut[0]]

        for frame_indice in frame_indices:
            temp_path = os.path.join(local_output_path, f"{frame_indice}.png")
            if os.path.exists(temp_path):
                print(f"already existing: {temp_path}")
                return

        # decord
        vr = VideoReader(video_path, ctx=cpu(0))
        batch_frames = vr.get_batch(frame_indices).asnumpy()
        cropped_frames = [frame[s_y:e_y, s_x:e_x, :] for frame in batch_frames]
        extracted_frames = [
            ImageOps.exif_transpose(Image.fromarray(frame)) for frame in cropped_frames
        ]

        for image, frame_indice in zip(extracted_frames, frame_indices):
            temp_path = os.path.join(local_output_path, f"{frame_indice}.png")
            image.save(temp_path)
            print(f"save to {temp_path}")

    except Exception as e:
        print(f"wrong:{e}")
        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video_json",
        default="demo_result/step4/merge_final_json/dataset1.json",
    )
    parser.add_argument(
        "--input_video_root",
        type=str,
        default="demo_result/step0/videos/dataset1",
    )
    parser.add_argument(
        "--output_image_folder",
        type=str,
        default="demo_result/step5/temp_offload_images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=96,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_video_json = args.input_video_json
    input_video_root = args.input_video_root
    output_image_folder = args.output_image_folder

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder, exist_ok=True)

    with open(input_video_json, "r") as f:
        data = json.load(f)

    def process_with_thread(key, value, input_video_root, output_image_folder):
        process_json_file(key, value, input_video_root, output_image_folder)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        with tqdm(total=len(data), desc="Processing videos", unit="video") as pbar:

            def update_progress(result):
                pbar.update(1)

            futures = []
            for key, value in data.items():
                future = executor.submit(
                    process_with_thread,
                    key,
                    value,
                    input_video_root,
                    output_image_folder,
                )
                future.add_done_callback(update_progress)
                futures.append(future)

            concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
