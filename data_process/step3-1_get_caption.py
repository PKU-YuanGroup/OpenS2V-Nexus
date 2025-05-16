import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from torch.distributed import destroy_process_group

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import qwen_vl_utils
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "false"

input_prompt = (
    "Please generate a comprehensive caption for the following video, describing various aspects, including but not limited to: "
    "1. The main theme and setting of the image (such as location, time of day, weather conditions, etc.) "
    "2. Key objects and their characteristics (such as color, shape, size, etc.) "
    "3. Relationships and interactions between objects (such as positioning, actions, etc.) "
    "4. Any people present and their emotions or activities (such as expressions, postures, etc.) "
    "5. Background and environmental details (such as architecture, natural scenery, etc.) "
    "6. Motion of the Subject: The movement of people or objects in the video. Use verbs that describe movement. "
    "7. Camera motion control: zoom in, zoom out, push in, pull out, pan right, pan left, truck right, truck left, tilt up, tilt down, pedestal up, pedestal down, arc shot,  tracking shot, static shot, and handheld shot. "
    'Do not describe imagined content. Only describe what can be determined from the video. Avoid listing things. Do not use abstract concepts (love, hate, justice, infinity, joy) as subjects. Use concrete nouns (human, cup, dog, planet, headphones) for more accurate results. Use verbs to describe the movement and changes of the subject or people. Write your prompts in plain, conversational language. Start your description directly with the main subject, typically a noun. Without "\n", subheading and title. '
    "Please describe the content of the video and the changes that occur, in chronological order:"
)


def _read_video_torchvision_cus(
    ele: dict,
) -> Tuple[torch.Tensor, float]:
    video_path = ele["video"]
    video, audio, info = io.read_video(
        video_path,
        pts_unit="sec",
        output_format="TCHW",
    )
    # crop video
    s_x, e_x, s_y, e_y = ele["crop"]
    video = video[ele["video_start"] : ele["video_end"], :, s_y:e_y, s_x:e_x]
    # sample video
    total_frames = video.size(0)
    video_fps = info["video_fps"]
    nframes = 16
    # nframes = qwen_vl_utils.vision_process.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def _read_video_decord_cus(
    ele: dict,
) -> Tuple[torch.Tensor, float]:
    import decord

    vr = decord.VideoReader(ele["video"], num_threads=1)
    # crop video
    s_x, e_x, s_y, e_y = ele["crop"]
    # sample video
    total_frames = ele["video_end"] - ele["video_start"]
    _, video_fps = len(vr), vr.get_avg_fps()
    nframes = 16
    # nframes = qwen_vl_utils.vision_process.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    idx = [i + ele["video_start"] for i in idx]
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    video = video[:, :, s_y:e_y, s_x:e_x]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    del vr
    return video, sample_fps


qwen_vl_utils.vision_process.VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord_cus,
    "torchvision": _read_video_torchvision_cus,
}


class CaptionData(Dataset):
    def __init__(self, video_data, input_video_root, output_json_folder, processor):
        super().__init__()
        self.input_video_root = input_video_root
        self.output_json_folder = output_json_folder
        vid_paths = [i["path"] for i in video_data]
        crops = [i["crop"] for i in video_data]
        cuts = [i["cut"] for i in video_data]
        video_keys = [i["video_key"] for i in video_data]
        save_paths = [
            os.path.join(output_json_folder, (i["video_key"] + "_step4.json"))
            for i in video_data
        ]
        print("part x origin num", len(save_paths))
        self.paths = [
            [save_path, vid_path, crop, cut, video_key]
            for save_path, vid_path, crop, cut, video_key in zip(
                save_paths, vid_paths, crops, cuts, video_keys
            )
        ]
        print("part x need to process num", len(self.paths))

        self.processor = processor
        self.executor = ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return len(self.paths)

    def load_video(self, path, crop, cut):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": path,
                        # "total_pixels": 20480 * 28 * 28,
                        "min_pixels": 16 * 28 * 28,
                        # "max_pixels": 512 * 512,
                        "fps": 1.0,
                        "video_start": cut[0],
                        "video_end": cut[1],
                        "crop": crop,
                    },
                    {"type": "text", "text": input_prompt},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
        }

        return inputs

    def wrapper(self, index):
        save_path, video_path, crop, cut, video_key = self.paths[index]
        inputs = [self.load_video(video_path, crop, cut)]
        return save_path, inputs, video_key

    def __getitem__(self, index):
        try:
            future = self.executor.submit(self.wrapper, index)
            save_path, inputs, video_key = future.result(timeout=50)
            return save_path, inputs, video_key
        except Exception as e:
            print("error", e)
            return False, False, False


def collate_fn(batch):
    save_paths, inputs, video_key = zip(*batch)
    inputs = inputs[0]
    if not inputs:
        return False, False, False
    return save_paths, inputs, video_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct/",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--input_video_json",
        type=str,
        default="demo_result/step2/merge_final_json/dataset1.json",
    )
    parser.add_argument(
        "--input_video_root", type=str, default="demo_result/step0/videos/dataset1"
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        default="demo_result/step3/final_output/dataset1",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total_part", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.batch_size == 1

    model_id_or_path = args.model_id_or_path
    sampling_params = SamplingParams(
        repetition_penalty=1.05,
        temperature=0.1,
        top_p=0.001,
        # top_k=1,
        max_tokens=512,
    )

    llm = LLM(
        model_id_or_path,
        # max_model_len=32768 if process_vision_info is None else 4096,
        # tensor_parallel_size=4,
        # distributed_executor_backend="mp",
        # gpu_memory_utilization=0.9
    )
    processor = AutoProcessor.from_pretrained(model_id_or_path)

    with open(args.input_video_json, "r") as f:
        origin_video_data = json.load(f)
    print("total data", len(origin_video_data))

    video_data = [
        {
            "path": os.path.join(args.input_video_root, i["metadata"]["path"]),
            "video_key": key,
            "cut": i["metadata"]["face_cut"],
            "crop": i["metadata"]["crop"],
            "aes": i["metadata"]["aesthetic"],
            "tech": i["metadata"]["tech"],
            "motion": i["metadata"]["motion"],
            "num_frames": i["metadata"]["num_frames"],
            "resolution": [
                i["metadata"]["resolution"]["height"],
                i["metadata"]["resolution"]["width"],
            ],
        }
        for key, i in tqdm(origin_video_data.items())
    ]

    print("after filter data", len(video_data))
    video_data = video_data[args.part :: args.total_part]
    data = CaptionData(
        video_data, args.input_video_root, args.output_json_folder, processor
    )
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    for save_paths, frames, video_key in tqdm(loader):
        if not save_paths:
            print(f"{save_paths} is broking")
            continue
        if os.path.exists(save_paths[0]):
            print(f"{save_paths} is already exists")
            continue
        if len(save_paths[0]) > 255:
            print("Name too long, skipping :", save_paths[0])
            continue

        folder, filename = os.path.split(save_paths[0])
        os.makedirs(folder, exist_ok=True)

        try:
            results = []
            for inputs in frames:
                with torch.inference_mode():
                    outputs = llm.generate([inputs], sampling_params=sampling_params)
                    generated_text = outputs[0].outputs[0].text
                    results.append(generated_text)

            base_dict = origin_video_data[video_key[0]]
            base_dict["metadata"]["face_cap_qwen"] = results[0]

            with open(save_paths[0], "w") as f:
                json.dump(base_dict, f, indent=4)
        except Exception as e:
            print(e)

    destroy_process_group()
    print("Done")


if __name__ == "__main__":
    main()
