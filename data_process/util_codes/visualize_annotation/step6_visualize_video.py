import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from pycocotools import mask as mask_util
from supervision.draw.color import ColorPalette
from tqdm import tqdm


CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]


def draw_masks(scene, detections, opacity=0.5, color=(30, 144, 255)):
    if detections.mask is None or len(detections.mask) == 0:
        return scene

    overlay = scene.copy()
    output = scene.copy()

    sorted_indices = np.flip(np.argsort(detections.area))

    colors = [color for _ in range(len(detections))]

    for idx in sorted_indices:
        mask = detections.mask[idx]
        overlay[mask] = colors[idx]

    cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, output)
    return output


def fast_draw_masks(scene, detections, opacity=0.5):
    if detections.mask is None:
        return scene

    combined_mask = np.any(detections.mask, axis=0)
    overlay = scene.copy()
    overlay[combined_mask] = (30, 144, 255)

    output = scene.copy()
    cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, output)
    return output


input_json = "../../demo_result/step5/final_output/dataset1/gm1252760410-365677148_part2_step1-0-95_step2-0-95_step4_step5_step6.json"
input_video_path = (
    "../../demo_result/step0/videos/dataset1/gm1252760410-365677148_part2.mp4"
)
output_frame_path = "./step6/frames"
output_anno_frame_path = "./step6/anno_frames"
target_object_names = ["2", "4"]
# target_object_names = None

if not os.path.exists(output_anno_frame_path):
    os.makedirs(output_anno_frame_path)

if not os.path.exists(output_frame_path):
    os.makedirs(output_frame_path)

with open(input_json, "r") as f:
    json_data = json.load(f)

mask_annotation = json_data["annotation"]["mask_annotation"]
cut = json_data["metadata"]["face_cut"]
crop = json_data["metadata"]["crop"]
# cut = None
# crop = None

video_info = sv.VideoInfo.from_video_path(input_video_path)
print(video_info)
frame_generator = sv.get_video_frames_generator(
    input_video_path, stride=1, start=cut[0] if cut else 0, end=cut[1] if cut else None
)

# saving video to frames
source_frames = Path(output_frame_path)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg"
) as sink:
    for frame_idx, frame in enumerate(
        tqdm(frame_generator, desc="Saving Video Frames")
    ):
        # Apply crop if specified
        if crop:
            s_x, e_x, s_y, e_y = crop
            frame = frame[s_y:e_y, s_x:e_x]
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(output_frame_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))

if cut is None:
    cut = [0]

for frame_idx in tqdm(mask_annotation):
    segments = mask_annotation[str(frame_idx)]
    frame_idx = int(frame_idx) - cut[0]
    img = cv2.imread(os.path.join(output_frame_path, frame_names[int(frame_idx)]))

    object_names = list(segments.keys())
    mask_rles = list(segments.values())

    if target_object_names is not None:
        filtered = [
            (name, rle)
            for name, rle in zip(object_names, mask_rles)
            if name in target_object_names
        ]
        if not filtered:
            continue
        object_names, mask_rles = zip(*filtered)

    masks = [
        mask_util.decode(
            {
                "counts": rle["counts"].encode("utf-8"),
                "size": [img.shape[0], img.shape[1]],
            }
        )
        for rle in mask_rles
    ]
    masks = np.array(masks)

    class_name_to_id = {name: idx for idx, name in enumerate(set(object_names))}
    object_ids_int = [class_name_to_id[name] for name in object_names]

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks,  # (n, h, w)
        class_id=np.array(object_ids_int, dtype=np.int32),
    )

    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=object_names
    )
    # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    start_time = time.time()
    annotated_frame = fast_draw_masks(annotated_frame, detections)
    elapsed_time = time.time() - start_time
    print(f"draw_masks 执行时间: {elapsed_time:.4f} 秒")

    cv2.imwrite(
        os.path.join(
            output_anno_frame_path, f"annotated_frame_{int(frame_idx):05d}.jpg"
        ),
        annotated_frame,
    )
