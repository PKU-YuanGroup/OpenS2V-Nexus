import torch
from decord import VideoReader

import argparse
import json
import os

import cv2
import numpy as np
from diffusers.training_utils import free_memory
from insightface.app import FaceAnalysis
from tqdm import tqdm


def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    ret = cv2.copyMakeBorder(
        np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    return ret, (left, top)


def batch_cosine_similarity(embedding_image, embedding_frames, device="cuda"):
    embedding_image = torch.tensor(embedding_image).to(device)
    embedding_frames = torch.tensor(embedding_frames).to(device)
    return (
        torch.nn.functional.cosine_similarity(embedding_image, embedding_frames, dim=-1)
        .cpu()
        .numpy()
    )


def save_json(faces_info, output_path, id_list, frame_list, metadata):
    json_data = {"metadata": metadata, "bbox": {}}

    for i, (frame, face_info) in enumerate(faces_info.items()):
        face = {}
        face["face"] = []

        bboxs = face_info["bboxs"]
        kpss_x = face_info["kpss_x"]
        kpss_y = face_info["kpss_y"]
        det_scores = face_info["det_scores"]
        face_num = len(bboxs)

        for index in range(face_num):
            box = {}
            box["x1"] = float(bboxs[index][0])
            box["y1"] = float(bboxs[index][1])
            box["x2"] = float(bboxs[index][2])
            box["y2"] = float(bboxs[index][3])

            kps_info = {}
            kps_info["x"] = kpss_x[index]
            kps_info["y"] = kpss_y[index]

            id_value = {}
            id_value["track_id"] = id_list[frame][index]
            id_value["box"] = box
            id_value["keypoints"] = kps_info
            id_value["confidence"] = float(det_scores[index])
            id_value["class"] = 0
            id_value["name"] = "FACE"

            face["face"].append(id_value)

        json_data["bbox"][str(frame_list[i])] = face

    with open(output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    return json_data


class IDTracker:
    def __init__(self, faces_infos):
        self.faces_infos = faces_infos
        self.id_list = {}

        self.embeds_forward = {}
        self.embeds_backward = {}

        self.standard_index = -1
        self.max_num = 0
        self.frames = len(faces_infos)

        max_index = []
        det_score = {}
        max_num = 0
        for index, faces_info in self.faces_infos.items():
            if len(faces_info) == 0:
                continue

            if len(faces_info) == max_num:
                max_index.append(index)

                score_sum = 0.0

                for face in faces_info:
                    score_sum += float(face.det_score)

                average_score = score_sum / len(faces_info)
                det_score[index] = average_score

            elif len(faces_info) > max_num:
                max_num = len(faces_info)
                max_index = []
                max_index.append(index)

                det_score = {}
                score_sum = 0.0
                for face in faces_info:
                    score_sum += float(face.det_score)

                average_score = score_sum / len(faces_info)
                det_score[index] = average_score

        self.max_num = max_num

        max_average_score = 0.0
        standard_index = -1
        for index in max_index:
            if det_score[index] > max_average_score:
                max_average_score = det_score[index]
                standard_index = index

        if len(max_index) > 0:
            for face_id, face_info in enumerate(self.faces_infos[standard_index]):
                self.embeds_forward[face_id] = face_info.embedding
                self.embeds_backward[face_id] = face_info.embedding

        self.standard_index = standard_index

    def get_id(self, embedding, standard_embed):
        if self.standard_index == -1:
            return None

        max_score = -1
        face_id = -1
        for id_index, stand_embedding in standard_embed.items():
            score = batch_cosine_similarity(stand_embedding, embedding)
            if score > max_score:
                max_score = score
                face_id = id_index

        if max_score < 0.1:
            return None

        return face_id

    def track_id(self):
        if self.standard_index == -1:
            return None

        standard_frame_id = {}
        for i in range(self.max_num):
            standard_frame_id[i] = i
        self.id_list[self.standard_index] = standard_frame_id

        for index in range(self.standard_index, -1, -1):
            self.id_list[index] = {}
            current_embeds = {}

            for id_index, face in enumerate(self.faces_infos[index]):
                if face.det_score > 0.65:
                    face_id = self.get_id(face.embedding, self.embeds_forward)
                    if face_id is None:
                        face_id = self.max_num + 1
                        self.max_num += 1
                        self.id_list[index][id_index] = face_id
                    else:
                        self.id_list[index][id_index] = face_id

                    current_embeds[face_id] = face.embedding
                else:
                    self.id_list[index][id_index] = -1

            for face_id, face in current_embeds.items():
                self.embeds_forward[face_id] = face

        for face_id, face in self.embeds_forward.items():
            if face_id in self.embeds_backward:
                continue
            else:
                self.embeds_backward[face_id] = face

        for index in range(self.standard_index, self.frames, 1):
            current_embeds = {}
            self.id_list[index] = {}

            for id_index, face in enumerate(self.faces_infos[index]):
                if face.det_score > 0.65:
                    face_id = self.get_id(face.embedding, self.embeds_backward)
                    if face_id is None:
                        face_id = self.max_num + 1
                        self.max_num += 1
                        self.id_list[index][id_index] = face_id
                    else:
                        self.id_list[index][id_index] = face_id

                    current_embeds[face_id] = face.embedding
                else:
                    self.id_list[index][id_index] = -1

            for face_id, face in current_embeds.items():
                self.embeds_backward[face_id] = face

        return self.id_list


def prepare_face_model(model_path, device_id):
    face_model_1 = FaceAnalysis(
        name="antelopev2",
        root=os.path.join(model_path, "face_extractor"),
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": device_id}],
    )
    face_model_1.prepare(ctx_id=device_id, det_size=(640, 640))

    # face_model_2 = FaceAnalysis(
    #     name="buffalo_l",
    #     root=os.path.join(model_path, "face_extractor"),
    #     providers=["CUDAExecutionProvider"],
    #     provider_options=[{"device_id": device_id}],
    # )
    # face_model_2.prepare(ctx_id=device_id, det_size=(640, 640))

    # face_model_3 = FaceAnalysis(
    #     name="antelopev2",
    #     root=os.path.join(model_path, "face_extractor"),
    #     providers=["CUDAExecutionProvider"],
    #     provider_options=[{"device_id": device_id}],
    # )
    # face_model_3.prepare(ctx_id=device_id, det_size=(320, 320))

    # face_model_4 = FaceAnalysis(
    #     name="antelopev2",
    #     root=os.path.join(model_path, "face_extractor"),
    #     providers=["CUDAExecutionProvider"],
    #     provider_options=[{"device_id": device_id}],
    # )
    # face_model_4.prepare(ctx_id=device_id, det_size=(160, 160))

    # return face_model_1, face_model_2, face_model_3, face_model_4

    return face_model_1, None, None, None


def get_faces_info(face_model, frames):
    faces_infos = {}
    origin_infos = {}

    detect_flag = False

    for index, image in enumerate(frames):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces_info = face_model.get(image_bgr)

        if len(faces_info) == 0:
            # padding, try again
            _h, _w = image_bgr.shape[:2]
            _img, left_top_coord = pad_np_bgr_image(image_bgr)
            faces_info = face_model.get(_img)
            # if len(faces_info) == 0:
            #     print("Warning: No face detected in the video. Continue processing...")

            min_coord = np.array([0, 0])
            max_coord = np.array([_w, _h])
            sub_coord = np.array([left_top_coord[0], left_top_coord[1]])
            for face in faces_info:
                face.bbox = np.minimum(
                    np.maximum(face.bbox.reshape(-1, 2) - sub_coord, min_coord),
                    max_coord,
                ).reshape(4)
                face.kps = face.kps - sub_coord

        if len(faces_info) != 0:
            detect_flag = True

        origin_infos[index] = faces_info

        bboxs = []
        kpss_x = []
        kpss_y = []
        det_scores = []

        for face_info in faces_info:
            bboxs.append([int(x) for x in face_info.bbox.tolist()])
            kpss_x.append([int(x) for x in face_info.kps[:, 0].tolist()])
            kpss_y.append([int(x) for x in face_info.kps[:, 1].tolist()])
            det_scores.append(float(face_info.det_score))

        info_dict = {}

        info_dict["bboxs"] = bboxs
        info_dict["kpss_x"] = kpss_x
        info_dict["kpss_y"] = kpss_y
        info_dict["det_scores"] = det_scores

        faces_infos[index] = info_dict

    if detect_flag:
        return faces_infos, origin_infos
    else:
        return None, None


def get_face_info_all(
    face_model_1, face_model_2=None, face_model_3=None, face_model_4=None, frames=None
):
    faces_infos, origin_infos = get_faces_info(face_model_1, frames)

    if faces_infos is None and face_model_2 is not None:
        faces_infos, origin_infos = get_faces_info(face_model_2, frames)

    if faces_infos is None and face_model_3 is not None:
        faces_infos, origin_infos = get_faces_info(face_model_3, frames)

    if faces_infos is None and face_model_4 is not None:
        faces_infos, origin_infos = get_faces_info(face_model_4, frames)

    return faces_infos, origin_infos


def process_video(
    video_metadatas, model_path, output_json_folder, input_video_root, device_id
):
    face_model_1, face_model_2, face_model_3, face_model_4 = prepare_face_model(
        model_path, device_id
    )

    free_memory()
    for key, value in tqdm(
        video_metadatas.items(), desc="Processing videos", unit="video"
    ):
        metadata = value["metadata"]
        cut = metadata["cut"]
        s_x, e_x, s_y, e_y = metadata["crop"]
        raw_path = metadata["path"]

        video_path = os.path.join(input_video_root, raw_path)
        output_path = os.path.join(
            output_json_folder, key + f"_step1-{cut[0]}-{cut[1]}.json"
        )

        os.makedirs(output_json_folder, exist_ok=True)

        if os.path.exists(output_path):
            print(f"Skipping existing file: {key}")
            continue

        try:
            vr = VideoReader(video_path)
            frame_list = np.arange(cut[0], cut[1])
            frames = vr.get_batch(frame_list).asnumpy()
            frames = frames[:, s_y:e_y, s_x:e_x, :]
            del vr
            free_memory()
        except Exception as e:
            print(e)
            continue

        if frames is None:
            continue

        faces_infos, origin_infos = get_face_info_all(
            face_model_1, face_model_2, face_model_3, face_model_4, frames
        )

        if faces_infos is None:
            flag = "None"
            with open(output_path, "w") as json_file:
                json.dump(flag, json_file, indent=4)
            print(f"Json file saved to {output_path}")
            continue

        Tracker = IDTracker(origin_infos)
        id_list = Tracker.track_id()

        save_json(faces_infos, output_path, id_list, frame_list, metadata)
        free_memory()


def split_list(data, nums, part):
    items = list(data.items())
    size = len(items)
    part_size = size // nums
    remainder = size % nums

    start = part * part_size + min(part, remainder)
    end = start + part_size + (1 if part < remainder else 0)

    return dict(items[start:end])


def parse_args():
    parser = argparse.ArgumentParser(description="Process MP4 files with YOLO models.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="OpenS2V-Weight",
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--input_video_json",
        type=str,
        default="demo_result/step0/merge_final_json/dataset1.json",
    )
    parser.add_argument(
        "--input_video_root",
        type=str,
        default="demo_result/step0/videos/dataset1",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        default="demo_result/step1/final_output/dataset1",
    )
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total_part", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    json_path = args.input_video_json
    input_video_root = args.input_video_root
    output_json_folder = args.output_json_folder

    with open(json_path, "r") as f:
        data = json.load(f)

    split_data = split_list(data, args.total_part, args.part)

    process_video(
        video_metadatas=split_data,
        model_path=model_path,
        output_json_folder=output_json_folder,
        input_video_root=input_video_root,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
