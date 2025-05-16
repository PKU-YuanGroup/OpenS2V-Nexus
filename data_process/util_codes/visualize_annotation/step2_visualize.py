import json
import os

import cv2


def draw_bbox_and_track_id(video_path, json_data, output_path, cut, crop):
    cap = cv2.VideoCapture(video_path)

    s_x, e_x, s_y, e_y = crop

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = e_x - s_x
    frame_height = e_y - s_y

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    start_frame, end_frame = cut

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
            frame = frame[s_y:e_y, s_x:e_x]
            frame_data = json_data.get(str(frame_count))

            if frame_data:
                for face in frame_data["face"]:
                    track_id = face["track_id"]
                    box = face["box"]
                    x1, y1, x2, y2 = (
                        int(box["x1"]),
                        int(box["y1"]),
                        int(box["x2"]),
                        int(box["y2"]),
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2,
                    )

            out.write(frame)

        if frame_count > end_frame:
            break

        frame_count += 1

    cap.release()
    out.release()
    print(
        f"Processing completed, only frames {start_frame} ~ {end_frame} were processed, and the results have been saved to: {output_path}"
    )


if __name__ == "__main__":
    input_jsons_dir = "../../demo_result/step2/final_output/dataset1"
    video_root = "../../demo_result/step0/videos/dataset1"
    output_dir = "./step2/visual_video"

    os.makedirs(output_dir, exist_ok=True)

    jsons_files = [f for f in os.listdir(input_jsons_dir) if f.endswith(".json")]

    for json_file in jsons_files:
        with open(os.path.join(input_jsons_dir, json_file), "r") as f:
            json_data = json.load(f)

        metadata = json_data["metadata"]
        bbox = json_data["bbox"]

        video_path = video_root + "/" + metadata["path"]
        output_path = os.path.join(output_dir, json_file.replace(".json", ".mp4"))

        draw_bbox_and_track_id(
            video_path, bbox, output_path, metadata["face_cut"], metadata["crop"]
        )
