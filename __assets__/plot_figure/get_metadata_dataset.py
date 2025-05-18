import json
import os
from tqdm import tqdm  # Import tqdm

json_folder_path = "merge_final_json"
output_path = "dataset"

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Initialize counters for the intervals
aesthetic_intervals = {"<5": 0, "5~5.5": 0, "5.5~6.0": 0, ">6": 0}
motion_intervals = {"<0.03": 0, "0.03~0.06": 0, ">0.06": 0}
resolution_intervals = {"480P": 0, "720P": 0, "1080P": 0}
tech_intervals = {"<0.5": 0, "0.5~1.0": 0, "1.0~1.5": 0, "1.5~2.0": 0, ">2.0": 0}
video_length_intervals = {"<5": 0, "5~10": 0, "10~15": 0, ">15": 0}
word_length_intervals = {"<100": 0, "100~150": 0, "150~200": 0, "200~250": 0, ">250": 0}
total_items = 0
total_video_length_seconds = 0

# Lists to store all values across all files
tech_values = []
motion_values = []
aesthetic_values = []
video_lengths = []
resolutions = []
word_lengths = []
face_cap = []

# Process each JSON file in the folder
json_files = [f for f in os.listdir(json_folder_path) if f.endswith(".json")]

for json_file in tqdm(json_files, desc="Processing JSON files", unit="json"):
    json_file_path = os.path.join(json_folder_path, json_file)

    # Open and load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Process the data for each file
    for key, value in tqdm(data.items(), desc="Processing metadata", unit="item"):
        total_items += 1
        metadata = value["metadata"]

        # Tech, motion, aesthetic, fps, num_frames, and face_cut info
        tech_values.append(metadata.get("tech", 0))
        motion_values.append(metadata.get("motion", 0))
        aesthetic_values.append(metadata.get("aesthetic", 0))

        # Video length: fps * (face_cut[1] - face_cut[0])
        fps = metadata.get("fps", 0)
        face_cut = metadata.get("face_cut", [0, 0])
        video_length = (face_cut[1] - face_cut[0]) / fps
        total_video_length_seconds += video_length
        video_lengths.append(video_length)

        # Resolution: width and height -> classify into 480P, 720P, 1080P
        width = metadata["resolution"]["width"]
        height = metadata["resolution"]["height"]
        if width < height:
            width, height = height, width
        resolution = (width, height)
        if width < 1280 or height < 720:
            resolutions.append("480P")
        elif width < 1920 or height < 1080:
            resolutions.append("720P")
        else:
            resolutions.append("1080P")

        # Word range for face_cap_qwen
        face_cap_qwen = metadata.get("face_cap_qwen", "")
        word_count = len(face_cap_qwen.split())
        word_lengths.append(word_count)
        face_cap.append(face_cap_qwen)

        # Process aesthetic
        aesthetic = metadata.get("aesthetic", 0)
        if aesthetic < 5:
            aesthetic_intervals["<5"] += 1
        elif 5 <= aesthetic < 5.5:
            aesthetic_intervals["5~5.5"] += 1
        elif 5.5 <= aesthetic < 6.0:
            aesthetic_intervals["5.5~6.0"] += 1
        else:
            aesthetic_intervals[">6"] += 1

        # Process motion
        motion = metadata.get("motion", 0)
        if motion < 0.03:
            motion_intervals["<0.03"] += 1
        elif 0.03 <= motion < 0.06:
            motion_intervals["0.03~0.06"] += 1
        else:
            motion_intervals[">0.06"] += 1

        # Process tech
        tech = metadata.get("tech", 0)
        if tech < 0.5:
            tech_intervals["<0.5"] += 1
        elif 0.5 <= tech < 1.0:
            tech_intervals["0.5~1.0"] += 1
        elif 1.0 <= tech < 1.5:
            tech_intervals["1.0~1.5"] += 1
        elif 1.5 <= tech < 2.0:
            tech_intervals["1.5~2.0"] += 1
        else:
            tech_intervals[">2.0"] += 1

        if video_length < 5:
            video_length_intervals["<5"] += 1
        elif 5 <= video_length < 10:
            video_length_intervals["5~10"] += 1
        elif 10 <= video_length < 15:
            video_length_intervals["10~15"] += 1
        else:
            video_length_intervals[">15"] += 1

        if width < 1280 or height < 720:
            resolution_intervals["480P"] += 1
        elif width < 1920 or height < 1080:
            resolution_intervals["720P"] += 1
        else:
            resolution_intervals["1080P"] += 1

        if word_count < 100:
            word_length_intervals["<100"] += 1
        elif 100 <= word_count < 150:
            word_length_intervals["100~150"] += 1
        elif 150 <= word_count < 200:
            word_length_intervals["150~200"] += 1
        elif 200 <= word_count < 250:
            word_length_intervals["200~250"] += 1
        else:
            word_length_intervals[">250"] += 1

average_length_seconds = (
    total_video_length_seconds / total_items if total_items > 0 else 0
)
total_video_duration_hours = (
    total_video_length_seconds / 3600
)  # Convert seconds to hours

output_file = os.path.join(output_path, "interval_counts.txt")
with open(output_file, "w") as f:
    f.write(f"Total number of items: {total_items}\n\n")
    f.write(f"Average Video Length (s): {average_length_seconds:.2f}\n")
    f.write(f"Total Video Duration (h): {total_video_duration_hours:.2f}\n\n")

    f.write("Aesthetic Intervals:\n")
    for interval, count in aesthetic_intervals.items():
        f.write(f"{interval}: {count}\n")

    f.write("\nMotion Intervals:\n")
    for interval, count in motion_intervals.items():
        f.write(f"{interval}: {count}\n")

    f.write("\nResolution Intervals:\n")
    for interval, count in resolution_intervals.items():
        f.write(f"{interval}: {count}\n")

    f.write("\nTech Intervals:\n")
    for interval, count in tech_intervals.items():
        f.write(f"{interval}: {count}\n")

    f.write("\nVideo Length Intervals:\n")
    for interval, count in video_length_intervals.items():
        f.write(f"{interval}: {count}\n")

    f.write("\nWord Length Intervals:\n")
    for interval, count in word_length_intervals.items():
        f.write(f"{interval}: {count}\n")

print("Interval counts saved to 'interval_counts.txt'")

# # Plot histograms and save the plots
# # Tech Distribution
# plt.figure(figsize=(10, 6))
# plt.hist(tech_values, bins=20, color='skyblue', edgecolor='black')
# plt.title('Tech Distribution')
# plt.xlabel('Tech Value')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(output_path, 'tech_distribution.png'))

# # Motion Distribution
# plt.figure(figsize=(10, 6))
# plt.hist(motion_values, bins=20, color='salmon', edgecolor='black')
# plt.title('Motion Distribution')
# plt.xlabel('Motion Value')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(output_path, 'motion_distribution.png'))

# # Aesthetic Distribution
# plt.figure(figsize=(10, 6))
# plt.hist(aesthetic_values, bins=20, color='lightgreen', edgecolor='black')
# plt.title('Aesthetic Distribution')
# plt.xlabel('Aesthetic Value')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(output_path, 'aesthetic_distribution.png'))

# # Video Length Distribution
# plt.figure(figsize=(10, 6))
# plt.hist(video_lengths, bins=20, color='orange', edgecolor='black')
# plt.title('Video Length Distribution')
# plt.xlabel('Video Length (fps * (face_cut[1] - face_cut[0]))')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(output_path, 'video_length_distribution.png'))

# # Resolution Distribution
# plt.figure(figsize=(10, 6))
# plt.hist(resolutions, bins=3, color='purple', edgecolor='black', rwidth=0.85)
# plt.title('Resolution Distribution')
# plt.xlabel('Resolution')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(output_path, 'resolution_distribution.png'))

# # Word Range Distribution for face_cap_qwen
# plt.figure(figsize=(10, 6))
# plt.hist(word_lengths, bins=20, color='gold', edgecolor='black')
# plt.title('Word Range Distribution for face_cap_qwen')
# plt.xlabel('Word Count')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(output_path, 'word_range_distribution.png'))

# # Generate and save WordCloud for face_cap_qwen
# all_words = ' '.join(face_cap)
# wordcloud = WordCloud(width=800, height=400, background_color='white', scale=4).generate(all_words)
# wordcloud.to_file(os.path.join(output_path, 'face_cap_qwen_wordcloud.png'))

# print("All plots and wordcloud have been saved to the local directory.")
