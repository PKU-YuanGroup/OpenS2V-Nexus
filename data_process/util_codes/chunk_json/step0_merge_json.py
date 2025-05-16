import argparse
import concurrent.futures
import json
import os
from functools import partial

from tqdm import tqdm


human_words = [
    "man",
    "woman",
    "people",
    "child",
    "adult",
    "boy",
    "girl",
    "person",
    "human",
    "family",
    "friend",
    "stranger",
    "neighbor",
    "relative",
    "leader",
    "follower",
    "individual",
    "citizen",
    "immigrant",
    "foreigner",
    "native",
    "worker",
    "employee",
    "employer",
    "manager",
    "teacher",
    "student",
    "doctor",
    "nurse",
    "patient",
    "lawyer",
    "judge",
    "soldier",
    "sailor",
    "pilot",
    "athlete",
    "artist",
    "writer",
    "poet",
    "musician",
    "singer",
    "dancer",
    "chef",
    "waiter",
    "businessman",
    "businesswoman",
    "entrepreneur",
    "scientist",
    "engineer",
    "technician",
    "farmer",
    "miner",
    "painter",
    "photographer",
    "designer",
    "architect",
    "inventor",
    "philosopher",
    "historian",
    "politician",
    "activist",
    "volunteer",
    "entrepreneur",
    "genius",
    "scholar",
    "intellectual",
    "rebel",
    "companion",
    "partner",
    "spouse",
    "girlfriend",
    "boyfriend",
    "husband",
    "wife",
    "grandfather",
    "grandmother",
    "uncle",
    "aunt",
    "nephew",
    "niece",
    "cousin",
    "baby",
    "toddler",
    "teenager",
    "senior",
    "elderly",
    "veteran",
    "survivor",
    "orphan",
    "refugee",
    "exile",
    "homeless",
    "criminal",
    "victim",
    "witness",
    "hero",
    "villain",
    "outsider",
]

human_words_lower = [word.lower() for word in human_words]


def contains_human_word(captions):
    for caption in captions:
        if any(human_word in caption.lower() for human_word in human_words_lower):
            return True
    return False


def process_file(filename, folder_path, merged_data):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                data = data[0]

            aesthetic = data["aesthetic"]
            motion = data["motion"]
            caption = data["cap"]
            cut = data["cut"]
            if contains_human_word(caption) and aesthetic > 4.75 and motion > 0.004:
                key = data["path"].replace(".mp4", f"_step1-{cut[0]}-{cut[1]}")
                merged_data[key] = {"metadata": data}
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
    parser.add_argument("--input_json_folder", type=str, default="0_demo_input/jsons")
    parser.add_argument(
        "--output_json_file", type=str, default="0_demo_output/step0/merge.json"
    )
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    merge_json_files(args.input_json_folder, args.output_json_file, args.num_workers)


if __name__ == "__main__":
    main()
