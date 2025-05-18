import json
import numpy as np

json_folder_path = "dataset1.json"
# output_image_path = "benchmark/wordcloud.png"

with open(json_folder_path, "r") as file:
    data = json.load(file)

prompts = [item["prompt"] for item in data.values()]
word_counts = [len(prompt.split()) for prompt in prompts]

bins = [
    0,
    50,
    100,
    150,
    200,
    float("inf"),
]  # Intervals: <50, 50-100, 100-150, 150-200, >200
bin_labels = ["<50", "50~100", "100~150", "150~200", ">200"]

interval_counts = np.histogram(word_counts, bins=bins)[0]

# plt.bar(bin_labels, interval_counts, color="skyblue", edgecolor="black")
# plt.title("Word Count Distribution of Prompts")
# plt.xlabel("Word Count Range")
# plt.ylabel("Frequency")
# plt.show()

for label, count in zip(bin_labels, interval_counts):
    print(f"Number of prompts in the range {label}: {count}")
