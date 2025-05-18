import json
from wordcloud import WordCloud

json_folder_path = "dataset1.json"
output_image_path = "benchmark/wordcloud.png"

with open(json_folder_path, "r") as file:
    data = json.load(file)

prompts = [item["prompt"] for item in data.values()]
all_prompts = " ".join(prompts)

wordcloud = WordCloud(
    width=800, height=400, background_color="white", scale=4
).generate(all_prompts)
wordcloud.to_file(output_image_path)
