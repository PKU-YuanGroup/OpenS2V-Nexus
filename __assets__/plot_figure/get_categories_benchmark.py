import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm


def extract_categories_from_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    categories = set()
    for obj in data.values():
        for label in obj.get("class_label", []):
            categories.add(label.lower())

    return list(categories)


def plot_biological_categories_fixed_labels(
    categories, filename="biological_categories_fixed_labels.png"
):
    num_categories = len(categories)
    angle = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

    colors = cm.tab20(np.linspace(0, 1, num_categories))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    inner_radius = 0.6
    outer_radius = 1
    width = (outer_radius - inner_radius) / num_categories
    bar_width = width * 11

    font_size = 12 if num_categories < 20 else max(8, 120 / num_categories)

    for i in range(num_categories):
        ax.bar(
            angle[i],
            1,
            width=bar_width,
            bottom=inner_radius,
            align="edge",
            color=colors[i],
        )

    for i, bar in enumerate(ax.patches):
        angle_offset = bar.get_x() + bar.get_width() / 2
        ax.text(
            angle_offset,
            (inner_radius + outer_radius) / 2 + 0.4,
            categories[i],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=font_size,
            rotation=angle_offset * 180 / np.pi,
        )

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.grid(False)

    ax.spines["polar"].set_visible(False)

    ax.set_title("")

    plt.savefig(filename, dpi=300, transparent=True)
    plt.close()


json_file = "Open-Domain_Eval.json"
output_image_path = "./benchmark/pie_benchmark.png"
categories = extract_categories_from_json(json_file)

plot_biological_categories_fixed_labels(categories, filename=output_image_path)
