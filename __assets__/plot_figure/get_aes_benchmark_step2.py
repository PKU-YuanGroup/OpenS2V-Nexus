import json
import matplotlib.pyplot as plt


def categorize_aes_scores(aes_scores):
    categories = {"<4.5": 0, "4.5~5.0": 0, "5.0~5.5": 0, "5.5~6.0": 0, ">6.0": 0}

    for score in aes_scores:
        if score < 4.5:
            categories["<4.5"] += 1
        elif 4.5 <= score < 5.0:
            categories["4.5~5.0"] += 1
        elif 5.0 <= score < 5.5:
            categories["5.0~5.5"] += 1
        elif 5.5 <= score < 6.0:
            categories["5.5~6.0"] += 1
        else:
            categories[">6.0"] += 1

    return categories


def plot_aes_score_distribution(aes_scores, output_file):
    plt.figure(figsize=(8, 6))
    plt.hist(aes_scores, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Distribution of AES Scores")
    plt.xlabel("AES Score")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")


def main():
    results_file_path = "benchmark/aes_scores_benchmark.json"
    output_image_path = "benchmark/aes_score_distribution.png"

    with open(results_file_path, "r") as f:
        aes_scores_dict = json.load(f)

    aes_scores = list(aes_scores_dict.values())

    categories = categorize_aes_scores(aes_scores)
    print("AES score categories:")
    for category, count in categories.items():
        print(f"{category}: {count} scores")

    plot_aes_score_distribution(aes_scores, output_image_path)

    aes_min = min(aes_scores)
    aes_max = max(aes_scores)
    print(f"AES score range: {aes_min} to {aes_max}")


if __name__ == "__main__":
    main()
