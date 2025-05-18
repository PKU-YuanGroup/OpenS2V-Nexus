import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection,
)


@torch.no_grad
def run_aesthetic_laion(model, image):
    if not isinstance(image, list):
        image = [image]
    return model(image)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, aes_clip_path, aes_main_path):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(aes_clip_path)
        self.processor = CLIPProcessor.from_pretrained(aes_clip_path)

        self.mlp = MLP()
        state_dict = torch.load(
            aes_main_path, weights_only=True, map_location=torch.device("cpu")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip(**inputs)[0]
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def get_image_paths(root_dir):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_paths.append(os.path.join(subdir, file))
    return image_paths


def main():
    device = "cuda"
    aes_clip_path = "openai/clip-vit-large-patch14"
    aes_main_path = "OpenS2V-Weight/aesthetic-model.pth"
    root_image_path = "images"
    results_file_path = "benchmark/aes_scores_benchmark.json"

    aes_model = AestheticScorer(
        dtype=torch.float32, aes_clip_path=aes_clip_path, aes_main_path=aes_main_path
    ).to(device)

    image_paths = get_image_paths(root_image_path)
    aes_scores_dict = {}

    for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(e)
            continue
        frames = [image]
        aes_scores = run_aesthetic_laion(aes_model, frames)
        aes_score = aes_scores.mean().detach().item()
        aes_scores_dict[image_path] = aes_score

    with open(results_file_path, "w") as f:
        json.dump(aes_scores_dict, f, indent=4)

    print(f"Aesthetic scores saved in {results_file_path}")


if __name__ == "__main__":
    main()
