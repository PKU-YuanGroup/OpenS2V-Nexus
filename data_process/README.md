# <u>Data Processing Pipeline</u> by *OpenS2V-5M*
This repo describes how to process your own video like [OpenS2V-5M](https://huggingface.co/datasets/BestWishYsh/OpenS2V-5M) datasets in the [OpenS2V-Nexus](https://arxiv.org) paper. Alternatively, if you want to directly use OpenS2V-5M, please refer to [this guide](https://huggingface.co/datasets/BestWishYsh/OpenS2V-5M#%F0%9F%93%A3-usage).

We provide some resources that may help you: [Tutorial 1](https://github.com/PKU-YuanGroup/OpenS2V-Nexus/blob/main/data_process/README.md), [Tutorial 2](https://huggingface.co/datasets/BestWishYsh/OpenS2V-5M), [Demo Regular-Pairs Json](https://github.com/PKU-YuanGroup/OpenS2V-Nexus/blob/main/data_process/demo_result/step5/merge_final_json/dataset1.json), [Demo Cross-Pairs Json](https://github.com/PKU-YuanGroup/OpenS2V-Nexus/blob/main/data_process/demo_result/step6/cross-frames-pairs/final_output/gm1252760410-365677148.json),
[Demo Dataloader](https://github.com/PKU-YuanGroup/OpenS2V-Nexus/blob/main/data_process/demo_dataloader.py).

## 🎉 Overview

<div align=center>
<img src="https://github.com/user-attachments/assets/d695ed59-cbf5-4303-883b-626b116441e3">
</div>

## ⚙️ Requirements and Installation

We recommend the requirements as follows. You should first follow the [instructions](https://github.com/PKU-YuanGroup/OpenS2V-Nexus/?tab=readme-ov-file#%EF%B8%8F-requirements-and-installation) to install the *Base Environment* and obtain the *Base Checkpoints*.

### Environment

```bash
# 0. Activate Base Environment
conda activate opens2v

# 1. Install the Detection and Segmentation
cd OpenS2V-Nexus/data_process/util_codes/groundingsam2
export CUDA_HOME=/path/to/cuda/
pip install -e .
pip install --no-build-isolation -e grounding_dino
```

### Download Weight

The weights can be download with the following commands.

```bash
cd OpenS2V-Nexus

# 0. Get the Caption Weight
huggingface-cli download --repo-type model \
Qwen/Qwen2.5-VL-7B-Instruct \
--local-dir ckpts/Qwen/Qwen2.5-VL-7B-Instruct

# 1. Get the GmeScore Weight
huggingface-cli download --repo-type model \
Alibaba-NLP/gme-Qwen2-VL-7B-Instruct \
--local-dir ckpts/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct

# 2. Get the AestheticScore Weight
huggingface-cli download --repo-type model \
openai/clip-vit-large-patch14 \
--local-dir ckpts/openai/clip-vit-large-patch14

# 3. Get the Tag Weight (Optional)
huggingface-cli download --repo-type model \
Qwen/Qwen3-32B \
--local-dir ckpts/Qwen/Qwen3-32B
```

Once ready, the weights will be organized in this format:

```bash
📦 OpenS2V-Nexus/
├── 📂 ckpts/
│   ├── 📂 Alibaba-NLP
│   ├── 📂 LaMa
│   ├── 📂 Qwen
│   ├── 📂 face_extractor
│   ├── 📂 openai
│   ├── 📄 aesthetic-model.pth
│   ├── 📄 glint360k_curricular_face_r101_backbone.bin
│   ├── 📄 groundingdino_swint_ogc.pth
│   ├── 📄 sam2.1_hiera_large.pt
│   ├── 📄 yolo_world_v2_l_image_prompt_adapter-719a7afb.pth
```

## 🗝️ Usage

 *For all steps, we provide both input and output examples in the `demo_result` folder.*

### Step 0 - Format Input

For each video, you are required to organize the video metadata into a JSON format as specified below. Take `video_id1.mp4` as an example: if the video has a total duration of `95` frames, a width of `1280` pixels, a height of `720` pixels, and a frame rate of `30` fps, then the corresponding metadata should be structured as follows:

```bash
{
    "video_id1": {
        "metadata": {
            "cut": [
                0,
                95
            ],
            "crop": [
                0,
                1280,
                0,
                720
            ],
            "fps": 30,
            "num_frames": 95,
            "resolution": {
                "height": 720,
                "width": 1280
            },
            "path": "video_id1.mp4",
        }
    },
    "video_id2": {
    	...
    }
}
```

### Step 1 - Video Quality Filtering (Optional)

To ensure data quality, we strongly recommend using [video-dataset-scripts](https://github.com/huggingface/video-dataset-scripts) to filter low-quality videos. Overview of the available filters:

* Watermark detection
* Aesthetic scoring
* NSFW scoring
* Motion scoring
* Filtering videos w.r.t reference videos/images
* Shot categories (color, lighting, composition, etc.)

Once ready, the input should be organized in this format ( `cut` records intervals without jump cuts, while `crop` specifies regions free of watermarks):

```bash
{
    "video_id1_part1": {
        "metadata": {
            "cut": [
                30,
                95
            ],
            "crop": [
                0,
                1080,
                0,
                720
            ],
            "fps": 30,
            "num_frames": 65,
            "resolution": {
                "height": 720,
                "width": 1080
            },
            "path": "video_id1.mp4",
            "tech": 2.082,
            "motion": 0.01223986130207777,
            "aesthetic": 5.070407867431641
        }
    },
    "video_id1_part2": {
        "metadata": {
            "cut": [
                0,
                30
            ],
            "crop": [
                0,
                1080,
                0,
                720
            ],
            "fps": 30,
            "num_frames": 30,
            "resolution": {
                "height": 720,
                "width": 1080
            },
            "path": "video_id1.mp4",
            "tech": 2.082,
            "motion": 0.01223986130207777,
            "aesthetic": 5.070407867431641
        }
    },
    "video_id2_part1": {
    	...
    }
}
```

### Step 2 - Human-Centric Filtering (Optional)

If you wish to obtain human-centered data, the following steps can be implemented:

```bash
python step1_get_bbox.py
python step2_get_pure_person_clip.py

# visualization (optional)
cd OpenS2V-Nexus/data_process/util_codes/visualize_annotation
python step1_visualize.py
python step2_visualize.py
```

### Step 3 - Get Subject-Centric Video Caption

We prompt *Qwen2.5-VL-7B* to describe the appearance and changes of the subject while preserving essential elements of the video, such as environmental context and camera movements, to get the subject-centric video caption.

```bash
python step3-0_merge_json.py
python step3-1_get_caption.py
```

### Step 4 - Extract Subject Tags

To obtain high-quality reference images, we use *DeepSeekV3* to extract keywords related to the environment and objects from the caption.

```bash
python step4-0_merge_json.py

# Use DeepSeek API
python step4-1_get_tag_api.py

# Use Qwen3 Locally (Optional)
python step4-1_get_tag_local.py
```

### Step 5 - Regular Data Annotation

We then input the first frame and tags into GroundingDino to extract reference images for each video. Then, the bounding boxes obtained from the previous step are fed into SAM2.1, which generates a mask for each subject. This mask can be used to extract subjects without background pixels.

```bash
python step5-0_merge_json.py
python step5-1_offload_frame.py
python step5-2_get_subject_image.py

# visualization (optional)
cd OpenS2V-Nexus/data_process/util_codes/visualize_annotation
python step6_visualize_image.py
python step6_visualize_video.py
```

### Step 6 - Nexus Data Annotation

We ensure subject‐information diversity by (1) segmenting subjects and building pairing information via cross‐video associations and (2) prompting GPT-4o on raw frames to synthesize multi-view representations.

```bash
python step6-0_merge_json.py

# For Cross-Frame Pairs
python step6-1_get_cluster.py
python step6-2_get_cross-frame.py
python step6-3_merge_json.py

# For GPT-Frame Pairs
python step6-4_get_gpt-frame.py
```

### Step 7 - Fetch High-Quality Subset

We curate a high-quality subset via tag whitelisting + sample/mask filtering + mask post-processing.

#### Tag Statistics & Filtering

1. **Tag counting**: count tags from all ref_img across samples.

2. **Rule-based filtering**: remove tags containing non-English characters or special symbols.

3. **GPT filtering**: use custom rules to remove overly-common / uninformative tags.

#### Sample & Mask Filtering

Apply the following filters in order:

- **Duration & resolution filter**: remove samples withduration < 2s or > 10s, or shorter side < 480.

- **Tag filter (whitelist)**: remove ref_img whose tag is not in the Allowed Tag List.

- **Aesthetic filter**: remove ref_img with aesthetic score < 4.5.

- **Mask area filter**: remove ref_img whose mask area is < 1/32 of the full image area.

#### Mask Post-processing

- **Mask dedup**: compute pairwise mask IoU; if IoU > 0.8, keep the mask with larger area.

- **Mask selection**: keep the top 6 masks with the highest aesthetic scores.

## 🗝️ Demo Dataloader

Regarding how to use OpenS2V-5M during the training phase, we provide a demo dataloader [here](https://github.com/PKU-YuanGroup/OpenS2V-Nexus/tree/main/data_process/demo_dataloader.py). Alternatively, you can refer to the [BindWeave code](https://github.com/bytedance/BindWeave), which uses OpenS2V-5M for training.

## 🔒 Limitation

- Although the current data pipeline can generate high-quality Regular Data and Nexus Data suitable for Subject-to-Video generation, it incurs substantial computational costs. We will continue to optimize the code in the future.
