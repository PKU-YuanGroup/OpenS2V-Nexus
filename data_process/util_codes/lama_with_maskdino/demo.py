import os
import sys

sys.path.append('./repo/lama-with-refiner/')
sys.path.append('./repo/MaskDINO/')
sys.path.append('./repo/MaskDINO/demo/')

import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2

import argparse
from omegaconf import OmegaConf

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config
from predictor import VisualizationDemo

from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint

import gradio as gr

className = {'person': 0,
             'bicycle': 1,
             'car': 2,
             'motorcycle': 3,
             'airplane': 4,
             'bus': 5,
             'train': 6,
             'truck': 7,
             'boat': 8,
             'traffic light': 9,
             'fire hydrant': 10,
             'stop sign': 11,
             'parking meter': 12,
             'bench': 13,
             'bird': 14,
             'cat': 15,
             'dog': 16,
             'horse': 17,
             'sheep': 18,
             'cow': 19,
             'elephant': 20,
             'bear': 21,
             'zebra': 22,
             'giraffe': 23,
             'backpack': 24,
             'umbrella': 25,
             'handbag': 26,
             'tie': 27,
             'suitcase': 28,
             'frisbee': 29,
             'skis': 30,
             'snowboard': 31,
             'sports ball': 32,
             'kite': 33,
             'baseball bat': 34,
             'baseball glove': 35,
             'skateboard': 36,
             'surfboard': 37,
             'tennis racket': 38,
             'bottle': 39,
             'wine glass': 40,
             'cup': 41,
             'fork': 42,
             'knife': 43,
             'spoon': 44,
             'bowl': 45,
             'banana': 46,
             'apple': 47,
             'sandwich': 48,
             'orange': 49,
             'broccoli': 50,
             'carrot': 51,
             'hot dog': 52,
             'pizza': 53,
             'donut': 54,
             'cake': 55,
             'chair': 56,
             'couch': 57,
             'potted plant': 58,
             'bed': 59,
             'dining table': 60,
             'toilet': 61,
             'tv': 62,
             'laptop': 63,
             'mouse': 64,
             'remote': 65,
             'keyboard': 66,
             'cell phone': 67,
             'microwave': 68,
             'oven': 69,
             'toaster': 70,
             'sink': 71,
             'refrigerator': 72,
             'book': 73,
             'clock': 74,
             'vase': 75,
             'scissors': 76,
             'teddy bear': 77,
             'hair drier': 78,
             'toothbrush': 79}

def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./repo/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        metavar="FILE",
        help="path to config file",
    )
    return parser

def setup_cfg():
    args = get_parser().parse_args(args=[])
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = './ckpt/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth'
    cfg.freeze()
    return cfg
	
def get_seg_model():
    cfg = setup_cfg()
    model = VisualizationDemo(cfg)
    return model

def get_inpaint_model():
    predict_config = OmegaConf.load('./repo/lama-with-refiner/configs/prediction/default.yaml')
    predict_config.model.path = './ckpt/models/'
    predict_config.refiner.gpu_ids = '0'

    device = torch.device(predict_config.device)
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')

    train_config = OmegaConf.load(train_config_path)
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(predict_config.model.path, 
                                   'models', 
                                   predict_config.model.checkpoint)

    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    return model,predict_config
	
def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')
	
seg_model = get_seg_model()
inpaint_model,predict_config = get_inpaint_model()

def inference(img,class_name,confidence_score,sigma,mask_threshold):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    predictions, visualized_output = seg_model.run_on_image(img)

    img = img.astype('float32') / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))

    preds = predictions['instances'].get_fields()

    masks = preds['pred_masks'][torch.logical_and(preds['pred_classes'] == className[class_name], preds['scores'] > confidence_score)]
    masks = torch.max(masks,axis=0)
    masks = masks.values.cpu().numpy()
    masks = gaussian_filter(masks, sigma=sigma)
    masks = (masks > mask_threshold) * 255

    batch = dict(image=img, mask=masks[None, ...])

    batch['unpad_to_size'] = [torch.tensor([batch['image'].shape[1]]),torch.tensor([batch['image'].shape[2]])]
    batch['image'] = torch.tensor(pad_img_to_modulo(batch['image'], predict_config.dataset.pad_out_to_modulo))[None].to(predict_config.device)
    batch['mask'] = torch.tensor(pad_img_to_modulo(batch['mask'], predict_config.dataset.pad_out_to_modulo))[None].float().to(predict_config.device)

    cur_res = refine_predict(batch, inpaint_model, **predict_config.refiner)
    cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

    return cur_res

demo = gr.Interface(fn=inference, 
                    inputs=[gr.Image(label='input'),
                            gr.Dropdown(list(className.keys()),value='person',label= 'class name'),
                            gr.Slider(0, 1, value=0.5, step=0.05,label='confidence score'),
                            gr.Slider(1, 20, value=7, step=1,label='gaussian blur kernel size'),
                            gr.Slider(0, 1, value=0.2, step=0.05,label='mask threshold')],
                    outputs="image",
                    examples=[["./images/buildings.png",'person',0.5,7,0.2],
                                  ["./images/park.png",'person',0.5,7,0.2],
                                  ["./images/cat.png",'remote',0.5,7,0.2]])

demo.launch()   