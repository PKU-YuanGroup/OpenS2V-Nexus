import torch
import numpy as np

import argparse
from omegaconf import OmegaConf

import sys
sys.path.append("lama_with_refiner")
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./repo/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        metavar="FILE",
        help="path to config file",
    )
    return parser


def get_inpaint_model():
    predict_config = OmegaConf.load('ckpt/default.yaml')
    predict_config.model.path = './ckpt/'
    predict_config.refiner.gpu_ids = '0'

    device = torch.device(predict_config.device)
    train_config_path = "ckpt/big-lama/config.yaml"

    train_config = OmegaConf.load(train_config_path)
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = "ckpt/big-lama/models/best.ckpt"

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


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import time
    from simple_lama_inpainting.utils.util import prepare_img_and_mask

    img_path = "/storage/ysh/Code/MultiID/Code/1_util_models/ConsisID-X/data_preprocess/output_step7/extracted_image_frame_20.png"
    mask_path = "/storage/ysh/Code/MultiID/Code/1_util_models/ConsisID-X/data_preprocess/output_step7/extracted_image_mask_20.png"
    
    inpaint_model, predict_config = get_inpaint_model()

    img = Image.open(img_path)
    masks = Image.open(mask_path)
    img, masks = prepare_img_and_mask(img, masks, device="cpu")
    batch = dict(image=img[0], mask=masks[0][0][None, ...])

    # import pdb;pdb.set_trace()
    batch['unpad_to_size'] = [torch.tensor([batch['image'].shape[1]]),torch.tensor([batch['image'].shape[2]])]
    batch['image'] = torch.tensor(pad_img_to_modulo(batch['image'], predict_config.dataset.pad_out_to_modulo))[None].to(predict_config.device)
    batch['mask'] = torch.tensor(pad_img_to_modulo(batch['mask'], predict_config.dataset.pad_out_to_modulo))[None].float().to(predict_config.device)

    cur_res = refine_predict(batch, inpaint_model, **predict_config.refiner)
    cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

    cur_res = Image.fromarray(cur_res)

    cur_res.save("2.png")
    