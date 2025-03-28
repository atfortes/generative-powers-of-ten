import os
import json
import torch
import argparse
from PIL import Image
import memory_management as mm
from utils import seed_everything, save_images
from model.gen_powers_10 import GenPowers10Pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='forest')
    parser.add_argument('--p', type=int, default=None)
    parser.add_argument('--DFIF_stage_1', type=str, default='XL', choices=['XL', 'L', 'M'])
    parser.add_argument('--DFIF_stage_2', type=str, default='L', choices=['L', 'M'])
    parser.add_argument('--negative', type=str, default='artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--viz_step', type=int, default=0)
    parser.add_argument('--cfg', type=float, default=7)
    parser.add_argument('--seed', type=int, default=83920174658)
    parser.add_argument('--use_photo', action='store_true')
    opt = parser.parse_args()

    seed_everything(opt.seed)
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(f'examples/{opt.name}'):
        raise ValueError(f'examples/{opt.name} does not exist')

    metadata = json.load(open(f'examples/{opt.name}/metadata.json', 'r'))

    prompts = metadata['prompts']
    p = opt.p or metadata['p']

    photograph = None
    if opt.use_photo and os.path.exists(f'examples/{opt.name}/photo.jpg'):
        photograph = Image.open(f'examples/{opt.name}/photo.jpg').resize((64, 64), Image.BILINEAR)

    opt.version = f'DFIF_{opt.DFIF_stage_1}_{opt.DFIF_stage_2}_X4'
    return opt.name, prompts, photograph, p, opt.version, opt.negative, opt.steps, opt.viz_step, opt.cfg, device, opt.seed


if __name__ == '__main__':
    name, prompts, photograph, p, version, negative, steps, viz_step, cfg, device, seed = parse_args()
    mm.gpu = device
    
    dir = f'examples/{name}/samples/' + ('photo_' if photograph is not None else '') + f'{version}' + f'_{seed}'
    os.makedirs(dir, exist_ok=True)

    pipeline = GenPowers10Pipeline(version)
    imgs = pipeline(prompts, negative, p, dir, num_inference_steps=steps, guidance_scale=cfg, photograph=photograph, viz_step=viz_step)
    save_images(imgs, dir, prompts)
