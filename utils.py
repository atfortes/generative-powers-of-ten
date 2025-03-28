import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def tensor_to_pil(images):
    pil_images = []
    for i in range(images.shape[0]):
        image = images[i].permute(1, 2, 0).cpu().detach().numpy()
        image = (image + 1) / 2 * 255
        image = image.astype(np.uint8)
        pil_images.append(Image.fromarray(image))
    return pil_images

def preprocess_photograph(photograph, height, width, device):
    transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    return transform(photograph).to(device)

def photo_collage(imgs, height, width):
    num_images = len(imgs)
    collage_width = width * num_images
    collage_height = height
    collage = Image.new('RGB', (collage_width, collage_height))

    for i, img in enumerate(imgs):
        collage.paste(img, (i * width, 0))
    return collage

def save_collage(imgs, dir, height, width, name):
    os.makedirs(dir, exist_ok=True)
    photo_collage(imgs, height, width).save(f'{dir}/{name}.png')

def save_images(imgs, dir, prompts):
    os.makedirs(dir, exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(f'{dir}/{i}_{prompts[i].lower().replace(".", "").replace(" ", "_")}.png')
