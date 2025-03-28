import PIL
import math
import torch
import torchvision.transforms.functional as TF


class ZoomStack:
    def __init__(self, num_levels, height, width, p, noise_shape, min_size=2):
        self.H = height
        self.W = width
        self.p = p
        self.min_size = min_size
        self.max_depth = self.calculate_max_depth()
        self.p_levels = [self.p ** i for i in range(num_levels)]

        self.L = torch.zeros(num_levels, 3, height, width)
        self.E = torch.randn(noise_shape)
    
    def calculate_max_depth(self):
        max_depth = int(math.log(self.H / self.min_size) / math.log(self.p))
        max_depth = max(1, max_depth)
        return max_depth

    def crop_center(self, image, level_diff):
        h, w = image.shape[-2:]
        h_down = h // self.p_levels[level_diff]
        w_down = w // self.p_levels[level_diff]
        h1 = (h - h_down) // 2
        h2 = h1 + h_down
        w1 = (w - w_down) // 2
        w2 = w1 + w_down
        return image[:, h1:h2, w1:w2]

    def downscale(self, image, level_diff):
        image_c = image.clone()
        p_level = self.p_levels[level_diff]
        kernel_size = max(3, min((p_level // 2) * 2 + 1, image.shape[-2], image.shape[-1]) | 1)
        sigma = p_level / 2
        image_c = TF.gaussian_blur(image_c, kernel_size=kernel_size, sigma=sigma)
        return image_c[:, ::p_level, ::p_level]

    def downscale_and_pad(self, image, level_diff):
        downscaled_image = self.downscale(image, level_diff)
        padded_image = torch.zeros_like(image)

        h_down, w_down = downscaled_image.shape[-2:]
        h, w = image.shape[-2:]
        h1 = (h - h_down) // 2
        h2 = h1 + h_down
        w1 = (w - w_down) // 2
        w2 = w1 + w_down
        padded_image[..., h1:h2, w1:w2] = downscaled_image

        mask = torch.zeros_like(image)
        mask[..., h1:h2, w1:w2] = 1

        return padded_image, mask

    def render_image(self, i):
        """
        Pi_image operator from the paper
        i: level
        """
        image = self.L[i]
        max_level_diff = min(self.max_depth, len(self.L) - i - 1)
        for level_diff in range(1, max_level_diff + 1):
            d_image, mask = self.downscale_and_pad(self.L[i + level_diff], level_diff)
            image = mask * d_image + (1 - mask) * image
        return image

    def sample_noise(self, shape):
        self.E = torch.randn(shape)

    def render_noise(self, i):
        """
        Pi_noise operator from the paper
        i: level
        """
        noise = self.E[i]
        max_level_diff = min(self.max_depth, len(self.E) - i - 1)
        for level_diff in range(1, max_level_diff + 1):
            noise_image, mask = self.downscale_and_pad(self.E[i + level_diff], level_diff)
            noise = (self.p_levels[level_diff]) * mask * noise_image + (1 - mask) * noise
        return noise

    def laplacian_pyramid(self, image, levels):
        gaussian_pyramid = [image]
        laplacian_pyramid = []

        for _ in range(levels - 1):
            gaussian_pyramid.append(self.downscale(gaussian_pyramid[-1], 1))
        for i in range(levels - 1):
            up = TF.resize(gaussian_pyramid[i + 1], gaussian_pyramid[i].shape[-2:], interpolation=PIL.Image.BILINEAR)
            d = gaussian_pyramid[i] - up
            laplacian_pyramid.append(d)
        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid
    
    def reconstruct_from_pyramid(self, pyramids):
        image = pyramids[-1]
        for laplacian in reversed(pyramids[:-1]):
            image = TF.resize(image, laplacian.shape[-2:], interpolation=PIL.Image.BILINEAR)
            image += laplacian
        return image

    def multi_resolution_blending(self, images, method="laplacian"):
        for i in range(len(self.L)):
            if method == "laplacian":
                laplacian_pyramids = []
                depth = min(self.max_depth, i + 1)
                for level_diff in range(depth):
                    image_crop = self.crop_center(images[i-level_diff], level_diff)
                    pyramid = self.laplacian_pyramid(image_crop, levels=(depth-level_diff))
                    laplacian_pyramids.append(pyramid)

                avg_pyramid = []
                for level_diff in range(depth):
                    current_level = []
                    for pyramid in laplacian_pyramids:
                        if len(pyramid) > 0:
                            current_level.append(pyramid.pop(-1))
                    if current_level:
                        current_level = torch.stack(current_level).mean(dim=0)
                        avg_pyramid.append(current_level)
                    else:
                        break

                if not avg_pyramid:
                    continue

                avg_pyramid.reverse()
                blended_image = self.reconstruct_from_pyramid(avg_pyramid)

            self.L[i] = blended_image
