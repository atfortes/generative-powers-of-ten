import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import memory_management as mm
from model.zoom_stack import ZoomStack
from transformers import T5Tokenizer, T5EncoderModel
from utils import save_collage, preprocess_photograph, numpy_to_pil
from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionUpscalePipeline, IFSuperResolutionPipeline


class GenPowers10Pipeline(nn.Module):
    def __init__(self, version):
        super().__init__()

        self.version = version
        _, stage_1, stage_2, _ = self.version.split('_')

        print(f'[INFO] loading DeepFloyd-IF...')

        if stage_1 == 'XL':
            model_key_1 = 'DeepFloyd/IF-I-XL-v1.0'
        elif stage_1 == 'L':
            model_key_1 = 'DeepFloyd/IF-I-L-v1.0'
        elif stage_1 == 'M':
            model_key_1 = 'DeepFloyd/IF-I-M-v1.0'
        else:
            raise ValueError(f'Invalid stage 1: {stage_1}')
        
        if stage_2 == 'L':
            model_key_2 = 'DeepFloyd/IF-II-L-v1.0'
        elif stage_2 == 'M':
            model_key_2 = 'DeepFloyd/IF-II-M-v1.0'
        else:
            raise ValueError(f'Invalid stage 2: {stage_2}')

        self.scheduler = DDPMScheduler.from_pretrained(model_key_1, subfolder="scheduler", torch_dtype=torch.float16)
        self.scheduler.variance_type = 'fixed_small'
        self.scheduler.config.variance_type = 'fixed_small'

        self.tokenizer = T5Tokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl")
        self.text_encoder = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl", torch_dtype=torch.float16).requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(model_key_1, subfolder="unet", torch_dtype=torch.float16, variant="fp16").requires_grad_(False)

        self.stage_2_pipe = IFSuperResolutionPipeline.from_pretrained(model_key_2, text_encoder=None, variant="fp16", torch_dtype=torch.float16)
        self.stage_2_pipe.safety_checker = None
        self.stage_2_pipe.watermarker = None

        self.stage_3_pipe = StableDiffusionUpscalePipeline.from_pretrained('stabilityai/stable-diffusion-x4-upscaler', torch_dtype=torch.float16)
        self.stage_3_pipe.safety_checker = None
        self.stage_3_pipe.watermarker = None

        print(f'[INFO] loaded DeepFloyd-IF!')

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        prompts = [prompt.lower().strip() for prompt in prompts]
        text_inputs = self.tokenizer(prompts, padding='max_length', max_length=77,
                                    truncation=True, add_special_tokens=True, return_tensors='pt')
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(mm.gpu),
                                          text_inputs.attention_mask.to(mm.gpu))[0]

        negative_prompts = [negative_prompt.lower().strip() for negative_prompt in negative_prompts]
        uncond_inputs = self.tokenizer(negative_prompts, padding='max_length', max_length=77,
                                    truncation=True, add_special_tokens=True, return_tensors='pt')
        uncond_embeds = self.text_encoder(uncond_inputs.input_ids.to(mm.gpu),
                                          uncond_inputs.attention_mask.to(mm.gpu))[0]

        return uncond_embeds, prompt_embeds

    def ddpm_update(self, intermediate_images, images, noise, t, alpha_prod_t_prev):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * images + current_sample_coeff * intermediate_images
        
        variance = (self.scheduler._get_variance(t) ** 0.5) * noise if t > 0 else 0

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
    
    @torch.no_grad()
    def unet_forward(self, intermediate_images, t, text_embeds, guidance_scale):
        model_input = torch.cat([intermediate_images] * 2)
        self.scheduler.scale_model_input(model_input, t)

        noise_pred = self.unet(
            model_input,
            t,
            encoder_hidden_states=text_embeds,
            return_dict=False,
        )[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)

        return noise_pred

    def photograph_loss(self, zoom_stack, denoised_images, photograph):
        loss = 0
        for i, denoised_image in enumerate(denoised_images):
            downscaled_image, mask = zoom_stack.downscale_and_pad(denoised_image, i)
            masked_diff = downscaled_image - mask * photograph
            loss += torch.sum(masked_diff ** 2)
        return loss

    @torch.no_grad()
    def __call__(self, prompts, negative_prompt, p, dir, num_inference_steps=50,
                 guidance_scale=7.5, photograph=None, generator=None, viz_step=10):

        height = 64
        width = 64
        num_levels = len(prompts)
        intermediate_images = torch.randn(
            (num_levels, self.unet.config.in_channels, height, width),
            generator=generator,
            dtype=torch.float16
        ).to(device=mm.gpu)
        intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
        zoom_stack = ZoomStack(num_levels, height, width, p, intermediate_images.shape)

        # prompts -> text embeds
        negative_prompts = [negative_prompt] * num_levels
        mm.load_models_to_gpu([self.text_encoder])
        uncond_embeds, prompt_embeds = self.get_text_embeds(prompts, negative_prompts)
        text_embeds = torch.cat([uncond_embeds, prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        if photograph is not None:
            photograph = preprocess_photograph(photograph, height, width, mm.gpu)
        
        mm.load_models_to_gpu([self.unet])
        for counter, t in enumerate(tqdm(timesteps)):

            # render: x_t = Pi_image(L_t); e_t = Pi_noise(E)
            images = torch.stack([zoom_stack.render_image(i) for i in range(num_levels)]).to(device=mm.gpu, dtype=torch.float16)
            zoom_stack.sample_noise(zoom_stack.E.shape)
            noise = torch.stack([zoom_stack.render_noise(i) for i in range(num_levels)]).to(device=mm.gpu, dtype=torch.float16)
            
            # ddpm update: z_t-1 = DDPM_update(z_t, x_t, e)
            prev_t = self.scheduler.previous_timestep(t)
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.one
            intermediate_images = self.ddpm_update(intermediate_images, images, noise, t, alpha_prod_t_prev)

            # unet forward: \hat{e}_t-1 = (1+w)*unet(z_t-1, t-1, y) - w*unet(z_t-1, t-1)
            noise_pred = self.unet_forward(intermediate_images, t, text_embeds, guidance_scale)

            # update images (epsilon): \hat{x}_t-1 = (z_t-1 - sigma_t-1 * \hat{e}_t-1) / alpha_prod_t-1
            denoised_images = (intermediate_images - (1 - alpha_prod_t_prev) ** (0.5) * noise_pred) / alpha_prod_t_prev ** (0.5)
            if self.scheduler.config.thresholding:
                denoised_images = self.scheduler._threshold_sample(denoised_images)
            elif self.scheduler.config.clip_sample:
                denoised_images = denoised_images.clamp(-self.scheduler.config.clip_sample_range, self.scheduler.config.clip_sample_range)

            # update the denoised images for photograph-based zoom
            if photograph is not None and counter < num_inference_steps * 0.8:
                with torch.enable_grad():
                    denoised_images_copy = denoised_images.clone().to(dtype=torch.float32).requires_grad_(True)
                    optimizer = optim.Adam([denoised_images_copy], lr=0.1)
                    for _ in range(5):
                        optimizer.zero_grad()
                        loss = self.photograph_loss(zoom_stack, denoised_images_copy, photograph)
                        loss.backward()
                        optimizer.step()
                    denoised_images = denoised_images_copy.detach().to(dtype=torch.float16)
            # multi-resolution blending: L_t-1 = Blending(\hat{x}_t-1)
            zoom_stack.multi_resolution_blending(denoised_images)

            if viz_step > 0 and counter % viz_step == 0:
                viz_images = (zoom_stack.L / 2 + 0.5).clamp(0, 1)
                viz_images = viz_images.cpu().permute(0, 2, 3, 1).float().numpy()
                viz_images = numpy_to_pil(viz_images)
                save_collage(viz_images, os.path.join(dir, 'steps'), height, width, f'{counter}')

        stage_1_output = torch.stack([zoom_stack.render_image(i) for i in range(num_levels)])
        images = (stage_1_output / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = numpy_to_pil(images)
        save_collage(images, dir, height, width, f'collage_{height}_{width}')

        # stage 2
        print("[INFO] upscaling stage 2")
        mm.load_models_to_gpu([self.stage_2_pipe])
        stage_2_output = self.stage_2_pipe(
            image=stage_1_output,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=uncond_embeds,
            guidance_scale=10.0,
            noise_level=128,
            generator=generator,
            output_type='pt'
        ).images

        zoom_stack.L = stage_2_output
        zoom_stack.H = 256
        zoom_stack.W = 256
        zoom_stack.calculate_max_depth()
        images = torch.stack([zoom_stack.render_image(i) for i in range(num_levels)])
        zoom_stack.multi_resolution_blending(images)
        images = (zoom_stack.L / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = numpy_to_pil(images)
        save_collage(images, dir, 256, 256, f'collage_256_256')

        # stage 3
        print("[INFO] upscaling stage 3")
        mm.load_models_to_gpu([self.stage_3_pipe])
        
        # Process in batches of 7 to avoid OOM
        batch_size = 7
        upscaled_images = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            batch_neg_prompts = negative_prompts[i:i+batch_size]
            
            batch_upscaled = self.stage_3_pipe(
                image=batch_images,
                prompt=batch_prompts, 
                negative_prompt=batch_neg_prompts,
                guidance_scale=10.0,
                noise_level=128,
                generator=generator,
                output_type='pil'
            ).images
            
            upscaled_images.extend(batch_upscaled)

        save_collage(upscaled_images, dir, 1024, 1024, f'collage_1024_1024')
        print(f'[INFO] saved images to {dir}.')

        return upscaled_images
