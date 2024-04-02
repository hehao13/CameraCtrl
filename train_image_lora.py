import os
import math
import random
import time
import inspect
import argparse
import datetime
import subprocess
import omegaconf

from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer

from cameractrl.data.dataset import RealEstate10K
from cameractrl.utils.util import setup_logger, format_time


def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)

    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')

    return local_rank


def main(name: str,
         launcher: str,
         port: int,

         output_dir: str,
         pretrained_model_path: str,
         unet_subfolder: str,

         train_data: Dict,
         validation_data: Dict,
         cfg_random_null_text: bool = True,
         cfg_random_null_text_ratio: float = 0.1,

         noise_scheduler_kwargs: Dict = None,

         do_sanity_check: bool = True,
         max_train_epoch: int = -1,
         max_train_steps: int = 100,
         validation_steps: int = 100,
         validation_steps_tuple: Tuple = (-1,),

         learning_rate: float = 3e-5,
         lr_warmup_steps: int = 0,
         lr_scheduler: str = "constant",

         lora_rank: int = 4,

         num_workers: int = 32,
         train_batch_size: int = 1,
         adam_beta1: float = 0.9,
         adam_beta2: float = 0.999,
         adam_weight_decay: float = 1e-2,
         adam_epsilon: float = 1e-08,
         max_grad_norm: float = 1.0,
         gradient_accumulation_steps: int = 1,
         gradient_checkpointing: bool = False,
         checkpointing_epochs: int = 5,
         checkpointing_steps: int = -1,

         mixed_precision_training: bool = True,
         enable_xformers_memory_efficient_attention: bool = True,

         global_seed: int = 42,
         logger_interval: int = 10,

         resume_from: str = None
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher, port=port)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)

    *_, config = inspect.getargvalues(inspect.currentframe())

    logger = setup_logger(output_dir, global_rank)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder=unet_subfolder)

    # Freeze unet, vae, and text_encoder
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=lora_rank if lora_rank > 16 else hidden_size // lora_rank,
        )

    unet.set_attn_processor(lora_attn_procs)

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = AttnProcsLayers(unet.attn_processors)
    trainable_param_names = [pname for pname, p in lora_layers.named_parameters() if p.requires_grad]
    trainable_params = list(filter(lambda p: p.requires_grad, lora_layers.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        logger.info(f"trainable params number: {len(trainable_params)}")
        logger.info(f"trainable params name: {trainable_param_names}")
        logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = RealEstate10K(**train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    validation_pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_path, unet=unet, vae=vae,
                                                                  tokenizer=tokenizer, text_encoder=text_encoder,
                                                                  scheduler=noise_scheduler, safety_checker=None,)
    validation_pipeline.enable_vae_slicing()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    if resume_from is not None:
        logger.info(f"Resuming the training from the checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=text_encoder.device)
        global_step = ckpt['global_step']
        trained_iterations = (global_step % len(train_dataloader))
        first_epoch = int(global_step // len(train_dataloader))
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lora_state_dict = ckpt['lora_state_dict']
        _, uk = unet.load_state_dict({'module.' + k: v for k, v in lora_state_dict.items()}, strict=False)
        logger.info(f"Loading the lora weight done, unexpected keys: {uk}")
        logger.info(f"Loading done, training from the {global_step + 1}th iteration")
        lr_scheduler.last_epoch = first_epoch
    else:
        trained_iterations = 0

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        data_iter = iter(train_dataloader)
        for step in range(trained_iterations, len(train_dataloader)):
            iter_start_time = time.time()
            batch = next(data_iter)
            data_end_time = time.time()
            if cfg_random_null_text:
                batch['caption'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['caption']]

            # Data batch sanity check
            if epoch == first_epoch and step == 0 and do_sanity_check:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['caption']
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value / 2. + 0.5
                    torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")

            ### >>>> Training >>>> ###

            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['caption'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            # progress_bar.update(1)
            global_step += 1
            iter_end_time = time.time()

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "lora_state_dict": lora_layers.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
                logger.info(f"Saved state to {save_path} (global_step: {global_step})")

            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                if isinstance(train_data, omegaconf.listconfig.ListConfig):
                    height = train_data[0].sample_size[0] if not isinstance(train_data[0].sample_size, int) else train_data[0].sample_size
                    width = train_data[0].sample_size[1] if not isinstance(train_data[0].sample_size, int) else train_data[0].sample_size
                else:
                    height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                    width = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts

                for idx, prompt in enumerate(prompts):
                    sample = validation_pipeline(
                        prompt,
                        generator=generator,
                        height=height,
                        width=width,
                        num_inference_steps=validation_data.get("num_inference_steps", 50),
                        guidance_scale=validation_data.get("guidance_scale", 8.),
                    ).images[0]
                    sample = torchvision.transforms.functional.to_tensor(sample)
                    samples.append(sample)

                samples = torch.stack(samples)
                save_path = f"{output_dir}/samples/sample-{global_step}.png"
                torchvision.utils.save_image(samples, save_path, nrow=4)

                logger.info(f"Saved samples to {save_path}")

            if (global_step % logger_interval) == 0 or global_step == 0:
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                msg = f"Iter: {global_step}/{max_train_steps}, Loss: {loss.detach().item(): .4f}, " \
                      f"lr: {lr_scheduler.get_last_lr()}, Data time: {format_time(data_end_time - iter_start_time)}, " \
                      f"Iter time: {format_time(iter_end_time - data_end_time)}, " \
                      f"ETA: {format_time((iter_end_time - iter_start_time) * (max_train_steps - global_step))}, " \
                      f"GPU memory: {gpu_memory: .2f} G"
                logger.info(msg)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, port=args.port, **config)
