import omegaconf.listconfig
import os
import math
import random
import time
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.models.attention_processor import AttnProcessor

from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange

from cameractrl.data.dataset import RealEstate10KPose
from cameractrl.utils.util import setup_logger, format_time, save_videos_grid
from cameractrl.pipelines.pipeline_animation import CameraCtrlPipeline
from cameractrl.models.unet import UNet3DConditionModelPoseCond
from cameractrl.models.pose_adaptor import CameraPoseEncoder, PoseAdaptor
from cameractrl.models.attention_processor import AttnProcessor as CustomizedAttnProcessor


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

         train_data: Dict,
         validation_data: Dict,
         cfg_random_null_text: bool = True,
         cfg_random_null_text_ratio: float = 0.1,

         unet_additional_kwargs: Dict = {},
         unet_subfolder: str = "unet",

         lora_rank: int = 4,
         lora_scale: float = 1.0,
         lora_ckpt: str = None,
         motion_module_ckpt: str = "",
         motion_lora_rank: int = 0,
         motion_lora_scale: float = 1.0,

         pose_encoder_kwargs: Dict = None,
         attention_processor_kwargs: Dict = None,
         noise_scheduler_kwargs: Dict = None,

         do_sanity_check: bool = True,

         max_train_epoch: int = -1,
         max_train_steps: int = 100,
         validation_steps: int = 100,
         validation_steps_tuple: Tuple = (-1,),

         learning_rate: float = 3e-5,
         lr_warmup_steps: int = 0,
         lr_scheduler: str = "constant",

         num_workers: int = 32,
         train_batch_size: int = 1,
         adam_beta1: float = 0.9,
         adam_beta2: float = 0.999,
         adam_weight_decay: float = 1e-2,
         adam_epsilon: float = 1e-08,
         max_grad_norm: float = 1.0,
         gradient_accumulation_steps: int = 1,
         checkpointing_epochs: int = 5,
         checkpointing_steps: int = -1,

         mixed_precision_training: bool = True,

         global_seed: int = 42,
         logger_interval: int = 10,
         resume_from: str = None,
         ):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank = init_dist(launcher=launcher, port=port)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
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
    unet = UNet3DConditionModelPoseCond.from_pretrained_2d(pretrained_model_path, subfolder=unet_subfolder,
                                                           unet_additional_kwargs=unet_additional_kwargs)
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)

    # init attention processor
    logger.info(f"Setting the attention processors")
    unet.set_all_attn_processor(add_spatial_lora=lora_ckpt is not None,
                                add_motion_lora=motion_lora_rank > 0,
                                lora_kwargs={"lora_rank": lora_rank, "lora_scale": lora_scale},
                                motion_lora_kwargs={"lora_rank": motion_lora_rank, "lora_scale": motion_lora_scale},
                                **attention_processor_kwargs)

    if lora_ckpt is not None:
        logger.info(f"Loading the image lora checkpoint from {lora_ckpt}")
        lora_checkpoints = torch.load(lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        logger.info(f'Loading done')
    else:
        logger.info(f'We do not add image lora')

    if motion_module_ckpt != "":
        logger.info(f"Loading the motion module checkpoint from {motion_module_ckpt}")
        mm_checkpoints = torch.load(motion_module_ckpt, map_location=unet.device)
        if 'motion_module_state_dict' in mm_checkpoints:
            mm_checkpoints = {k.replace('module.', ''): v for k, v in mm_checkpoints['motion_module_state_dict'].items()}
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        logger.info("Loading done")
    else:
        logger.info(f"We do not load pretrained motion module checkpoint")

    # Freeze vae, and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    spatial_attn_proc_modules = torch.nn.ModuleList([v for v in unet.attn_processors.values()
                                                     if not isinstance(v, (CustomizedAttnProcessor, AttnProcessor))])
    temporal_attn_proc_modules = torch.nn.ModuleList([v for v in unet.mm_attn_processors.values()
                                                      if not isinstance(v, (CustomizedAttnProcessor, AttnProcessor))])
    spatial_attn_proc_modules.requires_grad_(True)
    temporal_attn_proc_modules.requires_grad_(True)
    pose_encoder.requires_grad_(True)
    # set requires_grad of image lora to False
    for n, p in spatial_attn_proc_modules.named_parameters():
        if 'lora' in n:
            p.requires_grad = False
            logger.info(f'Setting the `requires_grad` of parameter {n} to false')
    pose_adaptor = PoseAdaptor(unet, pose_encoder)

    encoder_trainable_params = list(filter(lambda p: p.requires_grad, pose_encoder.parameters()))
    encoder_trainable_param_names = [p[0] for p in
                                     filter(lambda p: p[1].requires_grad, pose_encoder.named_parameters())]
    attention_trainable_params = [v for k, v in unet.named_parameters() if v.requires_grad and 'merge' in k and 'lora' not in k]
    attention_trainable_param_names = [k for k, v in unet.named_parameters() if v.requires_grad and 'merge' in k and 'lora' not in k]

    trainable_params = encoder_trainable_params + attention_trainable_params
    trainable_param_names = encoder_trainable_param_names + attention_trainable_param_names

    if is_main_process:
        logger.info(f"trainable parameter number: {len(trainable_params)}")
        logger.info(f"encoder trainable number: {len(encoder_trainable_params)}")
        logger.info(f"attention processor trainable number: {len(attention_trainable_params)}")
        logger.info(f"trainable parameter names: {trainable_param_names}")
        logger.info(f"encoder trainable scale: {sum(p.numel() for p in encoder_trainable_params) / 1e6:.3f} M")
        logger.info(f"attention processor trainable scale: {sum(p.numel() for p in attention_trainable_params) / 1e6:.3f} M")
        logger.info(f"trainable parameter scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    logger.info(f'Building training datasets')
    train_dataset = RealEstate10KPose(**train_data)
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

    # Get the validation dataset
    logger.info(f'Building validation datasets')
    validation_dataset = RealEstate10KPose(**validation_data)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
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
    validation_pipeline = CameraCtrlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        pose_encoder=pose_encoder)
    validation_pipeline.enable_vae_slicing()

    # DDP wrapper
    pose_adaptor.to(local_rank)
    pose_adaptor = DDP(pose_adaptor, device_ids=[local_rank], output_device=local_rank)

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

    if resume_from is not None:
        logger.info(f"Resuming the training from the checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=pose_adaptor.device)
        global_step = ckpt['global_step']
        trained_iterations = (global_step % len(train_dataloader))
        first_epoch = int(global_step // len(train_dataloader))
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        pose_encoder_state_dict = ckpt['pose_encoder_state_dict']
        attention_processor_state_dict = ckpt['attention_processor_state_dict']
        pose_enc_m, pose_enc_u = pose_adaptor.module.pose_encoder.load_state_dict(pose_encoder_state_dict, strict=False)
        import pdb
        pdb.set_trace()
        assert len(pose_enc_m) == 0 and len(pose_enc_u) == 0
        _, attention_processor_u = pose_adaptor.module.unet.load_state_dict(attention_processor_state_dict, strict=False)
        assert len(attention_processor_u) == 0
        logger.info(f"Loading the pose encoder and attention processor weights done.")
        logger.info(f"Loading done, resuming training from the {global_step + 1}th iteration")
        lr_scheduler.last_epoch = first_epoch
    else:
        trained_iterations = 0

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        pose_adaptor.train()

        data_iter = iter(train_dataloader)
        for step in range(trained_iterations, len(train_dataloader)):

            iter_start_time = time.time()
            batch = next(data_iter)
            data_end_time = time.time()
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]

            # Data batch sanity check
            if epoch == first_epoch and step == 0 and do_sanity_check:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value,
                                     f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif",
                                     rescale=True)

            ### >>>> Training >>>> ###

            # Convert videos to latent space
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)  # [b, c, f, h, w]
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # [b, c, f h, w]

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                    return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]  # b l c

            # Predict the noise residual and compute loss
            # Mixed-precision training
            plucker_embedding = batch["plucker_embedding"].to(device=local_rank)  # [b, f, 6, h, w]
            plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")  # [b, 6, f h, w]
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = pose_adaptor(noisy_latents,
                                          timesteps,
                                          encoder_hidden_states=encoder_hidden_states,
                                          pose_embedding=plucker_embedding)  # [b c f h w]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, pose_adaptor.parameters()),
                                               max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, pose_adaptor.parameters()),
                                               max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            iter_end_time = time.time()

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "pose_encoder_state_dict": pose_adaptor.module.pose_encoder.state_dict(),
                    "attention_processor_state_dict": {k: v for k, v in unet.state_dict().items()
                                                       if k in attention_trainable_param_names},
                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
                logger.info(f"Saved state to {save_path} (global_step: {global_step})")

            # Periodically validation
            if is_main_process and (
                    (global_step + 1) % validation_steps == 0 or (global_step + 1) in validation_steps_tuple):

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                if isinstance(train_data, omegaconf.listconfig.ListConfig):
                    height = train_data[0].sample_size[0] if not isinstance(train_data[0].sample_size, int) else \
                    train_data[0].sample_size
                    width = train_data[0].sample_size[1] if not isinstance(train_data[0].sample_size, int) else \
                    train_data[0].sample_size
                else:
                    height = train_data.sample_size[0] if not isinstance(train_data.sample_size,
                                                                         int) else train_data.sample_size
                    width = train_data.sample_size[1] if not isinstance(train_data.sample_size,
                                                                        int) else train_data.sample_size

                validation_data_iter = iter(validation_dataloader)

                for idx, validation_batch in enumerate(validation_data_iter):
                    plucker_embedding = validation_batch['plucker_embedding'].to(device=unet.device)
                    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
                    sample = validation_pipeline(
                        prompt=validation_batch['text'],
                        pose_embedding=plucker_embedding,
                        video_length=video_length,
                        height=height,
                        width=width,
                        num_inference_steps=25,
                        guidance_scale=8.,
                        generator=generator,
                    ).videos[0]  # [3 f h w]
                    sample_gt = torch.cat([sample, (validation_batch['pixel_values'][0].permute(1, 0, 2, 3) + 1.0) / 2.0], dim=2)  # [3, f, 2h, w]
                    if 'clip_name' in validation_batch:
                        save_path = f"{output_dir}/samples/sample-{global_step}/{validation_batch['clip_name'][0]}.gif"
                    else:
                        save_path = f"{output_dir}/samples/sample-{global_step}/{idx}.gif"
                    save_videos_grid(sample_gt[None, ...], save_path)
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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, port=args.port, **config)
