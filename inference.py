import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from packaging import version as pver

from cameractrl.pipelines.pipeline_animation import StableVideoDiffusionPipelinePoseCond
from cameractrl.models.unet import UNetSpatioTemporalConditionModelPoseCond
from cameractrl.models.pose_adaptor import CameraPoseEncoder
from cameractrl.utils.util import save_videos_grid


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_relative_pose(cam_params, zero_first_frame_scale):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    source_cam_c2w = abs_c2ws[0]
    if zero_first_frame_scale:
        cam_to_origin = 0
    else:
        cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    return plucker


def get_pipeline(ori_model_path, unet_subfolder, down_block_types, up_block_types, pose_encoder_kwargs,
                 attention_processor_kwargs, pose_adaptor_ckpt, enable_xformers, device):
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(ori_model_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(ori_model_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(ori_model_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(ori_model_path, subfolder="vae")
    unet = UNetSpatioTemporalConditionModelPoseCond.from_pretrained(ori_model_path,
                                                                    subfolder=unet_subfolder,
                                                                    down_block_types=down_block_types,
                                                                    up_block_types=up_block_types)
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)
    print("Setting the attention processors")
    unet.set_pose_cond_attn_processor(enable_xformers=(enable_xformers and is_xformers_available()), **attention_processor_kwargs)
    print(f"Loading weights of camera encoder and attention processor from {pose_adaptor_ckpt}")
    ckpt_dict = torch.load(pose_adaptor_ckpt, map_location=unet.device)
    pose_encoder_state_dict = ckpt_dict['pose_encoder_state_dict']
    pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
    assert len(pose_encoder_m) == 0 and len(pose_encoder_u) == 0
    attention_processor_state_dict = ckpt_dict['attention_processor_state_dict']
    _, attention_processor_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attention_processor_u) == 0
    print("Loading done")
    vae.to(device)
    image_encoder.to(device)
    unet.to(device)
    pipeline = StableVideoDiffusionPipelinePoseCond(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_encoder=pose_encoder
    )
    pipeline = pipeline.to(device)
    return pipeline


def main(args):
    os.makedirs(os.path.join(args.out_root, 'generated_videos'), exist_ok=True)
    os.makedirs(os.path.join(args.out_root, 'reference_images'), exist_ok=True)
    rank = args.local_rank
    setup_for_distributed(rank == 0)
    gpu_id = rank % torch.cuda.device_count()
    model_configs = OmegaConf.load(args.model_config)
    device = f"cuda:{gpu_id}"
    print(f'Constructing pipeline')
    pipeline = get_pipeline(args.ori_model_path, model_configs['unet_subfolder'], model_configs['down_block_types'],
                            model_configs['up_block_types'], model_configs['pose_encoder_kwargs'],
                            model_configs['attention_processor_kwargs'], args.pose_adaptor_ckpt, args.enable_xformers, device)
    print('Done')

    print('Loading K, R, t matrix')
    with open(args.trajectory_file, 'r') as f:
        poses = f.readlines()
    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = args.image_width / args.image_height
    pose_wh_ratio = args.original_pose_width / args.original_pose_height
    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = args.image_height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / args.image_width
    else:
        resized_ori_h = args.image_width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / args.image_height
    intrinsic = np.asarray([[cam_param.fx * args.image_width,
                             cam_param.fy * args.image_height,
                             cam_param.cx * args.image_width,
                             cam_param.cy * args.image_height]
                            for cam_param in cam_params], dtype=np.float32)
    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params, zero_first_frame_scale=True)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, args.image_height, args.image_width, device='cpu')       # b f h w 6
    plucker_embedding = plucker_embedding.permute(0, 1, 4, 2, 3).contiguous().to(device=device)

    prompt_dict = json.load(open(args.prompt_file, 'r'))
    prompt_images = prompt_dict['image_paths']
    prompt_captions = prompt_dict['captions']
    N = int(len(prompt_images) // args.n_procs)
    remainder = int(len(prompt_images) % args.n_procs)
    prompts_per_gpu = [N + 1 if gpu_id < remainder else N for gpu_id in range(args.n_procs)]
    low_idx = sum(prompts_per_gpu[:gpu_id])
    high_idx = low_idx + prompts_per_gpu[gpu_id]
    prompt_images = prompt_images[low_idx: high_idx]
    prompt_captions = prompt_captions[low_idx: high_idx]
    print(f"rank {rank} / {torch.cuda.device_count()}, number of prompts: {len(prompt_images)}")

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    for prompt_image, prompt_caption in tqdm(zip(prompt_images, prompt_captions)):
        save_name = "_".join(prompt_caption.split(" "))
        condition_image = Image.open(prompt_image)
        with torch.no_grad():
            sample = pipeline(
                image=condition_image,
                pose_embedding=plucker_embedding,
                height=args.image_height,
                width=args.image_width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                min_guidance_scale=args.min_guidance_scale,
                max_guidance_scale=args.max_guidance_scale,
                do_image_process=True,
                generator=generator,
                output_type='pt'
            ).frames[0].transpose(0, 1).cpu()      # [3, f, h, w] 0-1
        resized_condition_image = condition_image.resize((args.image_width, args.image_height))
        save_videos_grid(sample[None], f"{os.path.join(args.out_root, 'generated_videos')}/{save_name}.mp4", rescale=False)
        resized_condition_image.save(os.path.join(args.out_root, 'reference_images', f'{save_name}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str)
    parser.add_argument("--image_height", type=int, default=320)
    parser.add_argument("--image_width", type=int, default=576)
    parser.add_argument("--num_frames", type=int, default=14, help="14 for svd and 25 for svd-xt", choices=[14, 25])
    parser.add_argument("--ori_model_path", type=str)
    parser.add_argument("--unet_subfolder", type=str, default='unet')
    parser.add_argument("--enable_xformers", action='store_true')
    parser.add_argument("--pose_adaptor_ckpt", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--min_guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_guidance_scale", type=float, default=3.0)
    parser.add_argument("--prompt_file", required=True, help='prompts path, json or txt')
    parser.add_argument("--trajectory_file", required=True)
    parser.add_argument("--original_pose_width", type=int, default=1280)
    parser.add_argument("--original_pose_height", type=int, default=720)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--n_procs", type=int, default=8)

    # DDP args
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of the distributed processes.")
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Replica rank on the current node. This field is required '
                             'by `torch.distributed.launch`.')
    args = parser.parse_args()
    main(args)
