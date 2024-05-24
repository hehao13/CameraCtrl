import argparse
import json
import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline


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


def get_pipeline(model_id, device):
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", local_files_only=True)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe = pipe.to(device)
    return pipe


def main(args):
    os.makedirs(args.out_root, exist_ok=True)
    rank = args.local_rank
    setup_for_distributed(rank == 0)
    gpu_id = rank % torch.cuda.device_count()
    device = f"cuda:{gpu_id}"
    print(f'Constructing pipeline')
    pipeline = get_pipeline(args.model_id, device)
    print('Done')

    if args.caption_file.endswith('.json'):
        prompts = json.load(open(args.caption_file, 'r'))['prompts']
    else:
        assert args.caption_file.endswith('txt')
        with open(args.caption_file, 'r') as f:
            prompts = f.readlines()
        prompts = [x.strip() for x in prompts]

    N = int(len(prompts) // args.n_procs)
    remainder = int(len(prompts) % args.n_procs)
    prompts_per_gpu = [N + 1 if gpu_id < remainder else N for gpu_id in range(args.n_procs)]
    low_idx = sum(prompts_per_gpu[:gpu_id])
    high_idx = low_idx + prompts_per_gpu[gpu_id]
    prompts = prompts[low_idx: high_idx]
    print(f"rank {rank} / {torch.cuda.device_count()}, number of prompts: {len(prompts)}")

    if args.seed_txt is not None:
        with open(args.seed_txt, 'r') as f:
            all_seeds = f.readlines()
        all_seeds = [x.strip() for x in all_seeds]
    else:
        all_seeds = [42, ]

    generator = torch.Generator(device=device)

    with torch.no_grad():
        for seed in all_seeds:
            os.makedirs(os.path.join(args.out_root, seed), exist_ok=True)
            generator.manual_seed(int(seed))
            for prompt in tqdm(prompts):
                save_name = "_".join(prompt.split(" "))
                sample = pipeline(
                    prompt=prompt,
                    height=args.image_height,
                    width=args.image_width,
                    # guidance=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator
                ).images[0]
                sample.save(f"{args.out_root}/{seed}/{save_name}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str)
    parser.add_argument("--image_height", type=int, default=320)
    parser.add_argument("--image_width", type=int, default=576)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--caption_file", required=True, help='prompts path, json or txt')
    parser.add_argument("--seed_txt", type=str)
    parser.add_argument("--n_procs", type=int, default=8)

    # DDP args
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of the distributed processes.")
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Replica rank on the current node. This field is required '
                             'by `torch.distributed.launch`.')
    args = parser.parse_args()
    main(args)
