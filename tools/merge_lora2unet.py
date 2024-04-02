import argparse
import torch
import os
import shutil
from diffusers.models import UNet2DConditionModel
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from safetensors.torch import save_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_scale', type=float, default=1.0)
    parser.add_argument('--lora_ckpt_path', type=str, required=True)
    parser.add_argument('--unet_ckpt_path', type=str, required=True, help='root path of the sd1.5 model')
    parser.add_argument('--save_path', type=str, required=True, help='args.unet_ckpt_path + a new subfolder name')
    parser.add_argument('--unet_config_path', type=str, required=True, help='path to unet config, in the `unet` subfolder of args.unet_ckpt_path')
    parser.add_argument('--lora_keys', nargs='*', type=str, default=['to_q', 'to_k', 'to_v', 'to_out'])
    parser.add_argument('--negative_lora_keys', type=str, default="bias")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    unet = UNet2DConditionModel.from_pretrained(args.unet_ckpt_path, subfolder='unet')
    fused_state_dict = unet.state_dict()

    print(f'Loading the lora weights from {args.lora_ckpt_path}')
    lora_state_dict = torch.load(args.lora_ckpt_path, map_location='cpu')
    if 'state_dict' in lora_state_dict:
        lora_state_dict = lora_state_dict['state_dict']
    print(f'Loading done')
    print(f'Fusing the lora weight to unet weight')
    used_lora_key = []
    for lora_key in args.lora_keys:
        unet_keys = [x for x in fused_state_dict.keys() if lora_key in x and args.negative_lora_keys not in x]
        print(f'There are {len(unet_keys)} unet keys for lora key: {lora_key}')
        for unet_key in unet_keys:
            prefixes = unet_key.split('.')
            idx = prefixes.index(lora_key)
            lora_down_key = ".".join(prefixes[:idx]) + f".processor.{lora_key}_lora.down" + f".{prefixes[-1]}"
            lora_up_key = ".".join(prefixes[:idx]) + f".processor.{lora_key}_lora.up" + f".{prefixes[-1]}"
            assert lora_down_key in lora_state_dict and lora_up_key in lora_state_dict
            print(f'Fusing lora weight for {unet_key}')
            fused_state_dict[unet_key] = fused_state_dict[unet_key] + torch.bmm(lora_state_dict[lora_up_key][None, ...], lora_state_dict[lora_down_key][None, ...])[0] * args.lora_scale
            used_lora_key.append(lora_down_key)
            used_lora_key.append(lora_up_key)
    assert len(set(used_lora_key) - set(lora_state_dict.keys())) == 0
    print(f'Fusing done')
    save_path = os.path.join(args.save_path, SAFETENSORS_WEIGHTS_NAME)
    print(f'Saving the fused state dict to {save_path}')
    save_file(fused_state_dict, save_path)
    config_dst_path = os.path.join(args.save_path, 'config.json')
    print(f'Copying the unet config to {config_dst_path}')
    shutil.copy(args.unet_config_path, config_dst_path)
    print('Done!')
