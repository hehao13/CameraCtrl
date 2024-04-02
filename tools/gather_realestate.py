import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', required=True, help='Path to the down loaded realestate10k txt files')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--save_name', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    all_txts = os.listdir(args.video_folder)
    print(f'There are {len(all_txts)} video clips in the folder {args.video_folder}')
    video_paths = defaultdict(list)
    for txt in tqdm(all_txts):
        with open(osp.join(args.video_folder, txt), 'r') as f:
            lines = f.readlines()
        video_name = lines[0].strip().split('=')[-1]
        video_paths[video_name].append(txt.split('.')[0])
    print(f'There are {len(video_paths)} videos in the folder {args.video_folder}')
    with open(osp.join(args.save_path, args.save_name), 'w') as f:
        json.dump(video_paths, fp=f)
