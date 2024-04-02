import argparse
import json
import os
import os.path as osp
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--caption_json', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--save_name', required=True)
    parser.add_argument('--video_suffix', default='.mp4')
    parser.add_argument('--video_folder', default='video_clips')
    parser.add_argument('--pose_suffix', default='.txt')
    parser.add_argument('--pose_folder', default='pose_files')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    save_root = args.root_path
    captions = json.load(open(args.caption_json, 'r'))
    captions = {k: v[0] for k, v in captions.items()}
    all_results = []
    for clip_path, caption in tqdm(captions.items()):
        clip_path = '/'.join(clip_path.split('/')[-2:])
        clip_name = clip_path.split('/')[-1].replace(args.video_suffix, '')
        clip_relative_path = osp.join(args.video_folder, clip_path)
        if not osp.exists(osp.join(save_root, clip_relative_path)):
            continue
        pose_file = args.pose_folder + '/' + clip_name + args.pose_suffix
        if not osp.exists(osp.join(save_root, pose_file)):
            continue

        all_results.append({"clip_name": clip_name, "clip_path": clip_relative_path,
                            "pose_file": pose_file, "caption": caption})
    print(f'There are {len(all_results)} clips after the processing')
    with open(osp.join(args.save_path, args.save_name), 'w') as f:
        json.dump(all_results, fp=f)
    print(f'Saved the generated json file to {osp.join(args.save_path, args.save_name)}')
