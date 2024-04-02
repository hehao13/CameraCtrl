import argparse
import json
import random
import os
import os.path as osp
import imageio
import cv2
import numpy as np
from decord import VideoReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--clip_names', nargs='*', help='the txt file names')
    parser.add_argument('--clip_txt_path', required=True, help='root path of downloaded realestate10k txt files')
    parser.add_argument('--json_file', required=True, help='json file generated using generate_realestate_json.py')
    parser.add_argument('--trajectory_names', nargs='*', help='saving names for each trajectory')
    parser.add_argument('--sample_stride', type=int, default=4)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--video_width', type=int, default=384)
    parser.add_argument('--video_height', type=int, default=256)
    parser.add_argument('--save_images', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(osp.join(args.save_path, 'selected_pose_files'), exist_ok=True)
    os.makedirs(osp.join(args.save_path, 'selected_clips'), exist_ok=True)
    if args.save_images:
        os.makedirs(osp.join(args.save_path, 'selected_images'), exist_ok=True)
    clip_infos = json.load(open(args.json_file, 'r'))
    clip_name2clip_info = {x['clip_name']: x for x in clip_infos}
    clip_name2clip_info = {x: clip_name2clip_info[x] for x in args.clip_names}
    selected_clip_infos = []
    trajectory_names = args.clip_names if args.trajectory_names is None else args.trajectory_names
    for clip_info, trajectory_name in zip(clip_name2clip_info.values(), trajectory_names):
        pose_file = osp.join(args.clip_txt_path, clip_info['pose_file'])
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        html = poses[0].strip()
        poses = [x.strip() for x in poses[1:]]
        total_frames = len(poses)
        cropped_length = args.num_frames * args.sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)
        assert end_frame_ind - start_frame_ind >= args.num_frames
        frame_ind = np.linspace(start_frame_ind, end_frame_ind - 1, args.num_frames, dtype=int)
        poses = [html, ] + [poses[ind] for ind in frame_ind]
        pose_save_file = osp.join(args.save_path, 'selected_pose_files', trajectory_name + '.txt')
        with open(pose_save_file, 'w') as f:
            for pose in poses:
                f.write(pose + '\n')
        clip_file = osp.join(args.clip_txt_path, clip_info['clip_path'])
        video_reader = VideoReader(clip_file)
        video_batch = video_reader.get_batch(frame_ind).asnumpy()
        video_batch = [cv2.resize(x, dsize=(args.video_width, args.video_height)) for x in video_batch]
        clip_save_file = osp.join(args.save_path, 'selected_clips', trajectory_name + '.mp4')
        imageio.mimsave(clip_save_file, video_batch, fps=8)
        selected_clip_infos.append({'clip_name': clip_info['clip_name'], 'caption': clip_info['caption'],
                                    'clip_path': clip_save_file, 'pose_file': pose_save_file,
                                    'trajectory_name': trajectory_name})
        if args.save_images:
            images_save_path = osp.join(args.save_path, 'selected_images', trajectory_name)
            os.makedirs(images_save_path, exist_ok=True)
            for image_idx, image in zip(frame_ind, video_batch):
                image_save_path = osp.join(images_save_path, f'{image_idx}.jpg')
                cv2.imwrite(image_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            selected_clip_infos[-1].update({'images_save_path': images_save_path})
    with open(osp.join(args.save_path, 'selected_clip_infos.json'), 'w') as f:
        json.dump(selected_clip_infos, fp=f)
