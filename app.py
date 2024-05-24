import spaces
import argparse
import torch
import tempfile
import os
import cv2

import numpy as np
import gradio as gr
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl


from omegaconf import OmegaConf
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from inference_cameractrl import get_relative_pose, ray_condition, get_pipeline
from cameractrl.utils.util import save_videos_grid

cv2.setNumThreads(1)
mpl.use('agg')

#### Description ####
title = r"""<h1 align="center">CameraCtrl: Enabling Camera Control for Video Diffusion Models</h1>"""
subtitle = r"""<h2 align="center">CameraCtrl Image2Video with <a href='https://arxiv.org/abs/2311.15127' target='_blank'> <b>Stable Video Diffusion (SVD)</b> </a>-xt <a href='https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt' target='_blank'> <b> model </b> </a> </h2>"""
description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/hehao13/CameraCtrl' target='_blank'><b>CameraCtrl: Enabling Camera Control for Video Diffusion Models</b></a>.<br>
CameraCtrl is capable of precisely controlling the camera trajectory during the video generation process.<br>
Note that, with SVD-xt, CameraCtrl only support Image2Video now.<br>
"""

closing_words = r"""

---

If you are interested in this demo or CameraCtrl is helpful for you, please give us a ⭐ of the <a href='https://github.com/hehao13/CameraCtrl' target='_blank'> CameraCtrl</a> Github Repo ! 
[![GitHub Stars](https://img.shields.io/github/stars/hehao13/CameraCtrl
)](https://github.com/hehao13/CameraCtrl)

---

📝 **Citation**
<br>
If you find our paper or code is useful for your research, please consider citing:
```bibtex
@article{he2024cameractrl,
      title={CameraCtrl: Enabling Camera Control for Text-to-Video Generation}, 
      author={Hao He and Yinghao Xu and Yuwei Guo and Gordon Wetzstein and Bo Dai and Hongsheng Li and Ceyuan Yang},
      journal={arXiv preprint arXiv:2404.02101},
      year={2024}
}
```

📧 **Contact**
<br>
If you have any questions, please feel free to contact me at <b>haohe@link.cuhk.edu.hk</b>.

**Acknowledgement**
<br>
We thank <a href='https://wzhouxiff.github.io/projects/MotionCtrl/' target='_blank'><b>MotionCtrl</b></a> and <a href='https://huggingface.co/spaces/lllyasviel/IC-Light' target='_blank'><b>IC-Light</b></a> for their gradio codes.<br>
"""


RESIZE_MODES = ['Resize then Center Crop', 'Directly resize']
CAMERA_TRAJECTORY_MODES = ["Provided Camera Trajectories", "Custom Camera Trajectories"]
height = 320
width = 576
num_frames = 25
device = "cuda" if torch.cuda.is_available() else "cpu"

config = "configs/train_cameractrl/svdxt_320_576_cameractrl.yaml"
model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
ckpt = "checkpoints/CameraCtrl_svdxt.ckpt"
if not os.path.exists(ckpt):
    os.makedirs("checkpoints", exist_ok=True)
    os.system("wget -c https://huggingface.co/hehao13/CameraCtrl_SVD_ckpts/resolve/main/CameraCtrl_svdxt.ckpt?download=true")
    os.system("mv CameraCtrl_svdxt.ckpt?download=true checkpoints/CameraCtrl_svdxt.ckpt")
model_config = OmegaConf.load(config)


pipeline = get_pipeline(model_id, "unet", model_config['down_block_types'], model_config['up_block_types'],
                        model_config['pose_encoder_kwargs'], model_config['attention_processor_kwargs'],
                        ckpt, True, device)


examples = [
    [
        "assets/example_condition_images/A_tiny_finch_on_a_branch_with_spring_flowers_on_background..png",
        "assets/pose_files/0bf152ef84195293_svdxt.txt",
        "Trajectory 1"
    ],
    [
        "assets/example_condition_images/A_beautiful_fluffy_domestic_hen_sitting_on_white_eggs_in_a_brown_nest,_eggs_are_under_the_hen..png",
        "assets/pose_files/0c9b371cc6225682_svdxt.txt",
        "Trajectory 2"
    ],
    [
        "assets/example_condition_images/Rocky_coastline_with_crashing_waves..png",
        "assets/pose_files/0c11dbe781b1c11c_svdxt.txt",
        "Trajectory 3"
    ],
    [
        "assets/example_condition_images/A_lion_standing_on_a_surfboard_in_the_ocean..png",
        "assets/pose_files/0f47577ab3441480_svdxt.txt",
        "Trajectory 4"
    ],
    [
        "assets/example_condition_images/An_exploding_cheese_house..png",
        "assets/pose_files/0f47577ab3441480_svdxt.txt",
        "Trajectory 4"
    ],
    [
        "assets/example_condition_images/Dolphins_leaping_out_of_the_ocean_at_sunset..png",
        "assets/pose_files/0f68374b76390082_svdxt.txt",
        "Trajectory 5"
    ],
    [
        "assets/example_condition_images/Leaves_are_falling_from_trees..png",
        "assets/pose_files/2c80f9eb0d3b2bb4_svdxt.txt",
        "Trajectory 6"
    ],
    [
        "assets/example_condition_images/A_serene_mountain_lake_at_sunrise,_with_mist_hovering_over_the_water..png",
        "assets/pose_files/2f25826f0d0ef09a_svdxt.txt",
        "Trajectory 7"
    ],
    [
        "assets/example_condition_images/Fireworks_display_illuminating_the_night_sky..png",
        "assets/pose_files/3f79dc32d575bcdc_svdxt.txt",
        "Trajectory 8"
    ],
    [
        "assets/example_condition_images/A_car_running_on_Mars..png",
        "assets/pose_files/4a2d6753676df096_svdxt.txt",
        "Trajectory 9"
    ],
]


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


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9 / 16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                  [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
                   vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical',
                          label='Frame Indexes')

    def show(self):
        plt.title('Camera Trajectory')
        plt.show()


def get_c2w(w2cs):
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    camera_positions = np.asarray([c2w[:3, 3] for c2w in ret_poses])  # [n_frame, 3]
    position_distances = [camera_positions[i] - camera_positions[i - 1] for i in range(1, len(camera_positions))]
    xyz_max = np.max(camera_positions, axis=0)
    xyz_min = np.min(camera_positions, axis=0)
    xyz_ranges = xyz_max - xyz_min  # [3, ]
    max_range = np.max(xyz_ranges)
    expected_xyz_ranges = 1
    scale_ratio = expected_xyz_ranges / max_range
    scaled_position_distances = [dis * scale_ratio for dis in position_distances]  # [n_frame - 1]
    scaled_camera_positions = [camera_positions[0], ]
    scaled_camera_positions.extend([camera_positions[0] + np.sum(np.asarray(scaled_position_distances[:i]), axis=0)
                                    for i in range(1, len(camera_positions))])
    ret_poses = [np.concatenate(
        (np.concatenate((ori_pose[:3, :3], cam_position[:, None]), axis=1), np.asarray([0, 0, 0, 1])[None]), axis=0)
                 for ori_pose, cam_position in zip(ret_poses, scaled_camera_positions)]
    transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)


def visualize_trajectory(trajectory_file):
    with open(trajectory_file, 'r') as f:
        poses = f.readlines()
    w2cs = [np.asarray([float(p) for p in pose.strip().split(' ')[7:]]).reshape(3, 4) for pose in poses[1:]]
    num_frames = len(w2cs)
    last_row = np.zeros((1, 4))
    last_row[0, -1] = 1.0
    w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
    c2ws = get_c2w(w2cs)
    visualizer = CameraPoseVisualizer([-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2])
    for frame_idx, c2w in enumerate(c2ws):
        visualizer.extrinsic2pyramid(c2w, frame_idx / num_frames, hw_ratio=9 / 16, base_xval=0.02, zval=0.1)
    visualizer.colorbar(num_frames)
    return visualizer.fig


vis_traj = visualize_trajectory('assets/pose_files/0bf152ef84195293_svdxt.txt')


@torch.inference_mode()
def process_input_image(input_image, resize_mode):
    global height, width
    expected_hw_ratio = height / width
    inp_w, inp_h = input_image.size
    inp_hw_ratio = inp_h / inp_w

    if inp_hw_ratio > expected_hw_ratio:
        resized_height = inp_hw_ratio * width
        resized_width = width
    else:
        resized_height = height
        resized_width = height / inp_hw_ratio
    resized_image = F.resize(input_image, size=[resized_height, resized_width])

    if resize_mode == RESIZE_MODES[0]:
        return_image = F.center_crop(resized_image, output_size=[height, width])
    else:
        return_image = resized_image

    return gr.update(visible=True, value=return_image, height=height, width=width), gr.update(visible=True), gr.update(
        visible=True), gr.update(visible=True), gr.update(visible=True)


def update_camera_trajectories(trajectory_mode):
    if trajectory_mode == CAMERA_TRAJECTORY_MODES[0]:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
               gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    elif trajectory_mode == CAMERA_TRAJECTORY_MODES[1]:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
               gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


def update_camera_args(trajectory_mode, provided_camera_trajectory, customized_trajectory_file):
    if trajectory_mode == CAMERA_TRAJECTORY_MODES[0]:
        res = "Provided " + str(provided_camera_trajectory)
    else:
        if customized_trajectory_file is None:
            res = " "
        else:
            res = f"Customized trajectory file {customized_trajectory_file.name.split('/')[-1]}"
    return res


def update_camera_args_reset():
    return " "


def update_trajectory_vis_plot(camera_trajectory_args, provided_camera_trajectory, customized_trajectory_file):
    if 'Provided' in camera_trajectory_args:
        if provided_camera_trajectory == "Trajectory 1":
            trajectory_file_path = "assets/pose_files/0bf152ef84195293_svdxt.txt"
        elif provided_camera_trajectory == "Trajectory 2":
            trajectory_file_path = "assets/pose_files/0c9b371cc6225682_svdxt.txt"
        elif provided_camera_trajectory == "Trajectory 3":
            trajectory_file_path = "assets/pose_files/0c11dbe781b1c11c_svdxt.txt"
        elif provided_camera_trajectory == "Trajectory 4":
            trajectory_file_path = "assets/pose_files/0f47577ab3441480_svdxt.txt"
        elif provided_camera_trajectory == "Trajectory 5":
            trajectory_file_path = "assets/pose_files/0f68374b76390082_svdxt.txt"
        elif provided_camera_trajectory == "Trajectory 6":
            trajectory_file_path = "assets/pose_files/2c80f9eb0d3b2bb4_svdxt.txt"
        elif provided_camera_trajectory == "Trajectory 7":
            trajectory_file_path = "assets/pose_files/2f25826f0d0ef09a_svdxt.txt"
        elif provided_camera_trajectory == "Trajectory 8":
            trajectory_file_path = "assets/pose_files/3f79dc32d575bcdc_svdxt.txt"
        else:
            trajectory_file_path = "assets/pose_files/4a2d6753676df096_svdxt.txt"
    else:
        trajectory_file_path = customized_trajectory_file.name
    vis_traj = visualize_trajectory(trajectory_file_path)
    return gr.update(visible=True), vis_traj, gr.update(visible=True), gr.update(visible=True), \
           gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
           gr.update(visible=True), gr.update(visible=True), trajectory_file_path


def update_set_button():
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


def update_buttons_for_example(example_image, example_traj_path, provided_traj_name):
    global height, width
    return_image = example_image
    return gr.update(visible=True, value=return_image, height=height, width=width), gr.update(visible=True), \
           gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
           gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), \
           gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), \
           gr.update(visible=True)

@spaces.GPU
@torch.inference_mode()
def sample_video(condition_image, trajectory_file, num_inference_step, min_guidance_scale, max_guidance_scale, fps_id, seed):
    global height, width, num_frames, device, pipeline
    with open(trajectory_file, 'r') as f:
        poses = f.readlines()
    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    cam_params = [Camera(cam_param) for cam_param in cam_params]
    sample_wh_ratio = width / height
    pose_wh_ratio = cam_params[0].fy / cam_params[0].fx
    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / height
    intrinsic = np.asarray([[cam_param.fx * width,
                             cam_param.fy * height,
                             cam_param.cx * width,
                             cam_param.cy * height]
                            for cam_param in cam_params], dtype=np.float32)
    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params, zero_first_frame_scale=True)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, height, width, device='cpu')  # b f h w 6
    plucker_embedding = plucker_embedding.permute(0, 1, 4, 2, 3).contiguous().to(device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    with torch.no_grad():
        sample = pipeline(
            image=condition_image,
            pose_embedding=plucker_embedding,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_step,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            fps=fps_id,
            do_image_process=True,
            generator=generator,
            output_type='pt'
        ).frames[0].transpose(0, 1).cpu()

    temporal_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
    save_videos_grid(sample[None], temporal_video_path, rescale=False)

    return temporal_video_path


def main(args):
    demo = gr.Blocks().queue()
    with demo:
        gr.Markdown(title)
        gr.Markdown(subtitle)
        gr.Markdown(description)

        with gr.Column():
            # step1: Input condition image
            step1_title = gr.Markdown("---\n## Step 1: Input an Image", show_label=False, visible=True)
            step1_dec = gr.Markdown(f"\n 1. Upload an Image by `Drag` or Click `Upload Image`; \
                                                \n 2. Click `{RESIZE_MODES[0]}` or `{RESIZE_MODES[1]}` to select the image resize mode. \
                                                \n - `{RESIZE_MODES[0]}`: First resize the input image, then center crop it into the resolution of 320 x 576. \
                                                \n - `{RESIZE_MODES[1]}`: Only resize the input image, and keep the original aspect ratio.",
                                    show_label=False, visible=True)
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    input_image = gr.Image(type='pil', interactive=True, elem_id='condition_image',
                                           elem_classes='image',
                                           visible=True)
                    with gr.Row():
                        resize_crop_button = gr.Button(RESIZE_MODES[0], visible=True)
                        directly_resize_button = gr.Button(RESIZE_MODES[1], visible=True)
                with gr.Column(scale=2):
                    processed_image = gr.Image(type='pil', interactive=False, elem_id='processed_image',
                                               elem_classes='image', visible=False)

            # step2: Select camera trajectory
            step2_camera_trajectory = gr.Markdown("---\n## Step 2: Select the camera trajectory", show_label=False,
                                                  visible=False)
            step2_camera_trajectory_des = gr.Markdown(f"\n - `{CAMERA_TRAJECTORY_MODES[0]}`: Including 9 camera trajectories extracted from the test set of RealEstate10K dataset, each has 25 frames. \
                                                                \n - `{CAMERA_TRAJECTORY_MODES[1]}`: You can provide the customized camera trajectories in the txt file.",
                                                      show_label=False, visible=False)
            with gr.Row(equal_height=True):
                provide_trajectory_button = gr.Button(CAMERA_TRAJECTORY_MODES[0], visible=False)
                customized_trajectory_button = gr.Button(CAMERA_TRAJECTORY_MODES[1], visible=False)
            with gr.Row():
                with gr.Column():
                    provided_camera_trajectory = gr.Markdown(f"---\n### {CAMERA_TRAJECTORY_MODES[0]}", show_label=False,
                                                             visible=False)
                    provided_camera_trajectory_des = gr.Markdown(f"\n 1. Click one of the provide camera trajectories, such as `Trajectory 1`; \
                                                                   \n 2. Click `Visualize Trajectory` to visualize the camera trajectory; \
                                                                   \n 3. Click `Reset Trajectory` to reset the camera trajectory. ",
                                                                 show_label=False, visible=False)

                    customized_camera_trajectory = gr.Markdown(f"---\n### {CAMERA_TRAJECTORY_MODES[1]}",
                                                               show_label=False,
                                                               visible=False)
                    customized_run_status = gr.Markdown(f"\n 1. Input the txt file containing camera trajectory. \
                                                    \n 2. Click `Visualize Trajectory` to visualize the camera trajectory; \
                                                    \n 3. Click `Reset Trajectory` to reset the camera trajectory. ",
                                                        show_label=False, visible=False)

                    with gr.Row():
                        provided_trajectories = gr.Dropdown(
                            ["Trajectory 1", "Trajectory 2", "Trajectory 3", "Trajectory 4", "Trajectory 5",
                             "Trajectory 6", "Trajectory 7", "Trajectory 8", "Trajectory 9"],
                            label="Provided Trajectories", interactive=True, visible=False)
                    with gr.Row():
                        customized_camera_trajectory_file = gr.File(
                            label="Upload customized camera trajectory (in .txt format).", visible=False, interactive=True)

                    with gr.Row():
                        camera_args = gr.Textbox(value=" ", label="Camera Trajectory Name", visible=False)
                        camera_trajectory_path = gr.Textbox(value=" ", visible=False)

                    with gr.Row():
                        camera_trajectory_vis = gr.Button(value="Visualize Camera Trajectory", visible=False)
                        camera_trajectory_reset = gr.Button(value="Reset Camera Trajectory", visible=False)
                with gr.Column():
                    vis_camera_trajectory = gr.Plot(vis_traj, label='Camera Trajectory', visible=False)

            # step3: Set inference parameters
            with gr.Row():
                with gr.Column():
                    step3_title = gr.Markdown(f"---\n## Step3: Setting the inference hyper-parameters.", visible=False)
                    step3_des = gr.Markdown(
                        f"\n 1. Set the mumber of inference step; \
                         \n 2. Set the seed; \
                         \n 3. Set the minimum guidance scale and the maximum guidance scale; \
                         \n 4. Set the fps; \
                          \n - Please refer to the SVD paper for the meaning of the last three parameter",
                        visible=False)
                    with gr.Row():
                        with gr.Column():
                            num_inference_steps = gr.Number(value=25, label='Number Inference Steps', step=1, interactive=True,
                                                            visible=False)
                        with gr.Column():
                            seed = gr.Number(value=42, label='Seed', minimum=1, interactive=True, visible=False, step=1)
                        with gr.Column():
                            min_guidance_scale = gr.Number(value=1.0, label='Minimum Guidance Scale', minimum=1.0, step=0.5,
                                                           interactive=True, visible=False)
                        with gr.Column():
                            max_guidance_scale = gr.Number(value=3.0, label='Maximum Guidance Scale', minimum=1.0, step=0.5,
                                                           interactive=True, visible=False)
                        with gr.Column():
                            fps = gr.Number(value=7, label='FPS', minimum=1, step=1, interactive=True, visible=False)
                        with gr.Column():
                            _ = gr.Button("Seed", visible=False)
                        with gr.Column():
                            _ = gr.Button("Seed", visible=False)
                        with gr.Column():
                            _ = gr.Button("Seed", visible=False)
            with gr.Row():
                with gr.Column():
                    _ = gr.Button("Set", visible=False)
                with gr.Column():
                    set_button = gr.Button("Set", visible=False)
                with gr.Column():
                    _ = gr.Button("Set", visible=False)

            # step 4: Generate video
            with gr.Row():
                with gr.Column():
                    step4_title = gr.Markdown("---\n## Step4 Generating video", show_label=False, visible=False)
                    step4_des = gr.Markdown(f"\n - Click the `Start generation !` button to generate the video.; \
                    \n - If the content of generated video is not very aligned with the condition image, try to increase the `Minimum Guidance Scale` and `Maximum Guidance Scale`. \
                                         \n - If the generated videos are distored, try to increase `FPS`.",
                                            visible=False)
                    start_button = gr.Button(value="Start generation !", visible=False)
                with gr.Column():
                    generate_video = gr.Video(value=None, label="Generate Video", visible=False)
        resize_crop_button.click(fn=process_input_image, inputs=[input_image, resize_crop_button],
                                 outputs=[processed_image, step2_camera_trajectory, step2_camera_trajectory_des,
                                          provide_trajectory_button, customized_trajectory_button])
        directly_resize_button.click(fn=process_input_image, inputs=[input_image, directly_resize_button],
                                     outputs=[processed_image, step2_camera_trajectory, step2_camera_trajectory_des,
                                              provide_trajectory_button, customized_trajectory_button])
        provide_trajectory_button.click(fn=update_camera_trajectories, inputs=[provide_trajectory_button],
                                        outputs=[provided_camera_trajectory, provided_camera_trajectory_des,
                                                 provided_trajectories,
                                                 customized_camera_trajectory, customized_run_status,
                                                 customized_camera_trajectory_file,
                                                 camera_args, camera_trajectory_vis, camera_trajectory_reset])
        customized_trajectory_button.click(fn=update_camera_trajectories, inputs=[customized_trajectory_button],
                                           outputs=[provided_camera_trajectory, provided_camera_trajectory_des,
                                                    provided_trajectories,
                                                    customized_camera_trajectory, customized_run_status,
                                                    customized_camera_trajectory_file,
                                                    camera_args, camera_trajectory_vis, camera_trajectory_reset])

        provided_trajectories.change(fn=update_camera_args, inputs=[provide_trajectory_button, provided_trajectories, customized_camera_trajectory_file],
                                     outputs=[camera_args])
        customized_camera_trajectory_file.change(fn=update_camera_args, inputs=[customized_trajectory_button, provided_trajectories, customized_camera_trajectory_file],
                                                 outputs=[camera_args])
        camera_trajectory_reset.click(fn=update_camera_args_reset, inputs=None, outputs=[camera_args])
        camera_trajectory_vis.click(fn=update_trajectory_vis_plot, inputs=[camera_args, provided_trajectories, customized_camera_trajectory_file],
                                    outputs=[vis_camera_trajectory, vis_camera_trajectory, step3_title, step3_des,
                                             num_inference_steps, min_guidance_scale, max_guidance_scale, fps,
                                             seed, set_button, camera_trajectory_path])
        set_button.click(fn=update_set_button, inputs=None, outputs=[step4_title, step4_des, start_button, generate_video])
        start_button.click(fn=sample_video, inputs=[processed_image, camera_trajectory_path, num_inference_steps,
                                                    min_guidance_scale, max_guidance_scale, fps, seed],
                           outputs=[generate_video])

        # set example
        gr.Markdown("## Examples")
        gr.Markdown("\n Choosing the one of the following examples to get a quick start, by selecting an example, "
                    "we will set the condition image and camera trajectory automatically. "
                    "Then, you can click the `Visualize Camera Trajectory` button to visualize the camera trajectory.")
        gr.Examples(
            fn=update_buttons_for_example,
            run_on_click=True,
            cache_examples=False,
            examples=examples,
            inputs=[input_image, camera_args, provided_trajectories],
            outputs=[processed_image, step2_camera_trajectory, step2_camera_trajectory_des, provide_trajectory_button,
                     customized_trajectory_button,
                     provided_camera_trajectory, provided_camera_trajectory_des, provided_trajectories,
                     customized_camera_trajectory, customized_run_status, customized_camera_trajectory_file,
                     camera_args, camera_trajectory_vis, camera_trajectory_reset]
        )
        with gr.Row():
            gr.Markdown(closing_words)

    demo.launch(**args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', default='0.0.0.0')
    parser.add_argument('--broswer', action='store_true')
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    launch_kwargs = {'server_name': args.listen,
                     'inbrowser': args.broswer,
                     'share': args.share}
    main(launch_kwargs)
