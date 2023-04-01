import roboverse
import roboverse.bullet as bullet
from roboverse.policies import policies
from PIL import Image

import skvideo.io
import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm

ROBOT_VIEW_HEIGHT = 100
ROBOT_VIEW_WIDTH = 100
ROBOT_VIEW_CROP_X = 30


class BulletVideoLogger:
    def __init__(self, args, policy_kwargs={}):
        self.env_name = args.env
        self.num_timesteps_per_traj = args.num_timesteps
        self.accept_trajectory_key = args.accept_trajectory_key
        # self.noise = args.noise
        self.video_save_dir = args.video_save_dir
        self.save_all = args.save_all
        self.save_all_trajs_at_once = args.save_all_trajs_at_once
        save_mode_to_fn_map = {
            "png": self.save_images,
            "mp4": self.save_video,
            "gif": self.save_gif,
        }
        assert args.save_mode in save_mode_to_fn_map
        self.save_function = save_mode_to_fn_map[args.save_mode]

        self.image_size = 512
        self.add_robot_view = args.add_robot_view
        self.add_zoomed_view = args.add_zoomed_view

        if not os.path.exists(self.video_save_dir):
            os.makedirs(self.video_save_dir)
        # camera settings (default)
        # self.camera_target_pos = [0.57, 0.2, -0.22]
        # self.camera_roll = 0.0
        # self.camera_pitch = -10
        # self.camera_yaw = 215
        # self.camera_distance = 0.4

        # drawer cam (front view)
        # self.camera_target_pos = [0.60, 0.05, -0.30]
        # self.camera_roll = 0.0
        # self.camera_pitch = -30.0
        # self.camera_yaw = 180.0
        # self.camera_distance = 0.50

        # drawer cam (canonical view)
        # self.camera_target_pos = [0.55, 0., -0.30]
        # self.camera_roll = 0.0
        # self.camera_pitch = -30.0
        # self.camera_yaw = 150.0
        # self.camera_distance = 0.64

        # Old camera setting
        # self.camera_target_pos = [0.35, 0.4, -0.1]
        self.camera_target_pos = [0.55, 0.4, -0.15]
        self.camera_roll = 0.0
        self.camera_pitch = -30.0
        self.camera_yaw = 200  # 230.0
        self.camera_distance = 0.25  # 0.15

        if self.add_zoomed_view:
            # coordinates of original image to blow up here
            # assums square image size.
            self.r_coords = self.image_size * np.array([0.5, 0.95]) # rows
            self.c_coords = self.image_size * np.array([0.2, 0.8]) # cols

        self.view_matrix_args = dict(target_pos=self.camera_target_pos,
                                     distance=self.camera_distance,
                                     yaw=self.camera_yaw,
                                     pitch=self.camera_pitch,
                                     roll=self.camera_roll,
                                     up_axis_index=2)
        self.view_matrix = bullet.get_view_matrix(
            **self.view_matrix_args)
        self.projection_matrix = bullet.get_projection_matrix(
            self.image_size, self.image_size)
        # end camera settings
        self.env = roboverse.make(
            self.env_name, gui=False,
            transpose_image=False, num_tasks=args.num_tasks)
        assert args.policy_name in policies.keys()
        self.policy_name = args.policy_name
        policy_class = policies[self.policy_name]
        self.scripted_policy = policy_class(self.env, **policy_kwargs)
        self.trajectories_collected = 0

    def add_robot_view_to_video(self, images):
        image_x, image_y, image_c = images[0].shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(len(images)):
            robot_view_margin = 5
            robot_view = cv2.resize(images[i],
                                    (ROBOT_VIEW_HEIGHT, ROBOT_VIEW_WIDTH))
            robot_view = robot_view[ROBOT_VIEW_CROP_X:, :, :]
            image_new = np.copy(images[i])
            x_offset = ROBOT_VIEW_HEIGHT-ROBOT_VIEW_CROP_X
            y_offset = image_y - ROBOT_VIEW_WIDTH

            # Draw a background black rectangle
            image_new = cv2.rectangle(image_new, (self.image_size, 0),
                                      (y_offset - 2 * robot_view_margin,
                                      x_offset + 25 + robot_view_margin),
                                      (0, 0, 0), -1)

            image_new[robot_view_margin:x_offset + robot_view_margin,
                      y_offset - robot_view_margin:-robot_view_margin,
                      :] = robot_view
            image_new = cv2.putText(image_new, 'Robot View',
                                    (y_offset - robot_view_margin,
                                     x_offset + 18 + robot_view_margin),
                                    font, 0.55, (255, 255, 255), 1,
                                    cv2.LINE_AA)
            images[i] = image_new

        return images

    def add_zoomed_view_to_video(self, images):
        for i in range(len(images)):
            old_image = np.copy(images[i])

            zr_r_lo, zr_r_hi = int(self.r_coords[0]), int(self.r_coords[1])
            zr_c_lo, zr_c_hi = int(self.c_coords[0]), int(self.c_coords[1])
            zoom_region = old_image[zr_r_lo:zr_r_hi, zr_c_lo:zr_c_hi]
            new_w = self.image_size
            scaling_factor = new_w / zoom_region.shape[1]
            new_h = int(np.ceil(scaling_factor * zoom_region.shape[0]))
            zoomed_view = cv2.resize(zoom_region, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # interpolation method recommended by:
            # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=resize#resize
            image_new = np.concatenate([old_image, zoomed_view], axis=0)

            text_margin = int(0.05 * self.image_size)
            font = cv2.FONT_HERSHEY_SIMPLEX
            x_offset, y_offset = int(0.07 * self.image_size), int(1.05 * self.image_size)
            image_new = cv2.putText(image_new, 'Zoomed-In View',
                                    (x_offset + text_margin,
                                     y_offset + text_margin),
                                    font, 1.5, (0, 100, 255), 2,
                                    cv2.LINE_AA)

            images[i] = image_new

        return images

    def accept_criterion(self, info, obs_dict):
        c1 = self.save_all

        gripper_open = bool(obs_dict["state"][-1] == 1.0)
        c2 = (
            info[self.accept_trajectory_key] and
            gripper_open and
            info['num_objs_placed'] == 1)
        return c1 or c2

    def collect_traj_and_save_video(self, path_idx, task_idx=None):
        images = []
        if task_idx is not None:
            self.env.reset_task(task_idx)
        self.env.reset()
        self.scripted_policy.reset()
        for t in range(self.num_timesteps_per_traj):
            img, depth, segmentation = bullet.render(
                self.image_size, self.image_size,
                self.view_matrix, self.projection_matrix)
            images.append(img)
            action, _ = self.scripted_policy.get_action()
            obs, rew, done, info = self.env.step(action)

        if not self.accept_criterion(info, obs):
            # Traj didn't meet criteria for saving.
            return []

        return images

    def save_video(self, images, path_idx, task_idx=None):
        # Save Video
        save_path = self.get_save_path_wo_ext(task_idx, path_idx)
        save_path = f"{save_path}.mp4"
        if self.add_robot_view:
            dot_idx = save_path.index(".")
            save_path = save_path[:dot_idx] + "_with_robot_view" + \
                save_path[dot_idx:]
        save_dir = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("save_path", save_path)
        inputdict = {'-r': str(12)}
        outputdict = {'-vcodec': 'libx264', '-pix_fmt': 'yuv420p'}
        writer = skvideo.io.FFmpegWriter(
            save_path, inputdict=inputdict, outputdict=outputdict)

        if self.add_zoomed_view:
            images = self.add_zoomed_view_to_video(images)

        if self.add_robot_view:
            images = self.add_robot_view_to_video(images)

        for i in range(len(images)):
            writer.writeFrame(images[i])
        writer.close()

    def save_images(self, images, path_idx, task_idx=None):
        # Save Video
        save_path = self.get_save_path_wo_ext(task_idx, path_idx)
        if self.add_robot_view:
            save_path += "_with_robot_view"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("save_path", save_path)

        if self.add_robot_view:
            self.add_robot_view_to_video(images)
        for i in range(len(images)):
            im = Image.fromarray(images[i])
            im.save(os.path.join(save_path, '{}.png'.format(i)))

    def save_gif(self, images, path_idx, task_idx=None):
        save_path = self.get_save_path_wo_ext(task_idx, path_idx)
        save_path = f"{save_path}.gif"
        if self.add_robot_view:
            dot_idx = save_path.index(".")
            save_path = save_path[:dot_idx] + "_with_robot_view" + \
                save_path[dot_idx:]
        save_dir = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("save_path", save_path)

        if self.add_robot_view:
            self.add_robot_view_to_video(images)

        pil_images = []
        for image in images:
            pil_image = Image.fromarray(image)
            pil_images.append(pil_image)

        pil_images[0].save(
            save_path, format='GIF',
            save_all=True, append_images=pil_images[1:],
            optimize=True, duration=60, loop=0)
        # optimize=True has no effect on grainy-ness.

    def get_save_path_wo_ext(self, task_idx, path_idx):
        task_id_str = ""
        if task_idx is not None:
            task_id_str = f"taskid_{task_idx}/"
        save_path_wo_ext = (
            f"{self.video_save_dir}/"
            f"{self.env_name}_scripted_{self.policy_name}/"
            f"{task_id_str}{path_idx}")
        return save_path_wo_ext

    def run(self, num_videos_per_task, task_idxs):
        for j, task_idx in tqdm(enumerate(task_idxs)):
            i = 0
            images_multi_trajs = []
            while i < num_videos_per_task:
                images = self.collect_traj_and_save_video(i, task_idx)
                if len(images) > 0:
                    if self.save_all_trajs_at_once:
                        images_multi_trajs.extend(images)
                    else:
                        self.save_function(images, i, task_idx)
                    i += 1

                print(
                    f"On Task {j+1}/{len(task_idxs)}: "
                    f"saved {i}/{num_videos_per_task} so far")

            if self.save_all_trajs_at_once:
                self.save_function(images_multi_trajs, 0, task_idx)


def get_task_idx_list(args):
    def create_task_indices_from_task_int_str_list(task_interval_str_list, num_tasks):
        task_idx_interval_list = []
        for interval in task_interval_str_list:
            interval = tuple([int(x) for x in interval.split("-")])
            assert len(interval) == 2

            if len(task_idx_interval_list) >= 1:
                # Make sure most recently added interval's endpoint is smaller than
                # current interval's startpoint.
                assert task_idx_interval_list[-1][-1] < interval[0]

            task_idx_interval_list.append(interval)

        task_indices = [] # to collect_data on
        for interval in task_idx_interval_list:
            start, end = interval
            assert 0 <= start <= end <= num_tasks
            task_indices.extend(list(range(start, end + 1)))

        return task_indices

    # Either have --task-idx-intervals or --task-idxs
    assert (len(args.task_idx_intervals) > 0) != (len(args.task_idxs) > 0)

    if len(args.task_idx_intervals) > 0:
        task_idxs = create_task_indices_from_task_int_str_list(
            args.task_idx_intervals, args.num_tasks)
    elif len(args.task_idxs) > 0:
        task_idxs = list(args.task_idxs)
    else:
        raise NotImplementedError
    print("task_idxs", task_idxs)
    return task_idxs


def log_videos(args, task_idxs):
    policy_kwargs = {
        "drop_container_scheme": "target",
        "pick_object_scheme": "target",
    }
    vid_log = BulletVideoLogger(args, policy_kwargs=policy_kwargs)
    vid_log.run(args.num_videos_per_task, task_idxs)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str)
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("-d", "--video-save-dir", type=str,
                        default="data/scripted_rollouts")
    parser.add_argument("-n", "--num-videos-per-task", type=int, default=1)
    parser.add_argument("--add-robot-view", action="store_true", default=False)
    parser.add_argument("--add-zoomed-view", action="store_true", default=False)
    parser.add_argument("--save-mode", default="mp4", choices=["png", "mp4", "gif"])
    parser.add_argument("--save-all", action="store_true", default=False)
    # parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--task-idx-intervals", type=str, nargs='+', default=[])
    parser.add_argument("--task-idxs", type=int, nargs='+', default=[])
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--save-all-trajs-at-once", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    task_idxs = get_task_idx_list(args)
    log_videos(args, task_idxs)
