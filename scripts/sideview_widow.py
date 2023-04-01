import roboverse
import os
import argparse
import numpy as np
from PIL import Image
import roboverse.bullet as bullet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    num_tasks = 300
    camera_target_pos = [0.55, 0.4, -0.15]  # [0.35, 0.4, -0.1]
    camera_distance = 0.25  # 0.2
    camera_yaw, camera_pitch, camera_roll = 200, -30, 0  # 230, -30, 0  # Side view: 90, 0, 0
    img_h, img_w = 512, 512
    env = roboverse.make(
        'Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0',
        gui=False, transpose_image=False, num_tasks=num_tasks)

    for task_idx in range(2 * num_tasks):
        env.reset_task(task_idx)
        env.reset()
        obs, _, _, _ = env.step(np.array([0]*8))
        view_matrix_args = dict(target_pos=camera_target_pos,
                                distance=camera_distance,
                                yaw=camera_yaw,
                                pitch=camera_pitch,
                                roll=camera_roll,
                                up_axis_index=2)
        view_matrix_obs = bullet.get_view_matrix(**view_matrix_args)
        projection_matrix_obs = bullet.get_projection_matrix(
            img_h, img_w)
        img, _, _ = bullet.render(
            img_h, img_w, view_matrix_obs,
            projection_matrix_obs, shadow=0)
        # print("img", img)
        im = Image.fromarray(img)
        out_im_path = os.path.join(args.save_path, '{}.png'.format(task_idx))
        im.save(out_im_path)
        print(f"saved image {out_im_path}")
