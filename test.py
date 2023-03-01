import glob
import json
import os

import numpy
import numpy as np
import cv2

import utils
from visual_odometry import PinholeCamera, VisualOdometry

# // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
# CALIB_PARAMS_YAML_4="" \
#                     "1333.0333496269382," \
#                     "1152.7953767273623," \
#                     "1362.2385792503271," \
#                     "705.0287442757885," \
#                     "-0.1697821786723204," \
#                     "0.08551140923416184," \
#                     "-0.0019370125115926041," \
#                     "-0.005935762788516092," \
#                     "-0.02775070400965532," \
#                     "0," \
#                     "0," \
#                     "0"

cam = PinholeCamera(
    width=2704,
    height=1538,
    fx=1333.03,
    fy=1152.79,
    cx=1362.23,
    cy=705.028,
)
vo = VisualOdometry(cam=cam)

traj = np.zeros((600, 600, 3), dtype=np.uint8)

BASE_PATH = '/home/gkiavash/Downloads/Master-Thesis-Structure-from-Motion/scripts/dataset_vo/*.jpg'
OUTPUT_PATH = "/home/gkiavash/Downloads/Master-Thesis-Structure-from-Motion/scripts/dataset_vo/out_cv.json"

images = glob.glob(BASE_PATH)
output = {}

for img_id, fname in enumerate(sorted(images)):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vo.update(img, img_id)

    cur_t = vo.cur_t
    cur_R = vo.cur_R
    if (img_id > 2):
        print("t", cur_t)
        print("r", cur_R)
        x, y, z = cur_t[0], cur_t[1], cur_t[2]

        cur_R = utils.rotationMatrixToEulerAngles(cur_R, is_radian=True)
        cur_R = utils.get_quaternion_from_euler(*cur_R)
        o_x, o_y, o_z, o_w = cur_R
    else:
        x, y, z = 0., 0., 0.
        o_x, o_y, o_z, o_w = 0., 0., 0., 0.

    draw_x, draw_y = int(x) + 290, int(z) + 90
    true_x, true_y = int(vo.trueX) + 290, int(vo.trueZ) + 90

    cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)
    cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
    cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)
    cv2.waitKey(1)

    # save trajectory:
    output[os.path.basename(fname)] = [
        o_w, o_x, o_y, o_z,
        x if type(x) is not numpy.ndarray else x[0],
        y if type(y) is not numpy.ndarray else y[0],
        z if type(z) is not numpy.ndarray else z[0]
    ]

cv2.imwrite('map.png', traj)

with open(OUTPUT_PATH, 'w') as outfile:
    outfile.write(utils.vo_to_colmap_images_txt(output))
