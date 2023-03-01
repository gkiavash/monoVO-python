import math

import numpy as np


def isRotationMatrix(R):
    """
    Checks if a matrix is a valid rotation matrix
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R, is_radian=True):
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    :param R:
    :param is_radian:
    :return:
    """
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    if is_radian:
        return np.array([x, y, z])
    else:
        pi = 22 / 7
        return np.array([
            x * (180 / pi),
            y * (180 / pi),
            z * (180 / pi),
        ])


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def vo_to_colmap_images_txt(vo_output):
    out_str = ""
    for ind, fname in enumerate(sorted(vo_output.keys())):
        values_str = [str(x) for x in vo_output[fname]]
        out_str += f"{ind+1} {' '.join(values_str)} {fname}\n"
        # print(out_str)
    return out_str


def test_equal_index_output_to_colmap():
    import json

    colmap_output = "/home/gkiavash/Downloads/sfm_projects/sfm_compare_4_pp_calib_3_prato/sfm_compare_pp_calib/sparse/model/images.txt"
    monovo_output = "/home/gkiavash/PycharmProjects/monoVO-python/dataset_vo/out_cv.json"

    def parse_images_txt(file_str):
        col_json = {}
        col_data = file_str.split("\n")
        for col_row in col_data:
            col_ind = col_row.split(" ")
            if len(col_ind) > 0:
                print(col_ind[0], col_ind[-1])
                col_json[col_ind[0]] = col_ind[-1]
        return col_json


    with open(colmap_output, 'r') as file_colmap_output, \
            open(monovo_output, 'r') as file_monovo_output:

        col_json = parse_images_txt(file_colmap_output.read())
        mvo_json = parse_images_txt(file_monovo_output.read())

        assert json.dumps(mvo_json, sort_keys=True) == json.dumps(col_json, sort_keys=True)
