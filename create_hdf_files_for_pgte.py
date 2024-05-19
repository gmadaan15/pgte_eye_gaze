"""
Copyright 2019 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os

import cv2 as cv
import eos
import h5py
import numpy as np

face_model_3d_coordinates = None

full_face_model_3d_coordinates = None

normalized_camera = {  # Face for ST-ED
    'focal_length': 500,
    'distance': 600,
    'size': (128, 128),
}
# for eye either right or left
normalized_camera_eye = {
    'focal_length': 1300,
    'distance': 600,
    'size': (128, 128),
}

# camera metrices
norm_camera_matrix = np.array(
    [
        [normalized_camera['focal_length'], 0, 0.5*normalized_camera['size'][0]],  # noqa
        [0, normalized_camera['focal_length'], 0.5*normalized_camera['size'][1]],  # noqa
        [0, 0, 1],
    ],
    dtype=np.float64,
)
norm_camera_matrix_eye = np.array(
    [
        [normalized_camera_eye['focal_length'], 0, 0.5*normalized_camera_eye['size'][0]],  # noqa
        [0, normalized_camera_eye['focal_length'], 0.5*normalized_camera_eye['size'][1]],  # noqa
        [0, 0, 1],
    ],
    dtype=np.float64,
)

# class for undistorting
class Undistorter:

    _map = None
    _previous_parameters = None

    def __call__(self, image, camera_matrix, distortion, is_gazecapture=False):
        h, w, _ = image.shape
        all_parameters = np.concatenate([camera_matrix.flatten(),
                                         distortion.flatten(),
                                         [h, w]])
        if (self._previous_parameters is None
                or len(self._previous_parameters) != len(all_parameters)
                or not np.allclose(all_parameters, self._previous_parameters)):
            print('Distortion map parameters updated.')
            self._map = cv.initUndistortRectifyMap(
                camera_matrix, distortion, R=None,
                newCameraMatrix=camera_matrix if is_gazecapture else None,
                size=(w, h), m1type=cv.CV_32FC1)
            print('fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f' % (
                    camera_matrix[0, 0], camera_matrix[1, 1],
                    camera_matrix[0, 2], camera_matrix[1, 2]))
            self._previous_parameters = np.copy(all_parameters)

        # Apply
        return cv.remap(image, self._map[0], self._map[1], cv.INTER_LINEAR)


undistort = Undistorter()

# some util functions
# Functions to calculate relative rotation matrices for gaze dir. and head pose
def R_x(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    return np.array([
        [1., 0., 0.],
        [0., cos_, -sin_],
        [0., sin_, cos_]
    ]).astype(np.float32)


def R_y(phi):
    sin_ = np.sin(phi)
    cos_ = np.cos(phi)
    return np.array([
        [cos_, 0., sin_],
        [0., 1., 0.],
        [-sin_, 0., cos_]
    ]).astype(np.float32)


def calculate_rotation_matrix(e):
    return np.matmul(R_y(e[1]), R_x(e[0]))

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2,
              color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx,
                                   eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_out


def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
    #print(vectors)
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def pitchyaw_to_vector(pitchyaws):
    #print(pitchyaws)
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def data_normalization(dataset_name, dataset_path, group, output_path, subject_id):

    # Prepare methods to organize per-entry outputs
    to_write_right = {}
    to_write_left = {}
    def add(key, value, to_write):  # noqa
        if key not in to_write:
            to_write[key] = [value]
        else:
            to_write[key].append(value)

    # Iterate through group (person_id)
    num_entries = next(iter(group.values())).shape[0]
    for i in range(num_entries):
        # Perform data normalization
        processed_entry = data_normalization_entry(dataset_name, dataset_path,
                                                   group, i, subject_id)
        if processed_entry is None:
            continue


        add('eye', processed_entry['right_eye'], to_write_right)
        add('face', processed_entry['face'], to_write_right)
        add('origin', processed_entry['right_eye_origin'], to_write_right)
        add('head_rot_matrix', processed_entry['right_rot_matrix'], to_write_right)
        add('subject_id', processed_entry['subject_id'], to_write_right)
        add('side', 1, to_write_right)
        add('gaze', processed_entry['right_gaze'], to_write_right)

        add('eye', processed_entry['flipped_left_eye'], to_write_right)
        add('face', processed_entry['flipped_face'], to_write_right)
        add('origin', processed_entry['flipped_left_eye_origin'], to_write_right)
        add('head_rot_matrix', processed_entry['flipped_left_rot_matrix'], to_write_right)
        add('subject_id', processed_entry['subject_id'], to_write_right)
        add('side', 0, to_write_right)
        add('gaze', processed_entry['flipped_left_gaze'], to_write_right)

    if len(to_write_right) == 0:
        return

    # Cast to numpy arrays
    #print(to_write_right)
    #print(to_write_left)

    for key, values in to_write_right.items():
        to_write_right[key] = np.asarray(values)
        print('%s: ' % key, to_write_right[key].shape)

    for key, values in to_write_left.items():
        to_write_left[key] = np.asarray(values)
        print('%s: ' % key, to_write_right[key].shape)

    # Write to HDF, this will be used for training
    def write_to_hdf(to_write, output_path):
        with h5py.File(output_path,
                       'a' if os.path.isfile(output_path) else 'w') as f:
            if person_id in f:
                del f[person_id]
            group = f.create_group(person_id)
            for key, values in to_write.items():
                group.create_dataset(
                    key, data=values,
                    chunks=(
                        tuple([1] + list(values.shape[1:]))
                        if isinstance(values, np.ndarray)
                        else None
                    ),
                    compression='lzf',
                )
    write_to_hdf(to_write_right, output_path)
    #write_to_hdf(to_write_left, output_path)

# the basic normalisation code given undistorted normal camera features and images
def get_perspective_patch(image, norm_camera_matrix, camera_matrix, origin, rotate_mat):
    # Code below is an adaptation of code by Xucong Zhang
    # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/
    # actual distance between gaze origin and original camera
    distance = np.linalg.norm(origin)  # (g_o)
    z_scale = normalized_camera['distance'] / distance
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = z_scale

    hRx = rotate_mat[:, 0]
    forward = (origin / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    # transformation matrix
    W = np.dot(np.dot(norm_camera_matrix, S),
               np.dot(R, np.linalg.inv(camera_matrix)))

    ow, oh = normalized_camera['size']
    patch = cv.warpPerspective(image, W, (ow, oh))  # image normalization

    R = np.asmatrix(R)
    return patch, R, W

# data normalisation per entry
def data_normalization_entry(dataset_name, dataset_path, group, i, subject_id):

    # Form original camera matrix
    fx, fy, cx, cy = group['camera_parameters'][i, :]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=np.float64)

    # Grab image
    distortion = group['distortion_parameters'][i, :]
    image_path = '%s/%s' % (dataset_path,
                            group['file_name'][i].decode('utf-8'))
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = undistort(image, camera_matrix, distortion,
                      is_gazecapture=(dataset_name == 'GazeCapture'))
    image = image[:, :, ::-1]  # BGR to RGB

    # Calculate rotation matrix and euler angles
    group
    rvec = group['head_pose'][i, :3].reshape(3, 1)
    tvec = group['head_pose'][i, 3:].reshape(3, 1)
    rotate_mat, _ = cv.Rodrigues(rvec)

    # Project 3D face model points, and check if any are beyond image frame
    points_2d = cv.projectPoints(full_face_model_3d_coordinates, rvec, tvec,
                                 camera_matrix, distortion)[0].reshape(-1, 2)

    face_points = cv.projectPoints(face_model_3d_coordinates, rvec, tvec,
                                 camera_matrix, distortion)[0].reshape(-1, 2)
    tmp_image = np.copy(image[:, :, ::-1])
    for x, y in face_points:
        cv.drawMarker(tmp_image, (int(x), int(y)), color=[0, 0, 255],
                      markerType=cv.MARKER_CROSS,
                      markerSize=2, thickness=1)
    ih, iw, _ = image.shape

    # Take mean face model landmarks and get transformed 3D positions
    landmarks_3d = np.matmul(rotate_mat, face_model_3d_coordinates.T).T
    landmarks_3d += tvec.T

    # Gaze-origin (g_o) and target (g_t)
    g_o = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
    g_o = landmarks_3d[-1, :]  # Face
    g_o = g_o.reshape(3, 1)
    g_t = group['3d_gaze_target'][i, :].reshape(3, 1)
    g = g_t - g_o
    g /= np.linalg.norm(g)

    # Gaze origins and vectors for left/right eyes
    g_l_o = np.mean(landmarks_3d[9:11, :], axis=0).reshape(3, 1)
    # because later we need a flipped image
    flipped_g_l_o = np.copy(g_l_o)
    flipped_g_l_o[0] = -flipped_g_l_o[0]
    g_r_o = np.mean(landmarks_3d[11:13, :], axis=0).reshape(3, 1)
    g_l = g_t - g_l_o
    g_r = g_t - g_r_o
    g_l /= np.linalg.norm(g_l)
    g_r /= np.linalg.norm(g_r)


    # normalise and warp with respect to normalized cameras
    ow, oh = normalized_camera['size']
    patch, R, W_face = get_perspective_patch(image, norm_camera_matrix, camera_matrix, g_o, rotate_mat)
    right_eye, R_right_eye, W_right_eye = get_perspective_patch(image, norm_camera_matrix_eye, camera_matrix, g_r_o, rotate_mat)
    left_eye, R_left_eye, W_left_eye = get_perspective_patch(image, norm_camera_matrix_eye, camera_matrix, g_l_o, rotate_mat)


    # Correct head pose
    h = np.array([np.arcsin(rotate_mat[1, 2]),
                  np.arctan2(rotate_mat[0, 2], rotate_mat[2, 2])])
    head_mat = R * rotate_mat
    head_mat_right = R_right_eye * rotate_mat

    # we need a flipped head_mat
    head_mat_left = R_left_eye * rotate_mat
    head_mat_left_pitchYaw = np.array([np.arcsin(head_mat_left[1, 2]),
                    np.arctan2(head_mat_left[0, 2], head_mat_left[2, 2])])

    # flipping means adding np.pi to phi, pitch
    head_mat_left_pitchYaw[1] = head_mat_left_pitchYaw[1] + np.pi

    flipped_head_mat_left = calculate_rotation_matrix(head_mat_left_pitchYaw)
    n_h = np.array([np.arcsin(head_mat[1, 2]),
                    np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    # Correct gaze
    n_g = R * g
    n_g /= np.linalg.norm(n_g)
    n_g = vector_to_pitchyaw(-n_g.T).flatten()

    # Gaze for left/right eyes
    n_g_l = R_left_eye * g_l
    n_g_r = R_right_eye * g_r
    n_g_l /= np.linalg.norm(n_g_l)
    n_g_r /= np.linalg.norm(n_g_r)
    n_g_l_vec = n_g_l
    n_g_r_vec = n_g_r
    #print(-n_g_l)
    n_g_l = vector_to_pitchyaw(-n_g_l.T).flatten()
    n_g_r = vector_to_pitchyaw(-n_g_r.T).flatten()

    # ignore, just checking if the dimensions are intact
    '''
    n_g_l = pitchyaw_to_vector(n_g_l)
    n_g_r = pitchyaw_to_vector(n_g_r)

    print(n_g_l_vec)
    print(-n_g_l.T)

    print(n_g_r_vec)
    print(-n_g_r.T)
    '''
    #left_eye = patch[int(0.22 * oh):int(0.40 * oh), int(0.10 * ow):int(0.5 * ow)]
    #right_eye = patch[int(0.22 * oh):int(0.40 * oh), int(0.5 * ow):int(0.90 * ow)]
    n_g_r_vec = pitchyaw_to_vector(np.expand_dims(n_g_r, axis=0)).flatten()
    n_g_r_vec = -n_g_r_vec.T

    flipped_left_eye = cv.flip(left_eye, 1)
    flipped_face = cv.flip(patch, 1)
    n_g_l_flipped = np.copy(n_g_l)
    n_g_l_flipped[1] = n_g_l_flipped[1] + np.pi
    #print(np.array([n_g_l_flipped]))
    n_g_l_flipped_vec = pitchyaw_to_vector(np.expand_dims(n_g_l_flipped, axis=0)).flatten()
    n_g_l_flipped_vec = -n_g_l_flipped_vec.T
    #print(n_g_l_flipped_vec)

    # Basic visualization for debugging purposes
    # we draw the gaze directions before and after flipping
    if i % 1 == 0:
        to_visualize = cv.equalizeHist(cv.cvtColor(patch, cv.COLOR_RGB2GRAY))
        to_visualize = draw_gaze(to_visualize, (0.25 * ow, 0.3 * oh), n_g_l,
                                 length=80.0, thickness=1)
        to_visualize = draw_gaze(to_visualize, (0.75 * ow, 0.3 * oh), n_g_r,
                                 length=80.0, thickness=1)

        n_g_r_ = vector_to_pitchyaw(-np.expand_dims(n_g_r_vec, axis=1).T).flatten()
        to_visualize = draw_gaze(to_visualize, (0.75 * ow, 0.3 * oh), n_g_r_,
                                 length=80.0, thickness=1, color=(0, 255, 0))
        if np.allclose( n_g_r_,  n_g_r) == False:
            print(n_g_r_)
            print(n_g_r)
            cv.imwrite("{}.png".format(i), to_visualize)
            print('failed')
            exit(1)

        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.3 * oh), n_g,
                                 length=80.0, thickness=1)
        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.5 * oh), n_h,
                                 length=40.0, thickness=3, color=(0, 0, 0))
        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.5 * oh), n_h,
                                 length=40.0, thickness=1,
                                 color=(255, 255, 255))
        n_g_l_flipped_ = vector_to_pitchyaw(-np.expand_dims(n_g_l_flipped_vec, axis=1).T).flatten()
        if n_g_l_flipped_[1] < 0:
            n_g_l_flipped_[1] += 2*np.pi
        to_visualize1 = cv.flip(to_visualize, 1)
        n_g_l[1] = n_g_l[1] + np.pi
        to_visualize1 = draw_gaze(to_visualize1, (0.75 * ow, 0.3 * oh), n_g_l_flipped_,
                                  length=80.0, thickness=1, color=(0, 255, 0))
        if np.allclose(n_g_l_flipped_, n_g_l) == False or np.allclose(n_g_l_flipped_, n_g_l_flipped) == False:
            print(n_g_l)
            print(n_g_l_flipped)
            print(n_g_l_flipped_)
            print("failed")
            cv.imwrite("flipped_{}.png".format(i), to_visualize1)
            exit(1)



        cv.imwrite("right_eye.png", right_eye)
        cv.imwrite("flipped_left_eye.png", flipped_left_eye)
        cv.imwrite("face.png", to_visualize)
        cv.imwrite("flipped_face.png", to_visualize1)
        cv.imshow("l_eye", flipped_left_eye)
        cv.imshow("r_eye", right_eye)
        cv.imshow('normalized_patch', to_visualize)
        cv.imshow('flipped', to_visualize1)
        cv.waitKey(1)

    return {
        'right_eye' : right_eye.astype(np.uint8),
        'face': patch.astype(np.uint8),
        'right_eye_origin': g_r_o.astype(np.float32),
        'right_gaze': n_g_r_vec.astype(np.float32),
        'right_rot_matrix' : head_mat_right.astype(np.float32),
        'flipped_left_eye' : flipped_left_eye.astype(np.uint8),
        'flipped_face' : flipped_face.astype(np.uint8),
        'flipped_left_eye_origin': flipped_g_l_o.astype(np.float32),
        'flipped_left_gaze': n_g_l_flipped_vec.astype(np.float32),
        'flipped_left_rot_matrix' : flipped_head_mat_left.astype(np.float32),
        'subject_id' : subject_id
    }


if __name__ == '__main__':
    # Grab SFM coordinates and store
    face_model_fpath = './sfm_face_coordinates.npy'
    face_model_3d_coordinates = np.load(face_model_fpath)

    # Grab all face coordinates
    sfm_model = eos.morphablemodel.load_model('./eos/sfm_shape_3448.bin')
    shape_model = sfm_model.get_shape_model()
    sfm_points = np.array([shape_model.get_mean_at_point(d)
                           for d in range(1, 3448)]).reshape(-1, 3)
    rotate_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
    sfm_points = np.matmul(sfm_points, rotate_mat)
    between_eye_point = np.mean([sfm_points[181, :], sfm_points[614, :]],
                                axis=0)
    sfm_points -= between_eye_point.reshape(1, 3)
    full_face_model_3d_coordinates = sfm_points

    # Preprocess some datasets
    output_dir = './outputs_pgte/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    datasets = {
        'MPIIGaze': {
            # Path to the MPIIFaceGaze dataset
            # Sub-folders names should consist of person IDs, for example:
            # p00, p01, p02, ...
            'input-path': '/Users/gagesh/Desktop/MPIIFaceGaze',  # '/media/wookie/WookExt4/datasets/MPIIFaceGaze',

            # A supplementary HDF file with preprocessing data,
            # as provided by us. See grab_prerequisites.bash
            'supplementary': './MPIIFaceGaze_supplementary.h5',

            # Desired output path for the produced HDF
            'output-path': output_dir + '/MPIIGaze1.h5',
        },

    }
    for dataset_name, dataset_spec in datasets.items():
        # Perform the data normalization
        with h5py.File(dataset_spec['supplementary'], 'r') as f:
            id = 0
            for person_id, group in f.items():
                print('')
                print('Processing %s/%s' % (dataset_name, person_id))
                data_normalization(dataset_name,
                                   dataset_spec['input-path'],
                                   group,
                                   dataset_spec['output-path'], id)
                id+=1
