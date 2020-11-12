# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        """ Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        """
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def project_depth_to_velo(self, depth, constraint_box=True):
        depth_pt3d = get_depth_pt3d(depth)
        depth_UVDepth = np.zeros_like(depth_pt3d)
        depth_UVDepth[:, 0] = depth_pt3d[:, 1]
        depth_UVDepth[:, 1] = depth_pt3d[:, 0]
        depth_UVDepth[:, 2] = depth_pt3d[:, 2]
        # print("depth_pt3d:",depth_UVDepth.shape)
        depth_pc_velo = self.project_image_to_velo(depth_UVDepth)
        # print("dep_pc_velo:",depth_pc_velo.shape)
        if constraint_box:
            depth_box_fov_inds = (
                (depth_pc_velo[:, 0] < cbox[0][1])
                & (depth_pc_velo[:, 0] >= cbox[0][0])
                & (depth_pc_velo[:, 1] < cbox[1][1])
                & (depth_pc_velo[:, 1] >= cbox[1][0])
                & (depth_pc_velo[:, 2] < cbox[2][1])
                & (depth_pc_velo[:, 2] >= cbox[2][0])
            )
            depth_pc_velo = depth_pc_velo[depth_box_fov_inds]
        return depth_pc_velo


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform_from_rot_trans(R, t):
    """ Transforation matrix from rotation matrix and translation vector. """
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_depth_v(img_filename):
    # return cv2.imread(img_filename)
    disp_img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)
    disp_img = disp_img.astype(np.float)
    return disp_img / 256.0


def load_depth0(img_filename):
    # return cv2.imread(img_filename)
    depth_img = np.array(Image.open(img_filename), dtype=int)

    depth_img = depth_img.astype(np.float) / 256.0

    return depth_img


def load_depth(img_filename):
    isexist = True
    disp_img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)
    if disp_img is None:
        isexist = False
        disp_img = np.zeros((370, 1224))
    else:
        disp_img = disp_img.astype(np.float)
    return disp_img / 256.0, isexist


def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def lidar_to_top_coords(x, y):
    # print("TOP_X_MAX-TOP_X_MIN:",TOP_X_MAX,TOP_X_MIN)
    Xn = int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Yn = int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
    yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)

    return xx, yy


def lidar_to_top(lidar):

    idx = np.where(lidar[:, 0] > TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 0] < TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 1] > TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 1] < TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 2] > TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 2] < TOP_Z_MAX)
    lidar = lidar[idx]

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]
    qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
    qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
    # qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION
    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    Z0, Zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
    height = Xn - X0
    width = Yn - Y0
    channel = Zn - Z0 + 2
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height, width, channel), dtype=np.float32)

    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  # new method
        for x in range(Xn):
            ix = np.where(quantized[:, 0] == x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0:
                continue
            yy = -x

            for y in range(Yn):
                iy = np.where(quantized_x[:, 1] == y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if count == 0:
                    continue
                xx = -y

                top[yy, xx, Zn + 1] = min(1, np.log(count + 1) / math.log(32))
                max_height_point = np.argmax(quantized_xy[:, 2])
                top[yy, xx, Zn] = quantized_xy[max_height_point, 3]

                for z in range(Zn):
                    iz = np.where(
                        (quantized_xy[:, 2] >= z) & (quantized_xy[:, 2] <= z + 1)
                    )
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0:
                        continue
                    zz = z

                    # height per slice
                    max_height = max(0, np.max(quantized_xyz[:, 2]) - z)
                    top[yy, xx, zz] = max_height

    # if 0: #unprocess
    #     top_image = np.zeros((height,width,3),dtype=np.float32)
    #
    #     num = len(lidar)
    #     for n in range(num):
    #         x,y = qxs[n],qys[n]
    #         if x>=0 and x <width and y>0 and y<height:
    #             top_image[y,x,:] += 1
    #
    #     max_value=np.max(np.log(top_image+0.001))
    #     top_image = top_image/max_value *255
    #     top_image=top_image.astype(dtype=np.uint8)

    return top


MATRIX_Mt = np.array(
    [
        [2.34773698e-04, 1.04494074e-02, 9.99945389e-01, 0.00000000e00],
        [-9.99944155e-01, 1.05653536e-02, 1.24365378e-04, 0.00000000e00],
        [-1.05634778e-02, -9.99889574e-01, 1.04513030e-02, 0.00000000e00],
        [5.93721868e-02, -7.51087914e-02, -2.72132796e-01, 1.00000000e00],
    ]
)

MATRIX_Kt = np.array(
    [[721.5377, 0.0, 0.0], [0.0, 721.5377, 0.0], [609.5593, 172.854, 1.0]]
)


def box3d_to_rgb_box00(box3d):

    # box3d = boxes3d[n]
    Ps = np.hstack((box3d, np.ones((8, 1))))
    Qs = np.matmul(Ps, MATRIX_Mt)
    Qs = Qs[:, 0:3]
    qs = np.matmul(Qs, MATRIX_Kt)
    zs = qs[:, 2].reshape(8, 1)
    qs = qs / zs

    return qs[:, 0:2]


def box3d_to_rgb_box0000(boxes3d, Mt=None, Kt=None):
    # if (cfg.DATA_SETS_TYPE == 'kitti'):
    if Mt is None:
        Mt = np.array(MATRIX_Mt)
    if Kt is None:
        Kt = np.array(MATRIX_Kt)

    num = len(boxes3d)
    projections = np.zeros((num, 8, 2), dtype=np.int32)
    for n in range(num):
        box3d = boxes3d[n]
        Ps = np.hstack((box3d, np.ones((8, 1))))
        Qs = np.matmul(Ps, Mt)
        Qs = Qs[:, 0:3]
        qs = np.matmul(Qs, Kt)
        zs = qs[:, 2].reshape(8, 1)
        qs = qs / zs
        # pts_3d=project_to_image(qs[:,0:2], P)
        projections[n] = qs[:, 0:2]
        # projections[n] = proj3d_to_2d(qs[:,0:2])
        # projections[n] = pts_3d
    return projections


def proj3d_to_2d(rgbpoint):
    x0 = np.min(rgbpoint[:, 0])
    x1 = np.max(rgbpoint[:, 0])
    y0 = np.min(rgbpoint[:, 1])
    y1 = np.max(rgbpoint[:, 1])
    # x0 = max(0,x0)
    # x1 = min(x1, proj.image_width)
    # y0 = max(0,y0)
    # y1 = min(y1, proj.image_height)
    return np.array([x0, y0, x1, y1])


def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    """

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)


def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image


def draw_top_image(lidar_top):
    top_image = np.sum(lidar_top, axis=2)
    top_image = top_image - np.min(top_image)
    divisor = np.max(top_image) - np.min(top_image)
    top_image = top_image / divisor * 255
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image


def draw_box3d_on_top(
    image,
    boxes3d,
    color=(255, 255, 255),
    thickness=1,
    scores=None,
    text_lables=[],
    is_gt=False,
):

    # if scores is not None and scores.shape[0] >0:
    # print(scores.shape)
    # scores=scores[:,0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    num = len(boxes3d)
    startx = 5
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)
        if is_gt:
            color = (0, 255, 0)
            startx = 5
        else:
            color = heat_map_rgb(0.0, 1.0, scores[n]) if scores is not None else 255
            startx = 85
        cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    return img


# hypothesis function
def hypothesis_func(w, x):
    w1, w0 = w
    return w1 * x + w0


# error function
def error_func(w, train_x, train_y):
    return hypothesis_func(w, train_x) - train_y


def dump_fit_func(w_fit):
    w1, w0 = w_fit
    print("fitting line=", str(w1) + "*x + " + str(w0))
    return


# square error
def dump_fit_cost(w_fit, train_x, train_y):
    error = error_func(w_fit, train_x, train_y)
    square_error = sum(e * e for e in error)
    print("fitting cost:", str(square_error))
    return square_error


def linear_regression(train_x, train_y, test_x):
    # train set
    # train_x = np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
    # train_y = np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])

    # linear regression by leastsq
    # msg = "invoke scipy leastsq"
    w_init = [20, 1]  # weight factor init
    fit_ret = leastsq(error_func, w_init, args=(train_x, train_y))
    w_fit = fit_ret[0]

    # dump fit result
    dump_fit_func(w_fit)
    dump_fit_cost(w_fit, train_x, train_y)

    # test set
    # test_x = np.array(np.arange(train_x.min(), train_x.max(), 1.0))
    test_y = hypothesis_func(w_fit, test_x)
    test_y0 = hypothesis_func(w_fit, train_x)
    return test_y, test_y0
