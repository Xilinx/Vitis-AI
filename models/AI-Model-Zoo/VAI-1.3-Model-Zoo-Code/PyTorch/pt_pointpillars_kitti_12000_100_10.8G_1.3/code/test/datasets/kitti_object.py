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


def get_lidar_index_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return fov_inds


def depth_region_pt3d(depth, obj):
    b = obj.box2d
    # depth_region = depth[b[0]:b[2],b[2]:b[3],0]
    pt3d = []
    # import pdb; pdb.set_trace()
    for i in range(int(b[0]), int(b[2])):
        for j in range(int(b[1]), int(b[3])):
            pt3d.append([j, i, depth[j, i]])
    return np.array(pt3d)


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def get_point_index_in_range(pc_velo, range):
    keep_index = (
        (pc_velo[:, 0] >= range[0])
        & (pc_velo[:, 0] <= range[1])
        & (pc_velo[:, 1] >= range[2])
        & (pc_velo[:, 1] <= range[3])
        & (pc_velo[:, 2] >= range[4])
        & (pc_velo[:, 2] <= range[5])
    )
    return keep_index


def show_lidar_with_depth(
    pc_velo,
    objects,
    calib,
    fig,
    range=[0,70.4,-40,40,-3,1],
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
    constraint_box=False,
    pc_label=False,
    save=False,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))

    if range:
        pc_velo_index = get_point_index_in_range(pc_velo, range)
        pc_velo = pc_velo[pc_velo_index, :]
        print("point num in range {}: {}".format(range, pc_velo.shape[0]))

    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        print(("FOV point num: ", pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "data/object/training/depth_pc"
            save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))
            print(save_filename)
            # np.save(save_filename+".npy", np.array(depth_pc_velo))
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, label=obj.type)

    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show()


def save_depth0(
    data_idx,
    pc_velo,
    calib,
    img_fov,
    img_width,
    img_height,
    depth,
    constraint_box=False,
):

    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        type = np.zeros((pc_velo.shape[0], 1))
        pc_velo = np.hstack((pc_velo, type))
        print(("FOV point num: ", pc_velo.shape))
    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))

        type = np.ones((depth_pc_velo.shape[0], 1))
        depth_pc_velo = np.hstack((depth_pc_velo, type))
        print("depth_pc_velo:", depth_pc_velo.shape)

        depth_pc = np.concatenate((pc_velo, depth_pc_velo), axis=0)
        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def save_depth(
    data_idx,
    pc_velo,
    calib,
    img_fov,
    img_width,
    img_height,
    depth,
    constraint_box=False,
):

    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc = np.hstack((depth_pc_velo, indensity))

        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)

        # Draw depth
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))

        # Draw heading arrow
        _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show(1)


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where(
        (pc[:, 0] >= bmin[0])
        & (pc[:, 0] <= bmax[0])
        & (pc[:, 1] >= bmin[1])
        & (pc[:, 1] <= bmax[1])
        & (pc[:, 2] >= bmin[2])
        & (pc[:, 2] <= bmax[2])
    )[0]
    # print(pc[ind,:])
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def stat_lidar_with_boxes(pc_velo, objects, calib):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    # print(('All point num: ', pc_velo.shape[0]))

    # draw_lidar(pc_velo, fig=fig)
    # color=(0,1,0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = get_velo_whl(box3d_pts_3d_velo, pc_velo)
        print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    """ Project LiDAR points to image """
    img =  np.copy(img)
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            2,
            color=tuple(color),
            thickness=-1,
        )
    cv2.imshow("projection", img)
    return img


def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    print("top_image:", top_image.shape)
    # gt

    def bbox3d(obj):
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    cv2.imshow("top_image", top_image)
    return top_image


def dataset_viz(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    ## load 2d detection results
    #objects2ds = read_det_file("box2d.list")

    if args.show_lidar_with_depth:
        import mayavi.mlab as mlab

        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
        )
    for data_idx in range(len(dataset)):
        if args.ind > 0:
            data_idx = args.ind
        # Load data from dataset
        if args.split == "training":
            objects = dataset.get_label_objects(data_idx)
        else:
            objects = []
        #objects2d = objects2ds[data_idx]

        objects_pred = None
        if args.pred:
            # if not dataset.isexist_pred_objects(data_idx):
            #    continue
            objects_pred = dataset.get_pred_objects(data_idx)
            if objects_pred == None:
                continue
        if objects_pred == None:
            print("no pred file")
            # objects_pred[0].print_object()

        n_vec = 4
        if args.pc_label:
            n_vec = 5

        dtype = np.float32
        if args.dtype64:
            dtype = np.float64
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        if args.depth:
            depth, _ = dataset.get_depth(data_idx)
            print(data_idx, "depth shape: ", depth.shape)
        else:
            depth = None

        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))

        if args.stat:
            stat_lidar_with_boxes(pc_velo, objects, calib)
            continue
        print("======== Objects in Ground Truth ========")
        n_obj = 0
        for obj in objects:
            if obj.type != "DontCare":
                print("=== {} object ===".format(n_obj + 1))
                obj.print_object()
                n_obj += 1

        # Draw 3d box in LiDAR point cloud
        if args.show_lidar_topview_with_boxes:
            # Draw lidar top view
            show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred)

        # show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
        if args.show_image_with_boxes:
            # Draw 2d and 3d boxes on image
            show_image_with_boxes(img, objects, calib, True, depth)
        if args.show_lidar_with_depth:
            # Draw 3d box in LiDAR point cloud
            show_lidar_with_depth(
                pc_velo,
                objects,
                calib,
                fig,
                [0,70.4,-40,40,-3,1],
                args.img_fov,
                img_width,
                img_height,
                objects_pred,
                depth,
                img,
                constraint_box=args.const_box,
                save=args.save_depth,
                pc_label=args.pc_label,
            )
            # show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height, \
            #    objects_pred, depth, img)
        if args.show_lidar_on_image:
            # Show LiDAR points on image.
            show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height)
        input_str = raw_input()

        mlab.clf()
        if input_str == "killall":
            break


def depth_to_lidar_format(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    for data_idx in range(len(dataset)):
        # Load data from dataset

        pc_velo = dataset.get_lidar(data_idx)[:, 0:4]
        calib = dataset.get_calibration(data_idx)
        depth, _ = dataset.get_depth(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        print(data_idx, "depth shape: ", depth.shape)
        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))
        save_depth(
            data_idx,
            pc_velo,
            calib,
            args.img_fov,
            img_width,
            img_height,
            depth,
            constraint_box=args.const_box,
        )
        #input_str = raw_input()


def read_det_file(det_filename):
    """ Parse lines in 2D detection output files """
    #det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    objects = {}
    with open(det_filename, "r") as f:
        for line in f.readlines():
            obj = utils.Object2d(line.rstrip())
            if obj.img_name not in objects.keys():
                objects[obj.img_name] = []
            objects[obj.img_name].append(obj)
        # objects = [utils.Object2d(line.rstrip()) for line in f.readlines()]

    return objects


if __name__ == "__main__":
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    parser = argparse.ArgumentParser(description="KIITI Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="data/object",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-i",
        "--ind",
        type=int,
        default=0,
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-p", "--pred", action="store_true", help="show predict results"
    )
    parser.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help=" stat the w/h/l of point cloud in gt bbox",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="use training split or testing split (default: training)",
    )
    parser.add_argument(
        "-l",
        "--lidar",
        type=str,
        default="velodyne",
        metavar="N",
        help="velodyne dir  (default: velodyne)",
    )
    parser.add_argument(
        "-e",
        "--depthdir",
        type=str,
        default="depth",
        metavar="N",
        help="depth dir  (default: depth)",
    )
    parser.add_argument(
        "-r",
        "--preddir",
        type=str,
        default="pred",
        metavar="N",
        help="predicted boxes  (default: pred)",
    )
    parser.add_argument("--gen_depth", action="store_true", help="generate depth")
    parser.add_argument("--vis", action="store_true", help="show images")
    parser.add_argument("--depth", action="store_true", help="load depth")
    parser.add_argument("--img_fov", action="store_true", help="front view mapping")
    parser.add_argument("--const_box", action="store_true", help="constraint box")
    parser.add_argument(
        "--save_depth", action="store_true", help="save depth into file"
    )
    parser.add_argument(
        "--pc_label", action="store_true", help="5-verctor lidar, pc with label"
    )
    parser.add_argument(
        "--dtype64", action="store_true", help="for float64 datatype, default float64"
    )

    parser.add_argument(
        "--show_lidar_on_image", action="store_true", help="project lidar on image"
    )
    parser.add_argument(
        "--show_lidar_with_depth",
        action="store_true",
        help="--show_lidar, depth is supported",
    )
    parser.add_argument(
        "--show_image_with_boxes", action="store_true", help="show lidar"
    )
    parser.add_argument(
        "--show_lidar_topview_with_boxes",
        action="store_true",
        help="show lidar topview",
    )
    args = parser.parse_args()
    if args.pred:
        assert os.path.exists(args.dir + "/" + args.split + "/pred")

    if args.vis:
        dataset_viz(args.dir, args)
    if args.gen_depth:
        depth_to_lidar_format(args.dir, args)

