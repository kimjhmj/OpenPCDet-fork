"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    ]

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba

import matplotlib as plt
def intensity2rgb(intensity:np.ndarray, min=0, max=255):
    def adjust_display_range_ratio(data, min, max):
        data[data > max] = max
        data[data < min] = min
        return data
    def normalize(data, min, max):
        _range = max - min
        return (data - min) / _range
    intensity = adjust_display_range_ratio(intensity, min, max)
    scalar = normalize(intensity, min, max)
    cmap = plt.cm.get_cmap('rainbow')
    rgba_img = cmap(scalar)
    rgb_img = np.delete(rgba_img, 3, 2)
    return np.squeeze(rgb_img)

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    if scale < 0.001:
        scale = 0.001
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = 0.1
    cylinder_radius = 0.05
    mesh_frame = open3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None, color=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 1.0
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)

    if color is not None:
        pass

    return(mesh)

def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, radar_points=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    # draw lidar points
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    # intensity_vector = points[:,3:4]
    # rgb_color_space = intensity2rgb(intensity_vector)
    timelag_vector = points[:,4:5]
    rgb_color_space = intensity2rgb(timelag_vector, min=0.0, max=0.5)
    # pts.colors = open3d.utility.Vector3dVector(rgb_color_space)
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    vis.add_geometry(pts)

    # draw radar points
    if radar_points is not None:
        if isinstance(radar_points, torch.Tensor):
            radar_points = radar_points.cpu().numpy()
        radar_points[:, 2] += 2.0 # lift radar poitns for visualization
        rad_pts = open3d.geometry.PointCloud()
        rad_pts.points = open3d.utility.Vector3dVector(radar_points[:, :3])

        timelag_vector = radar_points[:,6:7]
        rgb_color_space = intensity2rgb(timelag_vector, min=0.0, max=0.5)
        rad_pts.colors = open3d.utility.Vector3dVector(rgb_color_space)
        # rad_pts.colors = open3d.utility.Vector3dVector(np.ones((radar_points.shape[0], 3)))
        vis.add_geometry(rad_pts)

        for i in range(radar_points.shape[0]):
            arrow = get_arrow(origin=radar_points[i, :3], vec=[radar_points[i, 3], radar_points[i, 4], 0])
            vis.add_geometry(arrow)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
