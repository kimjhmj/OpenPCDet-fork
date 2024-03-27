# from xml.etree.ElementTree import QName
import numpy as np
import open3d as o3d
# import open3d.visualization.gui as gui
from open3d.visualization import gui


def _draw_coordinate_plot(sceneWidget, g_plane, tabind):
    '''
    ### Draw coordinate grid at every 10 m
    '''
    pts_for_axis = []
    pts_for_coordinate_x = []
    pts_for_coordinate_y = []
    max_range = 55
    resolusion = 10
    axis_colors = [0.2, 0.2, 0.2]
    line_num = 0 
    
    ### Grid Coordinates and Boundary @ 130m
    pts_for_axis.append([max_range,max_range,0])
    pts_for_axis.append([max_range,-max_range,0])
    pts_for_axis.append([-max_range,max_range,0])
    pts_for_axis.append([-max_range,-max_range,0])

    line_indices = [[0,1],[0,2],[2,3],[1,3]]

    colors = [[1,0,0] for _ in range(0, line_num)]

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(pts_for_axis)
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    lines.colors = o3d.utility.Vector3dVector(colors)

    ### Grid Coordinates and Boundary Cicle @ R 130m
    pts_for_circle = [[np.round(max_range*np.cos(np.pi/180 * i), 1),
                        np.round(max_range*np.sin(np.pi/180 * i), 1), 0]
                        for i in range(360)]

    cline_indices = [[i,i+1] for i in range(359)] # yss

    colors = [[1,0,0] for _ in range(360)]

    clines = o3d.geometry.LineSet()
    clines.points = o3d.utility.Vector3dVector(pts_for_circle)
    clines.lines = o3d.utility.Vector2iVector(cline_indices)
    clines.colors = o3d.utility.Vector3dVector(colors)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 3  # note that this is scaled with respect to pixels,

    sceneWidget.scene.add_geometry("boundary", lines, mat) ### 130 meter ROI
    sceneWidget.scene.add_geometry("cboundary", clines, mat) ### R 130 meter ROI

    is_grid_show = True
    sceneWidget.scene.show_ground_plane(True, g_plane)

    ### Label3D text
    for i in range(-max_range, max_range + resolusion, resolusion):
        pts_for_coordinate_x.append([i,0,0])
        pts_for_coordinate_y.append([0,i,0])
        # line_num = line_num + 2

    points_x = o3d.geometry.PointCloud()
    points_y = o3d.geometry.PointCloud()
    points_x.points = o3d.utility.Vector3dVector(pts_for_coordinate_x)
    points_y.points = o3d.utility.Vector3dVector(pts_for_coordinate_y)

    color_list = []
    for idx in range(0,len(points_x.points)):
        color_list.append(axis_colors)
    colors = np.array(color_list)

    points_x.colors = o3d.utility.Vector3dVector(colors)
    points_y.colors = o3d.utility.Vector3dVector(colors)

    offset = 0.0
    x_axis_label, y_axis_label = [], []
    for idx in range(0, len(points_x.points)):
        l = sceneWidget.add_3d_label(points_x.points[idx] - np.array([0,0,offset]) , "{}".format(int(points_x.points[idx][0])))
        l.color = gui.Color(1,0,0)
        x_axis_label.append(l)

        j = sceneWidget.add_3d_label(points_y.points[idx] - np.array([0,0,offset]) , "{}".format(int(points_y.points[idx][1])))
        j.color = gui.Color(0,1,0)
        y_axis_label.append(j)

    return is_grid_show, (x_axis_label, y_axis_label)


def _shortcut_key_info():
    shortcut_list = []
    '''
    shortcut_list.append("    Viz    |    +/-    |  포인트 사이즈와 box 선 두께 up/down")
    shortcut_list.append("    Viz    |     1     |  [OD] 현재 lidar frame 의 PR 위치에서 취득된 포인트들 show")
    shortcut_list.append("    Viz    |     1     |  [SEG] 현재 lidar frame 의 PR 위치에서 취득된 포인트들 show/hide")
    shortcut_list.append("    Viz    |     2     |  [OD] 현재 lidar frame 의 PL 위치에서 취득된 포인트들 show")
    shortcut_list.append("    Viz    |     2     |  [SEG] 현재 lidar frame 의 PL 위치에서 취득된 포인트들 show/hide")
    shortcut_list.append("    Viz    |     3     |  [OD] 현재 lidar frame 의 CL 위치에서 취득된 포인트들 show")
    shortcut_list.append("    Viz    |     3     |  [SEG] 현재 lidar frame 의 CL 위치에서 취득된 포인트들 show/hide")
    shortcut_list.append("    Viz    |     4     |  [OD] 현재 lidar frame 의 PR+PL 위치에서 취득된 포인트들 show")
    shortcut_list.append("    Viz    |     5     |  [OD] 현재 lidar frame 의 PR+PL+CL 위치에서 취득된 포인트들 show")
    shortcut_list.append("    Viz    |     G     |  거리정보를 나타내주는 grid 와 ROI 영역 (130m) show/hide")
    shortcut_list.append("    Viz    | LEFT/RIGHT|  키보드 왼쪽/오른쪽 방향키 -> previous/next frame")

    shortcut_list.append("  Camera   |     T     |  Topview 로 시점 변경")
    shortcut_list.append("  Camera   |     F     |  비스듬히 바라보는 시점으로 변경 : 전방뷰")
    shortcut_list.append("  Camera   |     R     |  비스듬히 바라보는 시점으로 변경 : 후방뷰")
    shortcut_list.append("  Camera   |     W     |  현재 시점에서 카메라를 앞으로 이동")
    shortcut_list.append("  Camera   |     A     |  현재 시점에서 카메라를 왼쪽으로 이동")
    shortcut_list.append("  Camera   |     S     |  현재 시점에서 카메라를 뒤로 이동")
    shortcut_list.append("  Camera   |     D     |  현재 시점에서 카메라를 오른쪽으로 이동")
    shortcut_list.append("  Camera   |     Q     |  현재 시점에서 카메라 중심을 기준으로 시계방향으로 이동")
    shortcut_list.append("  Camera   |     R     |  현재 시점에서 카메라 중심을 기준으로 반시계방향으로 이동")
    shortcut_list.append("  Camera   |   SHIFT   |  카메라 이동/회전 움직임 비율 증가")
    shortcut_list.append("  Camera   | CAPSLOCK  |  카메라 이동/회전 움직임 비율 감소")

    shortcut_list.append("  Image    |     0     |  현재 lidar frame 와 매칭되는 camera 영상 panel show ")
    shortcut_list.append("  Image    |  ALT + D  |  FL 카메라 이미지 출력")
    shortcut_list.append("  Image    |  ALT + F  |  FC 카메라 이미지 출력")
    shortcut_list.append("  Image    |  ALT + G  |  FR 카메라 이미지 출력")
    shortcut_list.append("  Image    |  ALT + C  |  SL 카메라 이미지 출력")
    shortcut_list.append("  Image    |  ALT + V  |  RC 카메라 이미지 출력")
    shortcut_list.append("  Image    |  ALT + B  |  SR 카메라 이미지 출력")
    shortcut_list.append("  Image    |     B     |  현재 카메라의 다음 이미지 출력")
    shortcut_list.append("  Image    |     V     |  현재 카메라의 이전 이미지 출력")

    shortcut_list.append(" Settings  |     M     |  맨 오른쪽 \'main control setting panel\' hide/show")

    shortcut_list.append("Inspection |     K     |  [SEG] Point Select Mode -> \'CTRL\' 키와 마우스 왼쪽 버튼을 누르며 drag 하면 box가 선택되고, 해당 영역에 있는 point 선택됨")
    shortcut_list.append("Inspection |     L     |  [OD]  Point Select Mode -> \'CTRL\' 키와 마우스 왼쪽 버튼을 누르며 drag 하면 box가 선택되고, 해당 영역에 있는 point 선택됨")
    shortcut_list.append("Inspection |  SPACEBAR |  Point Select Mode 에서 선택된 포인트들이 하나의 cluster 로 묶이며, 노란색의 bbox 생성")
    shortcut_list.append("Inspection |     Z     |  현재 노란 bbox cluster 취소 (이전 상태로 되돌아가기)")
    shortcut_list.append("Inspection |   ENTER   |  현재 frame 을 벗어날 때(키보드 왼쪽/오른쪽 키 또는 prev/next 버튼) 나오는 inspection dialog box 에서 \'Ok\' 버튼 클릭 동작")
    shortcut_list.append("Inspection | BACKSPACE |  현재 frame 을 벗어날 때(키보드 왼쪽/오른쪽 키 또는 prev/next 버튼) 나오는 inspection dialog box 에서 \'Not yet\' 버튼 클릭 동작")
    '''

    
    # shortcut_list.append("     Viz    |    +/-    | Size up of the point size and box line thickness")
    # shortcut_list.append("     Viz    |     1     | [OD] Show PR lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     1     | [SEG] Show/hide PR lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     2     | [OD] Show PL lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     2     | [SEG] Show/hide PL lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     3     | [OD] Show CL lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     3     | [SEG] Show/hide CL lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     4     | [OD] Show PR+PL lidar points of current lidar frame")
    # shortcut_list.append("     Viz    |     5     | [OD] Show PR+PL+CL lidar points of current lidar frame")
    # shortcut_list.append("     Viz    |     0     | Show 6 camera images in new window, matching the current lidar frame ")
    # shortcut_list.append("     Viz    |     G     | Show/hide distance information grid and ROI area (130m)")
    # shortcut_list.append("     Viz    | LEFT/RIGHT| Keyboard left/right -> previous/next frame")
 
    # shortcut_list.append("   Camera   |     T     | Change view point : Topview")
    # shortcut_list.append("   Camera   |     F     | Change view point : front view")
    # shortcut_list.append("   Camera   |     R     | Change view point : rear view")
    # shortcut_list.append("   Camera   |     W     | Move camera : forward")
    # shortcut_list.append("   Camera   |     A     | Move camera : left")
    # shortcut_list.append("   Camera   |     S     | Move camera : backward")
    # shortcut_list.append("   Camera   |     D     | Move camera : right")
    # shortcut_list.append("   Camera   |     Q     | Move camera : clockwise from camera center at current viewpoint")
    # shortcut_list.append("   Camera   |     R     | Move camera : counterclockwise from camera center at current viewpoint")
    # shortcut_list.append("   Camera   |   SHIFT   | Increase camera rotation/movement step")
    # shortcut_list.append("   Camera   |  CAPSLOCK | Decrease camera rotation/movement step")

    # shortcut_list.append("  Settings  |     M     | Hide/show Main Control Setting Panel (MCSP)")
    # shortcut_list.append("  Settings  |     N     | [OD] Show/hide Current frame information (Number of objects by class) (created on the left side of the MCSP)")
    # shortcut_list.append("  Settings  |     N     | [SEG] Show/hide Current frame information (Number and ratio of points by class) (created on the left side of the MCSP) ")
    # shortcut_list.append("  Settings  | CTRL + S  | Save config file and Export csv file with current inspection results ")

    # # shortcut_list.append(" Inspection |     L     | Point Select Mode -> point is selected when dragging by pressing the left mouse button with the \'CTRL\' key")
    # # shortcut_list.append(" Inspection |     K     | Point Select Mode(Box) -> when dragging by pressing the left mouse button with \'CTRL\' key, a 2D box is drawn, and points in the box area are selected")
    # shortcut_list.append(" Inspection |     L     | Point Select Mode(Box) -> point is selected given a guided 2D box, when dragging by pressing the left mouse button with the \'CTRL\' key")
    # shortcut_list.append(" Inspection |  SPACEBAR | Selectd points are grouped into one cluster, and yellow box is created")
    # shortcut_list.append(" Inspection |     Z     | Cancel current yellow bbox or newly created point cluster (revert to previous state)")
    # shortcut_list.append(" Inspection |     C     | Clear all selected points and yellow bbox (reset/clear)")
    # shortcut_list.append(" Inspection |   ENTER   | 'Ok' button click action in inspection dialog box when leaving current frame (keyboard left/right keys or prev/next buttons)")
    # shortcut_list.append(" Inspection | BACKSPACE | 'Not yet' button click action in inspection dialog box when leaving current frame (keyboard left/right keys or prev/next buttons)")

    shortcut_list.append("     Viz    |    +/-    | Size up of the point size and box line thickness")
    # shortcut_list.append("     Viz    |     1     | [OD] Show PR lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     1     | [SEG] Show/hide PR lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     2     | [OD] Show PL lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     2     | [SEG] Show/hide PL lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     3     | [OD] Show CL lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     3     | [SEG] Show/hide CL lidar points of the current lidar frame")
    # shortcut_list.append("     Viz    |     4     | [OD] Show PR+PL lidar points of current lidar frame")
    # shortcut_list.append("     Viz    |     5     | [OD] Show PR+PL+CL lidar points of current lidar frame")
    # shortcut_list.append("     Viz    |     0     | Show 6 camera images in new window, matching the current lidar frame ")
    shortcut_list.append("     Viz    |     G     | Show/hide distance information grid and ROI area (130m)")
    # shortcut_list.append("     Viz    |     O     | [SEG] Show/hide OD Box GT information")
    shortcut_list.append("     Viz    | LEFT/RIGHT| Keyboard left/right -> previous/next frame")
 
    shortcut_list.append("   Camera   |     T     | Change view point : Topview")
    shortcut_list.append("   Camera   |     F     | Change view point : front view")
    shortcut_list.append("   Camera   |     R     | Change view point : rear view")
    shortcut_list.append("   Camera   |     W     | Move camera : forward")
    shortcut_list.append("   Camera   |     A     | Move camera : left")
    shortcut_list.append("   Camera   |     S     | Move camera : backward")
    shortcut_list.append("   Camera   |     D     | Move camera : right")
    shortcut_list.append("   Camera   |     Q     | Move camera : clockwise from camera center at current viewpoint")
    shortcut_list.append("   Camera   |     R     | Move camera : counterclockwise from camera center at current viewpoint")
    shortcut_list.append("   Camera   |   SHIFT   | Increase camera rotation/movement step")
    shortcut_list.append("   Camera   |  CAPSLOCK | Decrease camera rotation/movement step")

    # shortcut_list.append("    Image   |     0     |  Show a camera image in new window, matching the current lidar frame (default : FC)")
    # shortcut_list.append("    Image   |  ALT + D  |  change to FL camera view in camera window")
    # shortcut_list.append("    Image   |  ALT + F  |  change to FC camera view in camera window")
    # shortcut_list.append("    Image   |  ALT + G  |  change to FR camera view in camera window")
    # shortcut_list.append("    Image   |  ALT + C  |  change to SL camera view in camera window")
    # shortcut_list.append("    Image   |  ALT + V  |  change to RC camera view in camera window")
    # shortcut_list.append("    Image   |  ALT + B  |  change to SR camera view in camera window")
    # shortcut_list.append("    Image   |     B     |  Show next image, matching the next lidar frame")
    # shortcut_list.append("    Image   |     V     |  Show previous image, matching the previous lidar frame")

    shortcut_list.append("  Settings  |     M     | Hide/show Main Control Setting Panel (MCSP)")
    # shortcut_list.append("  Settings  | CTRL + S  | Save config file and Export csv file with current inspection results ")

    # shortcut_list.append(" Inspection |     L     | Point Select Mode -> point is selected when dragging by pressing the left mouse button with the \'CTRL\' key")
    # shortcut_list.append(" Inspection |     K     | [SEG] Point Select Mode(Box) -> point is selected given a guided 2D box, when dragging by pressing the left mouse button with the \'CTRL\' key")
    # shortcut_list.append(" Inspection |     L     | [OD]  Point Select Mode(Box) -> point is selected given a guided 2D box, when dragging by pressing the left mouse button with the \'CTRL\' key")
    # shortcut_list.append(" Inspection |  SPACEBAR | Selectd points are grouped into one cluster, and yellow box is created")
    # shortcut_list.append(" Inspection |     Z     | Cancel current yellow bbox or newly created point cluster (revert to previous state)")
    # shortcut_list.append(" Inspection |     C     | Clear all selected points and yellow bbox (reset/clear)")
    shortcut_list.append(" Inspection |   ENTER   | 'Ok' button click action in inspection dialog box when leaving current frame (keyboard left/right keys or prev/next buttons)")
    shortcut_list.append(" Inspection | BACKSPACE | 'Not yet' button click action in inspection dialog box when leaving current frame (keyboard left/right keys or prev/next buttons)")

    
    return shortcut_list

