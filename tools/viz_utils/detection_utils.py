# from ast import AsyncFunctionDef
# from pickle import FALSE
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys
import time
import warnings

from copy import deepcopy


from tool_utils import _draw_coordinate_plot

import platform

# for test
# import pickle
if platform.system() == "Linux":
    from typing import Optional, Sequence, Union
    import torch

    from tools.utils.load_config import load_config_file

    from modules.datasets import build_dataloader
    from modules.models import build_network, load_data_to_gpu
    # from mmengine.config import Config #, DictAction
    # from mmengine.utils import ProgressBar, mkdir_or_exist

    # from mmdet3d.registry import DATASETS, MODELS #, VISUALIZERS
    # # from mmdet3d.utils import replace_ceph_backend
    # # from easydict import EasyDict

    # from mmengine.registry import init_default_scope
    # from mmengine.runner import load_checkpoint


    # from mmengine.dataset import Compose, pseudo_collate
    


top_eyeview = np.array([0,0,-20])


class Object3d_HMC(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        # self.xmin = data[4]  # left
        # self.ymin = data[5]  # top
        # self.xmax = data[6]  # right
        # self.ymax = data[7]  # bottom
        # self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[4]  # box height
        self.w = data[5]  # box width
        self.l = data[6]  # box length (in meters)
        # self.t = (data[13], -data[11], data[12])  # location (x,y,z) in camera coord.
        # self.ry = -data[14]#-math.pi/2  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        # self.t = (data[13], -data[11], -data[12])  # location (x,y,z) in camera coord.
        # self.t = (data[11], data[12], data[13]/2)  # location (x,y,z) in camera coord.
        self.t = (data[7], data[8], data[9])  # location (x,y,z) in camera coord.
        self.ry = -data[10] #+math.pi/2  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.side = data[11]


    def estimate_difficulty(self):
        """ Function that estimate difficulty to detect the object as defined in kitti website"""
        try:
            # height of the bounding box
            bb_height = np.abs(self.xmax - self.xmin)

            if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
                return "Easy"
            elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
                return "Moderate"
            elif (
                bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50
            ):
                return "Hard"
            else:
                return "Unknown"
        except Exception as err:
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')
            
    def print_object(self):
        self.logger.info(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )  
        self.logger.info(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        self.logger.info("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        self.logger.info(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        self.logger.info("Difficulty of estimation: {}".format(self.estimate_difficulty()))


class Object3d_HMC_v2(object):
    """ 3d object label """

    def __init__(self, gt_class_label, gt_box):

        # extract label, truncation, occlusion
        self.type = gt_class_label  # 'Car', 'Pedestrian', ...
        self.truncation = 0.  # truncated pixel ratio [0..1]
        self.occlusion = 0
        
        # extract 3d bounding box information
        self.l = gt_box[3]  # box length (in meters)
        self.w = gt_box[4]  # box width
        self.h = gt_box[5]  # box height
        
        # self.t = (gt_box[0], gt_box[1], gt_box[2]+self.h/2)  # location (x,y,z) in camera coord.
        self.t = (gt_box[0], gt_box[1], gt_box[2])  # location (x,y,z) in camera coord.
        self.ry = gt_box[6] #+math.pi/2  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]



class Detection_utils(object):
    '''
    ### OD 
    ### List up all objects or class information in a scene with checkbox
    ### With the checkbox gui, objects or point cloud of the certain class can be shown and hidden
    ### And all objects or point cloud are indexed and theirs colormaps are also shown in the list
    '''

    def __init__(self, parent):
        self.parent = parent
        self._scene = parent._scene
        self.cfg = parent.cfg
        self.logger = self.cfg.logger
        self.root = ''
        self.save_root = ''

        self._is_first_plot = True
        
        self.window = parent.window
        self.info_title = parent.info_title
        self.info = parent.info
        self.data_info = parent.data_info

        em = parent.window.theme.font_size
        separation_height = int(round(0.5 * em))
        self.settings = parent.settings
        self.fov = parent.fov
        self.point_size = parent.settings.point_size
        self.line_width = parent.settings.line_width

        ### nuScenes dataset
        self.class_list=['CAR', 'TRUCK', 'BUS', 'CONSTRUCTION_VEHICLE', 'TRAILER', 'BARRIER', 'MOTORCYCLE',
                         'BICYCLE', 'PEDESTRIAN', 'TRAFFIC_CONE']
        ### HMC dataset
        # self.class_list=['CAR', 'TRUCK', 'BUS', 'UNKNOWN_VEHICLE', 'UNKNOWN_OBJECT', 
        #                  'TUBULAR_MARKER', 'PTW','CYCLIST', 'PEDESTRIAN', 'RUBBER_CONE']
        
        self._max_frame = 0


        self._is_topview = False

        self.dataset = {}
        self.DL_cfg = {}
        self.DL_model = []

        self.pcd = o3d.geometry.PointCloud()

        self.pred_results = {}
        self.pred_bbox_pcd = o3d.geometry.PointCloud()
        self.dataset_lidar_pcd = o3d.geometry.PointCloud()
        self.dataset_radar_pcd = o3d.geometry.PointCloud()
        self.dataset_imgs = []

        self.next_flag = 0
        self.g_plane = rendering.Scene.GroundPlane(1)

        self.geometry_strings = []
        self.before_geometry_strings = []

        self._animation_delay_secs = 0.001
        self._animation_frames = []
        self._data_load_flag = False
        self.is_grid_show = False
        self.is_point_select_mode = False
        self.is_point_box_select_mode = False
        self._store_button = False
        self._prev_button_in_select = False
        self.selected_vertex_coords = [] 
        self.center = np.array([0,0,0])

        self.tab2 = gui.Vert()

        #######################################
        #######################################
        view_ctrls = gui.CollapsableVert("OD Data Loader", 0.25 * em,
                                        gui.Margins(em, 0, 0, 0))

        self.dataset_load_button = gui.Button("Load")
        self.dataset_load_button.horizontal_padding_em = 0.5
        self.dataset_load_button.vertical_padding_em = 0
        self.dataset_load_button.set_on_clicked(self._on_dataset_load_button_od)
        self.dataset_load_button.enabled = False
    
        self._fileedit_od = gui.TextEdit()
        self._seqname_od = gui.TextEdit()
        self._seqname_od.enabled = False

        jsonfileedit_layout = gui.Horiz()
        jsonfileedit_layout.add_child(gui.Label("Select Dataset    "))


        v_combo = gui.Vert()
        self.big_group_dataset_combo_od = gui.Combobox()
        self.big_group_dataset_combo_od.add_item('---')
        self.big_group_dataset_combo_od.add_item('Public Dataset')
        self.big_group_dataset_combo_od.add_item('HMC Dataset')
        self.big_group_dataset_combo_od.set_on_selection_changed(self._on_big_group_dataset_combo_changed)
        h = gui.Horiz(0.25 * em)
        # h.add_stretch()
        # h.add_fixed(4*em)
        h.add_child(self.big_group_dataset_combo_od)
        # h.add_fixed(8*em)
        h.add_stretch()
        v_combo.add_child(h)
        v_combo.add_fixed(0.5*em)

        self.detailed_group_dataset_combo_od = gui.Combobox()
        self.detailed_group_dataset_combo_od.add_item('---')
        self.detailed_group_dataset_combo_od.add_item('nuScenes')
        self.detailed_group_dataset_combo_od.add_item('HMC-v1')
        self.detailed_group_dataset_combo_od.set_on_selection_changed(self._on_detailed_group_dataset_combo_changed)
        h = gui.Horiz(0.25 * em)
        # h.add_stretch()
        # h.add_fixed(4*em)
        h.add_child(self.detailed_group_dataset_combo_od)
        # h.add_fixed(8*em)
        h.add_stretch()
        v_combo.add_child(h)


        jsonfileedit_layout.add_child(v_combo)


        jsonfileedit_layout.add_fixed(0.6 * em)
        jsonfileedit_layout.add_child(self.dataset_load_button)
        jsonfileedit_layout.add_fixed(0.6 * em)
        view_ctrls.add_child(jsonfileedit_layout)


        grid = gui.VGrid(5, 0.25 * em)
        view_ctrls.add_child(grid)


        grid = gui.VGrid(5, 0.25 * em)
        view_ctrls.add_child(grid)



        #######################################
        #######################################
        ### Score Threshold
        if self.cfg.pth_path is not None:
            view_crtls_score_th = gui.CollapsableVert("DL Model Custom Settings", 0.25 * em,
                                                gui.Margins(em, 0, 0.25*em, 0))
            
            score_th_combo_layout = gui.Horiz()
            score_th_combo_layout.add_child(gui.Label("Select Score_Threshold    "))
            self.score_th_combo_od = gui.Combobox()
            # self.score_th_combo_od.add_item('0.0')
            self.score_th_combo_od.add_item('0.1')
            self.score_th_combo_od.add_item('0.2')
            self.score_th_combo_od.add_item('0.3')
            self.score_th_combo_od.add_item('0.4')
            self.score_th_combo_od.add_item('0.5')
            self.score_th_combo_od.add_item('0.6')
            self.score_th_combo_od.add_item('0.7')
            self.score_th_combo_od.add_item('0.8')
            self.score_th_combo_od.add_item('0.9')
            self.score_th_combo_od.add_item('1.0')
            self.score_th_combo_od.set_on_selection_changed(self._on_score_th_combo_od_changed)
            h = gui.Horiz(0.25 * em)
            h.add_child(score_th_combo_layout)
            # h.add_stretch()
            h.add_fixed(4*em)
            h.add_child(self.score_th_combo_od)
            # h.add_fixed(8*em)
            h.add_stretch()
            view_crtls_score_th.add_child(h)
    

        #######################################
        #######################################
        ### Second view collaps pannel
        view_ctrls_plot = gui.CollapsableVert("OD Data Plot", 0.25 * em,
                                        gui.Margins(em, 0, 0, 0))


        self._slider_od = gui.Slider(gui.Slider.INT)
        self._slider_od.set_limits(1, 50)
        self._slider_od.enabled = False
        self._slider_od.set_on_value_changed(self._on_animation_slider_changed_od)
        
        h = gui.Horiz(0.5 * em)
        self._slider_label_od = gui.Label("   Index   ")
        # self._slider_label_od.text_color = gui.Color(0.2, 0.4, 0.9)
        h.add_child(self._slider_label_od)
        h.add_fixed(0.75*em)
        h.add_child(self._slider_od)
        h.add_stretch()  
        view_ctrls_plot.add_child(h)
        ############################################################


        
        self._dataplot_prev_button_od = gui.Button("Prev")
        self._dataplot_prev_button_od.enabled = True
        self._dataplot_prev_button_od.horizontal_padding_em = 5.
        self._dataplot_prev_button_od.vertical_padding_em = 0
        self._dataplot_prev_button_od.set_on_clicked(self._pressed_prev_button)

        # self._play_button_od = gui.Button("Play")
        # self._play_button_od.enabled = True
        # self._play_button_od.horizontal_padding_em = 2.
        # self._play_button_od.vertical_padding_em = 0
        # self._play_button_od.set_on_clicked(self._on_start_animation_od)

        self._dataplot_next_button_od = gui.Button("Next")
        self._dataplot_next_button_od.enabled = True
        self._dataplot_next_button_od.horizontal_padding_em = 5.
        self._dataplot_next_button_od.vertical_padding_em = 0
        self._dataplot_next_button_od.set_on_clicked(self._pressed_next_button)

        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._dataplot_prev_button_od)
        h.add_stretch()  
        # h.add_child(self._play_button_od)
        # h.add_stretch()  
        h.add_child(self._dataplot_next_button_od)
        h.add_stretch()  
        view_ctrls_plot.add_child(h)


        #######################################
        #######################################

        ### Second view collaps pannel
        view_crtls_imgs = gui.CollapsableVert("Multi-view Camera", 0.25 * em,
                                            gui.Margins(em, 0, 0.25*em, 0))
        self._image_viewer = gui.ImageWidget()

        v = gui.Vert(0.25 * em)
        v.add_child(gui.Label("RGB Images"))
        # h.add_stretch()
        v.add_fixed(0.25*em)
        v.add_child(self._image_viewer)
        # h.add_fixed(8*em)
        v.add_stretch()
        view_crtls_imgs.add_child(v)
    
        
        #######################################
        #######################################

        
        self.view_ctrls = view_ctrls
        self.view_ctrls_plot = view_ctrls_plot
        self.view_crtls_imgs = view_crtls_imgs

        self.tab2.add_fixed(separation_height)
        self.tab2.add_child(view_ctrls)

        if self.cfg.pth_path is not None:
            self.view_crtls_score_th = view_crtls_score_th
            self.selected_score_th = float(self.score_th_combo_od.get_item(0))
            self.tab2.add_fixed(separation_height)
            self.tab2.add_child(view_crtls_score_th)

        self.tab2.add_fixed(separation_height)
        self.tab2.add_child(view_ctrls_plot)
        self.tab2.add_fixed(separation_height)
        self.tab2.add_child(view_crtls_imgs)

        if platform.system() == "Linux":
            self.show_param_od = False

    def update(self, parent):
        try:
            self._settings_panel = parent._settings_panel
            
        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)

            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')



    #########################################
         #### Base Callback Function ####
    #########################################
    def draw_gt_3dbox(self, obj):
        '''
        ### OD 
        ### 3D GT Box generation based on json data
        ### Modified from KITTIVIEWER (second)
        '''
        try:
            # for obj in labels:
            obj_class = obj.type
            rotation = obj.ry
            tr = obj.t
            l = obj.l
            w = obj.w
            h = obj.h
            occl = obj.occlusion
            trunc = obj.truncation
            

            ############################
            ### with Open3D ###
            ############################

            center = tr # gt_boxes[0:3]
            lwh = [l, w, h] #gt_boxes[3:6]
            axis_angles = np.array([0, 0, rotation + 1e-10])
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
            corner_box = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(corner_box)

            heading_line = [[0.0, lwh[0]],
                            [0.0, 0.0],
                            [0.0, 0.0]]
                            #[h/2, h/2]]
            heading_line = np.dot(rot, heading_line)
            center_repeated = np.array(center).repeat(2).reshape(3,2)
            heading_line += center_repeated

            return corner_box, heading_line.transpose(), obj_class, center, lwh, rot #, objbox


        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')


    ##########################################
    #### First Section: Callback Function ####
    ##########################################
    def _on_score_th_combo_od_changed(self, name, index):
        try:
            self.logger.info('Changed score_threshold to %s', name)
            self.selected_score_th = float(name)
            # self._set_plot_od(self.plot_idx)

            ## remove pred_bboxes instances in the scene
            def isStartWithF(x):
                """x값이 pred_로 시작하면 true를 리턴한다 그렇지 않으면 false"""
                return x.startswith('pred_')
            filtered_pred_bboxes = list(filter(isStartWithF, self.geometry_strings))
            # print('-----------------------')
            # print(self.geometry_strings)
            # print('-----------------------')
            # print(filtered_pred_bboxes)
            # print('-----------------------')
            for orig_pred_ind in range(len(filtered_pred_bboxes)):
                self._scene.scene.remove_geometry(filtered_pred_bboxes[orig_pred_ind])
                self.geometry_strings.remove(filtered_pred_bboxes[orig_pred_ind])
            # self.geometry_strings = list(set(self.geometry_strings) - set(filtered_pred_bboxes))

            ## pred_results
            Lines2 = [[0,1]]
            filtered_by_score = self.pred_results[0]['pred_scores'] > self.selected_score_th
            pred_bboxes_3d = self.pred_results[0]['pred_boxes'][filtered_by_score].cpu().numpy()
            pred_labels_3d = self.pred_results[0]['pred_labels'][filtered_by_score].cpu().numpy()

            # filtered_by_score = self.pred_results[0].pred_instances_3d.scores_3d.cpu().numpy() > self.selected_score_th
            # pred_bboxes_3d = self.pred_results[0].pred_instances_3d.bboxes_3d.tensor.cpu().numpy()[filtered_by_score]
            # pred_labels_3d = self.pred_results[0].pred_instances_3d.labels_3d.cpu().numpy()[filtered_by_score]
            for pred_ind in range(len(pred_labels_3d)):
                pred_label = pred_labels_3d[pred_ind]
                # gt_bbox = gt_bboxes_3d[gt_ind].cpu().numpy()[0]
                pred_bbox = pred_bboxes_3d[pred_ind,:]
                pred_labels_dict = Object3d_HMC_v2(pred_label, pred_bbox)
                pred_line_set2 = o3d.geometry.LineSet()
                # bbox, orientation, obj_label, obj_occl, obj_trunc, dim, rotation_matrix = self.draw_gt_3dbox(labels_dict)
                pred_bbox, pred_heading_line, pred_obj_label, orientation, dim, rotation_matrix = self.draw_gt_3dbox(pred_labels_dict)

                colormap = self.cfg.det_color_map[self.cfg.class_names[self.class_list[pred_obj_label-1]]]
                colormap = [0,0,255]

                # bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_info))
                pred_bbox.color = np.array(colormap, dtype=np.float64)/255.0

                pred_line_set2.points = o3d.utility.Vector3dVector(pred_heading_line)
                pred_line_set2.lines = o3d.utility.Vector2iVector(Lines2)
                # pred_line_set2.colors = o3d.utility.Vector3dVector(np.array([colormap], dtype=np.float64)/255.0)
                pred_line_set2.paint_uniform_color(np.clip(np.array(colormap)/255.0, 0.0, 1.0))

                pred_bbox2 = o3d.geometry.LineSet.create_from_oriented_bounding_box(pred_bbox)

                
                self._scene.scene.add_geometry('pred_bbox_'+str(pred_ind), pred_bbox2, self.settings.material_line)
                self._scene.scene.add_geometry('pred_bbox_heading_'+str(pred_ind), pred_line_set2, self.settings.material_line)
                self.geometry_strings.append('pred_bbox_'+str(pred_ind))
                self.geometry_strings.append('pred_bbox_heading_'+str(pred_ind))


            self._scene.force_redraw()

                    
        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')



    def _on_big_group_dataset_combo_changed(self, name, index):
        try:
            # print('combobox changed')
            if name == 'Public Dataset':
                self.detailed_group_dataset_combo_od.clear_items()
                self.detailed_group_dataset_combo_od.add_item('---')
                self.detailed_group_dataset_combo_od.add_item('nuScenes')
                # self.detailed_group_dataset_combo_od.change_item(1, 'nuScenes')

            elif name == 'HMC Dataset':
                self.detailed_group_dataset_combo_od.clear_items()
                self.detailed_group_dataset_combo_od.add_item('---')
                self.detailed_group_dataset_combo_od.add_item('HMC-v1')
                # self.detailed_group_dataset_combo_od.change_item(1, 'HMC-v1')
                    
        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')


    def _on_detailed_group_dataset_combo_changed(self, name, index):
        try:
            # print('combobox changed')
            if name != '---':
                self.dataset_load_button.enabled = True

            else:
                self.dataset_load_button.enabled = False
                    
        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')


    def init_DL_model(self, 
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',):
        
        self.DL_model = build_network(model_cfg=self.DL_cfg.model)
        # self.DL_model = MODELS.build(self.DL_cfg.model)
        if checkpoint is not None:
            self.DL_model.load_params_from_file(filename=checkpoint, to_cpu=False, logger=None)
            # checkpoint = load_checkpoint(self.DL_model, checkpoint, map_location='cpu')
            # # save the dataset_meta in the model for convenience
            # if 'dataset_meta' in checkpoint.get('meta', {}):
            #     # mmdet3d 1.x
            #     self.DL_model.dataset_meta = checkpoint['meta']['dataset_meta']
            # elif 'CLASSES' in checkpoint.get('meta', {}):
            #     # < mmdet3d 1.x
            #     classes = checkpoint['meta']['CLASSES']
            #     self.DL_model.dataset_meta = {'classes': classes}

            #     if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
            #         self.DL_model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
            # else:
            #     # < mmdet3d 1.x
            #     self.DL_model.dataset_meta = {'classes': self.DL_cfg.class_names}

            #     if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
            #         self.DL_model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']

            # test_dataset_cfg = deepcopy(self.DL_cfg.test_dataloader.dataset)
            # # lazy init. We only need the metainfo.
            # test_dataset_cfg['lazy_init'] = True
            # metainfo = DATASETS.build(test_dataset_cfg).metainfo
            # cfg_palette = metainfo.get('palette', None)
            # if cfg_palette is not None:
            #     self.DL_model.dataset_meta['palette'] = cfg_palette
            # else:
            #     if 'palette' not in self.DL_model.dataset_meta:
            #         warnings.warn(
            #             'palette does not exist, random is used by default. '
            #             'You can also set the palette to customize.')
            #         self.DL_model.dataset_meta['palette'] = 'random'

        self.DL_model.cfg = self.DL_cfg  # save the config in the model for convenience
        if device != 'cpu':
            torch.cuda.set_device(device)
        else:
            warnings.warn('Don\'t suggest using CPU device. '
                        'Some functions are not supported for now.')

        self.DL_model.to(device)
        self.DL_model.eval()
        return self.DL_model

    def load_dataset(self):
        try:
            # config_path = 'projects/HMCBEV/configs/bevfusion_lidar-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
            # config_path = 'projects/HMCBEV/configs/bevfusion_large_radar.py'
            config_path = self.parent.cfg.cfg_path

            self.DL_cfg = load_config_file(config_path).exp_config
            # self.DL_cfg = Config.fromfile(config_path)

            # init_default_scope(self.DL_cfg.get('default_scope', 'mmdet3d'))

            print('===============================================')

            # # dataset_cfg = deepcopy(cfg.train_dataloader.dataset)
            # dataset_cfg = deepcopy(config_path.data_config)

            # ### With the number of frames information of the loaded dataset,
            # ### 1. update the progress bar to fit with the whole number of frames
            # ### 2. plot the data (camera images, radar point cloud, and lidar point cloud) selected by frame index
            # ### 3. as an option, sweeps information should be included in the setting panel
            # '''
            # for i, item in enumerate(dataset):
            #     # the 3D Boxes in input could be in any of three coordinates
            #     data_input = item['inputs']
            #     data_sample = item['data_samples'].numpy()

            #     data_input['img']           ### shape: 6 images * 3 channel RGB * height * width
            #     data_input['points']        ### shape: number of points * 5 (x, y, z, intensity, ) --> dont know the rest one
            #     data_input['radar_points']  ### shaep: number of points * 7 (x, y, z, doppler, ) --> dont know the rest ones..
            # '''
            # self.dataset = DATASETS.build(dataset_cfg)

            train_set, train_loader, train_sampler = build_dataloader(
                dataset_cfg=self.DL_cfg.data_config,
                class_names=self.DL_cfg.class_names,
                batch_size=2,
                dist=False, workers=self.DL_cfg.OPTIMIZATION['NUM_WORKERS'],
                logger=None,
                training=False,
                merge_all_iters_to_one_epoch=False,
                total_epochs=self.DL_cfg.OPTIMIZATION['NUM_EPOCHS'],
                seed=666 
            )
            print('===============================================')

            self.dataset = train_loader.dataset
            # dataloader_iter = iter(train_loader)
            # batch = next(dataloader_iter)

            if self.parent.cfg.pth_path is not None:
                # checkpoint_pth_file = '/VDI_BIGDATA2/ADSTD/Personal/junlee/HMCBEV_models/work_dirs/bevfusion_lidar-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_largeradar/epoch_20.pth'
                checkpoint_pth_file = self.parent.cfg.pth_path
                self.DL_model = self.init_DL_model(checkpoint_pth_file)

            ### set slider length to fit with the number of frames in the dataset
            self._max_frame = len(self.dataset)
            self._slider_od.set_limits(1, self._max_frame)

            self._data_load_flag = True
            self._slider_od.enabled = True

            ### Frame index to visualize
            ### This index is the initial value (ex. frame index = 853)
            ### And this index will be changed 
            ###         if users press 'left/right' button of the keyboard or 'next/prev' gui button
            self._slider_od.int_value = 23 + 1 
            self._set_plot_od(23)
            
        
            
        except Exception as err:
            self._data_load_flag = False
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            error_strs.append('Please contact the developer of this tool.' + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')

    def _on_dataset_load_button_od(self):
        try:
            self.logger.info('dataset load')
            self.logger.info(self.big_group_dataset_combo_od.selected_text)
            self.logger.info(self.detailed_group_dataset_combo_od.selected_text)

            def update_info_text():
                text = "Loading Dataset......"
                self.info_title.text = text
                self.window.set_needs_layout()

            gui.Application.instance.post_to_main_thread(self.window, update_info_text)
            self._scene.force_redraw()

            self.parent.loading_sign = True
            
        except Exception as err:
            self._data_load_flag = False
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            error_strs.append('Please contact the developer of this tool.' + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')



    ##########################################
    #### First Section: Plot Function ####
    ##########################################
    #### 2. plot and play & stop ##########
    def _set_plot_od(self, ind):
        '''
        ### OD 
        ### Plot and Visualize selected frame with GT boxes
        '''
        try:
            ## point cloud --> self.points

            ### for image plot index
            self.plot_idx = ind
            
            ### init scene
            self._scene.scene.clear_geometry()

            
            if len(self.geometry_strings) > 0:
                self.before_geometry_strings = self.geometry_strings
                self.geometry_strings.clear()

            # self.parent.tabs
            if len(self.parent.axis_label) > 0:
                for x, y in zip(self.parent.axis_label[0], self.parent.axis_label[1]):
                    self._scene.remove_3d_label(x)
                    self._scene.remove_3d_label(y)
            if len(self.parent.obj_index_label) > 0:
                for obj_label in self.parent.obj_index_label:
                    self._scene.remove_3d_label(obj_label)
            if len(self.parent.cad_index_label) > 0:
                for cad_label in self.parent.cad_index_label:
                    self._scene.remove_3d_label(cad_label)

            self.is_grid_show, self.parent.axis_label = _draw_coordinate_plot(self._scene, self.g_plane, 1)

            ### for training dataset with augmentation
            ### whenever calling self.dataset[ind], augmentation is applied
            ### such as, 
            # #### augmentation functions are called
            # points = self.dataset[ind]['inputs']['points'] 
            # #### augmentation functions are called
            # gt_labels_3d = self.dataset[ind]'data_samples'].gt_instances_3d.bboxes_3d.numpy() 
            ### For this reason, points data and gt bboxes are not matched!!!

            current_frame_data = self.dataset.collate_batch([self.dataset[ind]]).copy()
            viz_data = deepcopy(current_frame_data)

            if self.parent.cfg.pth_path is not None:
                with torch.no_grad():
                    self.DL_model.cuda()
                    self.DL_model.eval()
                    current_frame_data.pop('points')
                    load_data_to_gpu(current_frame_data)
                    load_data_to_gpu(viz_data)
                    pred_dicts, _ = self.DL_model.forward(current_frame_data)

                    # self.dataset
                    # class_names = self.dataset.class_names

                    self.pred_results = pred_dicts
                    # self.pred_results = self.dataset.generate_prediction_dicts(
                    #                         current_frame_data, pred_dicts, class_names,
                    #                         output_path=None
                    #                     )

            #     collate_data = pseudo_collate([current_frame_data])
            #     # forward the model
            #     with torch.no_grad():
            #         pred_results = self.DL_model.test_step(collate_data)
            #         self.pred_results = pred_results

            ### gt_bboxes_3d: B * N * 9 shape (c_x, c_y, c_z, l, w, h, yaw_angle, vel_x, vel_y)
            ### gt_labels_3d: B * N  shape 
            if self.parent.cfg.pth_path is not None:

                gt_bboxes_3d = viz_data['gt_boxes'][0,:,:9].cpu().numpy()
                gt_labels_3d = viz_data['gt_boxes'][0,:,-1].cpu().numpy()
            else:
                gt_bboxes_3d = viz_data['gt_boxes'][0,:,:9]
                gt_labels_3d = viz_data['gt_boxes'][0,:,-1]
            # gt_labels_3d = current_frame_data['gt_names']

            if 'points' in viz_data:
                if self.parent.cfg.pth_path is not None:
                    ### first column refers batch_index (0, 1, 2~~)
                    loaded_lidar_points = viz_data['points'].cpu().numpy()[:,1:]
                else:
                    loaded_lidar_points = viz_data['points'][:,1:]
            else:
                loaded_lidar_points = np.zeros(0)

            if 'radar_points' in viz_data:
                if self.parent.cfg.pth_path is not None:
                    ### first column refers batch_index (0, 1, 2~~)
                    loaded_radar_points = viz_data['radar_points'].cpu().numpy()[:,1:]
                else:
                    loaded_radar_points = viz_data['radar_points'][:,1:]
            else:
                loaded_radar_points = np.zeros(0)
            # try:    ### for train dataset 
            #     gt_labels_3d = current_frame_data['data_samples'].gt_instances_3d.labels_3d.numpy()
            #     gt_bboxes_3d = current_frame_data['data_samples'].gt_instances_3d.bboxes_3d.numpy()
            # except: ### for validation/test dataset to evaluate
            #     gt_labels_3d = current_frame_data['data_samples'].eval_ann_info['gt_labels_3d']
            #     gt_bboxes_3d = current_frame_data['data_samples'].eval_ann_info['gt_bboxes_3d'].tensor.numpy()
            
            # loaded_lidar_points = current_frame_data['inputs']['points']         ### x, y, z, intensity, zero-padding (in mmdet3d)
            # loaded_radar_points = current_frame_data['inputs']['radar_points']   ### x, y, z, rcs, vxcomp(relative vel in x axis), vycomp(relative vel in y axis), t
            #                                                                     ### these 7 features are selected in 'Loadnuradarpoints' from 18 attributes of nuscenes radar point cloud
            if 'camera_imgs' in viz_data:
                loaded_images = viz_data['camera_imgs']                  ### 6 * 3 * h * w  --> should be changed to 3w * 2h * 3
                if self.parent.cfg.pth_path is not None:
                    img_1F = loaded_images[0,0,:,:,:].cpu().permute(1,2,0).numpy()
                    img_1R = loaded_images[0,1,:,:,:].cpu().permute(1,2,0).numpy()
                    img_1L = loaded_images[0,2,:,:,:].cpu().permute(1,2,0).numpy()
                    img_2B = loaded_images[0,3,:,:,:].cpu().permute(1,2,0).numpy()
                    img_2R = loaded_images[0,4,:,:,:].cpu().permute(1,2,0).numpy()
                    img_2L = loaded_images[0,5,:,:,:].cpu().permute(1,2,0).numpy()
                else:
                    # B, N, C, H, W = loaded_images.size()
                    img_1F = loaded_images[0,0,:,:,:].permute(1,2,0).numpy()
                    img_1R = loaded_images[0,1,:,:,:].permute(1,2,0).numpy()
                    img_1L = loaded_images[0,2,:,:,:].permute(1,2,0).numpy()
                    img_2B = loaded_images[0,3,:,:,:].permute(1,2,0).numpy()
                    img_2R = loaded_images[0,4,:,:,:].permute(1,2,0).numpy()
                    img_2L = loaded_images[0,5,:,:,:].permute(1,2,0).numpy()

                # img_1F = loaded_images[0,:,:,:].permute(1,2,0).numpy()
                # img_1R = loaded_images[1,:,:,:].permute(1,2,0).numpy()
                # img_1L = loaded_images[2,:,:,:].permute(1,2,0).numpy()
                # img_2B = loaded_images[3,:,:,:].permute(1,2,0).numpy()
                # img_2R = loaded_images[4,:,:,:].permute(1,2,0).numpy()
                # img_2L = loaded_images[5,:,:,:].permute(1,2,0).numpy()
                
                front_imgs = np.concatenate((img_1L, img_1F, img_1R), axis=1)
                rear_imgs = np.concatenate((img_2L, img_2B, img_2R), axis=1)

                one_big_img = np.concatenate((front_imgs,rear_imgs), axis=0)
                ##### update images
                o3d_images = o3d.geometry.Image(np.ascontiguousarray(one_big_img*255.).astype(np.uint8)) ## it takes h * w * 3 shape
                self._image_viewer.update_image(o3d_images)


            ##### update LiDAR & Radar point clouds and labels 
            
            if self.DL_cfg.data_config['input_modality']['use_lidar']:
                lidadr_colors = np.repeat(np.array([[180,180,180]],dtype=np.float64), loaded_lidar_points.shape[0], axis=0)
                self.dataset_lidar_pcd.points = o3d.utility.Vector3dVector(loaded_lidar_points[:,:3])
                self.dataset_lidar_pcd.colors = o3d.utility.Vector3dVector(lidadr_colors/255.0)
                self._scene.scene.add_geometry('LIDAR_PC', self.dataset_lidar_pcd, self.settings.material)
                self.geometry_strings.append('LIDAR_PC')

            if self.DL_cfg.data_config['input_modality']['use_radar']:
                radar_colors = np.repeat(np.array([[50,180,100]],dtype=np.float64), loaded_radar_points.shape[0], axis=0)
                self.dataset_radar_pcd.points = o3d.utility.Vector3dVector(loaded_radar_points[:,:3])
                self.dataset_radar_pcd.colors = o3d.utility.Vector3dVector(radar_colors/255.0)
                self._scene.scene.add_geometry('RADAR_PC', self.dataset_radar_pcd, self.settings.material_selected)
                self.geometry_strings.append('RADAR_PC')

            Lines2 = [[0,1]]
            for gt_ind in range(len(gt_labels_3d)):
                gt_label = gt_labels_3d[gt_ind]
                # gt_bbox = gt_bboxes_3d[gt_ind].cpu().numpy()[0]
                gt_bbox = gt_bboxes_3d[gt_ind,:]
                labels_dict = Object3d_HMC_v2(gt_label, gt_bbox)
                line_set2 = o3d.geometry.LineSet()
                # bbox, orientation, obj_label, obj_occl, obj_trunc, dim, rotation_matrix = self.draw_gt_3dbox(labels_dict)
                
                bbox, heading_line, obj_label, orientation, dim, rotation_matrix = self.draw_gt_3dbox(labels_dict)
                obj_name = self.DL_cfg.class_names[int(obj_label)-1].upper()
                colormap = self.cfg.det_color_map[self.cfg.class_names[obj_name]]

                # bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_info))
                bbox.color = np.array(colormap, dtype=np.float64)/255.0

                line_set2.points = o3d.utility.Vector3dVector(heading_line)
                line_set2.lines = o3d.utility.Vector2iVector(Lines2)
                # line_set2.colors = o3d.utility.Vector3dVector(np.array([colormap], dtype=np.float64)/255.0)
                line_set2.paint_uniform_color(np.clip(np.array(colormap)/255.0, 0.0, 1.0))

                bbox2 = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)

                
                self._scene.scene.add_geometry('bbox_'+str(gt_ind), bbox2, self.settings.material_line)
                self._scene.scene.add_geometry('bbox_heading_'+str(gt_ind), line_set2, self.settings.material_line)
                self.geometry_strings.append('bbox_'+str(gt_ind))
                self.geometry_strings.append('bbox_heading_'+str(gt_ind))


            ## pred_results
            if self.parent.cfg.pth_path is not None:
                filtered_by_score = self.pred_results[0]['pred_scores'] > self.selected_score_th
                pred_bboxes_3d = self.pred_results[0]['pred_boxes'][filtered_by_score].cpu().numpy()
                pred_labels_3d = self.pred_results[0]['pred_labels'][filtered_by_score].cpu().numpy()
                for pred_ind in range(len(pred_labels_3d)):
                    pred_label = pred_labels_3d[pred_ind]
                    # gt_bbox = gt_bboxes_3d[gt_ind].cpu().numpy()[0]
                    pred_bbox = pred_bboxes_3d[pred_ind,:]
                    pred_labels_dict = Object3d_HMC_v2(pred_label, pred_bbox)
                    pred_line_set2 = o3d.geometry.LineSet()
                    # bbox, orientation, obj_label, obj_occl, obj_trunc, dim, rotation_matrix = self.draw_gt_3dbox(labels_dict)
                    pred_bbox, pred_heading_line, pred_obj_label, orientation, dim, rotation_matrix = self.draw_gt_3dbox(pred_labels_dict)

                    colormap = self.cfg.det_color_map[self.cfg.class_names[self.class_list[pred_obj_label-1]]]
                    colormap = [0,0,255]

                    # bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_info))
                    pred_bbox.color = np.array(colormap, dtype=np.float64)/255.0

                    pred_line_set2.points = o3d.utility.Vector3dVector(pred_heading_line)
                    pred_line_set2.lines = o3d.utility.Vector2iVector(Lines2)
                    # pred_line_set2.colors = o3d.utility.Vector3dVector(np.array([colormap], dtype=np.float64)/255.0)
                    pred_line_set2.paint_uniform_color(np.clip(np.array(colormap)/255.0, 0.0, 1.0))

                    pred_bbox2 = o3d.geometry.LineSet.create_from_oriented_bounding_box(pred_bbox)

                    
                    self._scene.scene.add_geometry('pred_bbox_'+str(pred_ind), pred_bbox2, self.settings.material_line)
                    self._scene.scene.add_geometry('pred_bbox_heading_'+str(pred_ind), pred_line_set2, self.settings.material_line)
                    self.geometry_strings.append('pred_bbox_'+str(pred_ind))
                    self.geometry_strings.append('pred_bbox_heading_'+str(pred_ind))


            self._scene.force_redraw()
            
            def update_data_info_label():
                text = "#SWEEPS LIDAR: " + str(self.DL_cfg.data_config['sweeps_nums']['LIDAR_SWEEPS']-1) + \
                        "   |   #SWEEPS RADAR: " +  str(self.DL_cfg.data_config['sweeps_nums']['RADAR_SWEEPS']-1) + '\n' + \
                        "#points(LIDAR): " + str(loaded_lidar_points.shape[0]) + \
                        "   |   #points(RADAR): " +  str(loaded_radar_points.shape[0])
                self.data_info.text = text
                
                self.window.set_needs_layout()


            # This is not called on the main thread, so we need to
            # post to the main thread to safely access UI items.
            def update_label():    
                text = "Viewer Mode : " + str(ind) + '\'th Frame'
                self.info_title.text = text
                self.window.set_needs_layout()

            gui.Application.instance.post_to_main_thread(self.window, update_label)    
            gui.Application.instance.post_to_main_thread(self.window, update_data_info_label)
            self._scene.force_redraw()
            
            if self._is_first_plot:

                self._step_max = [6.0 , 20] # _step, _step_rot
                self._step_min = [0.6 , 2]
                self._step = 3.0
                self._step_rot = 10

                ### Camera Setting: Front View
                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                self.center = np.array([0.,0.,0.])
                self._scene.setup_camera(self.fov, bounds, self.center)    #### FoV, model_bounds, center; intrinsic, extrinsic, model_bounds; int, ext, int_width_px, int_hjeight_px, model_bounds
                # self._scene.look_at(self.center, self.center - [+150, 0, -80], [0, 0, 1])  ## original
                self._scene.look_at(self.center, self.center - [0, +120, -100], [0, 0, 1])   ## changed view point
                #### center, eye, up: Camera is located at 'eye', pointing towards 'center', oriented up vector = 'up'
                self._is_topview = False

                def press_on_viewer(ev):
                    try:
                        if ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.T:
                            bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                            self.center = np.array([0.,0.,0.])
                            eyemat = self._scene.scene.camera.get_view_matrix()[:3,3]
                            if int(np.abs(top_eyeview[2])) != int(np.abs(eyemat[2])):
                                try:
                                    top_eyeview[2] = eyemat[2]*2
                                except:
                                    top_eyeview[2] = -80
                            self._scene.setup_camera(self.fov, bounds, self.center)
                            self._scene.look_at(self.center, self.center - top_eyeview, [0,0,1])
                            self._is_topview = True

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.R:
                            # self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
                            bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                            self.center = np.array([0.,0.,0.])
                            self._scene.setup_camera(self.fov, bounds, self.center)    #### FoV, model_bounds, center; intrinsic, extrinsic, model_bounds; int, ext, int_width_px, int_hjeight_px, model_bounds
                            # self._scene.look_at(self.center, self.center - [-150, 0, -80], [0, 0, 1])  #### center, eye, up: Camera is located at 'eye', pointing towards 'center', oriented up vector = 'up'
                            self._scene.look_at(self.center, self.center - [0, -120, -100], [0, 0, 1])  #### center, eye, up: Camera is located at 'eye', pointing towards 'center', oriented up vector = 'up'
                            self._is_topview = False
                            
                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.F : #and (self.is_alt_pressed == 0):
                            # self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
                            bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                            self.center = np.array([0.,0.,0.])
                            self._scene.setup_camera(self.fov, bounds, self.center)    #### FoV, model_bounds, center; intrinsic, extrinsic, model_bounds; int, ext, int_width_px, int_hjeight_px, model_bounds
                            # self._scene.look_at(self.center, self.center - [+150, 0, -80], [0, 0, 1])  ## original
                            self._scene.look_at(self.center, self.center - [0, +120, -100], [0, 0, 1])   ## changed view point
                            #### center, eye, up: Camera is located at 'eye', pointing towards 'center', oriented up vector = 'up'
                            self._is_topview = False
                        
                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.W:
                            if self._is_topview :
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                eyemat = self._scene.scene.camera.get_view_matrix()[:3,3]
                                top_eyeview[2] = int(eyemat[2])
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                self.center[0] += 1
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, self.center - top_eyeview, [0,0,1])
                            else:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                
                                # step 1 : camera coordination
                                cam_coord = self._scene.scene.camera.get_model_matrix()[:3,3]

                                # step 2 : xy-plane unit vector 
                                _tmp_arr = np.array((self.center[0] - cam_coord[0], self.center[1] - cam_coord[1]))
                                _tmp_dist = np.linalg.norm(_tmp_arr)
                                
                                # step 3 : delta_x, delta_y
                                delta_x = (self.center[0] - cam_coord[0]) / _tmp_dist
                                delta_y = (self.center[1] - cam_coord[1]) / _tmp_dist

                                # step 4 : center move, cam move
                                self.center[0] += delta_x * self._step
                                self.center[1] += delta_y * self._step
                                cam_coord[0] += delta_x * self._step
                                cam_coord[1] += delta_y * self._step

                                # step 5 : camera setup
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, cam_coord, [0,0,1])

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.S:
                            # if self.is_ctrl_pressed == 0:
                            if self._is_topview:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                eyemat = self._scene.scene.camera.get_view_matrix()[:3,3]
                                top_eyeview[2] = int(eyemat[2])
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                self.center[0] -= 1
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, self.center - top_eyeview, [0,0,1])
                            else:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                
                                # step 1 : camera coordination
                                cam_coord = self._scene.scene.camera.get_model_matrix()[:3,3]

                                # step 2 : xy-plane unit vector 
                                _tmp_arr = np.array((self.center[0] - cam_coord[0], self.center[1] - cam_coord[1]))
                                _tmp_dist = np.linalg.norm(_tmp_arr)
                                
                                # step 3 : delta_x, delta_y
                                delta_x = (self.center[0] - cam_coord[0]) / _tmp_dist
                                delta_y = (self.center[1] - cam_coord[1]) / _tmp_dist

                                # step 4 : center move, cam move
                                self.center[0] -= delta_x * self._step
                                self.center[1] -= delta_y * self._step
                                cam_coord[0] -= delta_x * self._step
                                cam_coord[1] -= delta_y * self._step

                                # step 5 : camera setup
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, cam_coord, [0,0,1])

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.A:
                            if self._is_topview:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                eyemat = self._scene.scene.camera.get_view_matrix()[:3,3]
                                top_eyeview[2] = int(eyemat[2])
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                self.center[1] += 1
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, self.center - top_eyeview, [0,0,1])
                            else:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                
                                # step 1 : camera coordination
                                cam_coord = self._scene.scene.camera.get_model_matrix()[:3,3]

                                # step 2 : xy-plane unit vector 
                                _tmp_arr = np.array((self.center[0] - cam_coord[0], self.center[1] - cam_coord[1]))
                                _tmp_dist = np.linalg.norm(_tmp_arr)
                                
                                # step 3 : delta_x, delta_y
                                delta_x = (self.center[0] - cam_coord[0]) / _tmp_dist
                                delta_y = (self.center[1] - cam_coord[1]) / _tmp_dist

                                # step 4 : center move, cam move
                                self.center[0] -= delta_y * self._step
                                self.center[1] += delta_x * self._step
                                cam_coord[0] -= delta_y * self._step
                                cam_coord[1] += delta_x * self._step

                                # step 5 : camera setup
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, cam_coord, [0,0,1])

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.D: # and (self.is_alt_pressed == 0):
                            if self._is_topview:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                eyemat = self._scene.scene.camera.get_view_matrix()[:3,3]
                                top_eyeview[2] = int(eyemat[2])
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                self.center[1] -= 1
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, self.center - top_eyeview, [0,0,1])
                            else:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                
                                # step 1 : camera coordination
                                cam_coord = self._scene.scene.camera.get_model_matrix()[:3,3]
                                
                                # step 2 : xy-plane unit vector 
                                _tmp_arr = np.array((self.center[0] - cam_coord[0], self.center[1] - cam_coord[1]))
                                _tmp_dist = np.linalg.norm(_tmp_arr)

                                # step 3 : delta_x, delta_y
                                delta_x = (self.center[0] - cam_coord[0]) / _tmp_dist
                                delta_y = (self.center[1] - cam_coord[1]) / _tmp_dist

                                # step 4 : center move, cam move
                                self.center[0] += delta_y * self._step
                                self.center[1] -= delta_x * self._step
                                cam_coord[0] += delta_y * self._step
                                cam_coord[1] -= delta_x * self._step

                                # step 5 : camera setup
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, cam_coord, [0,0,1])

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.Q:
                            if self._is_topview:
                                pass
                            else:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                
                                # step 1 : camera coordination
                                cam_coord = self._scene.scene.camera.get_model_matrix()[:3,3]

                                # step 2 : camera rotation
                                _tmp_coord = cam_coord - self.center
                                _thta = (np.pi)/180 * self._step_rot
                                cam_coord[0] = np.cos(-_thta) * _tmp_coord[0] - np.sin(-_thta) *_tmp_coord[1] + self.center[0]
                                cam_coord[1] = np.sin(-_thta) * _tmp_coord[0] + np.cos(-_thta) *_tmp_coord[1] + self.center[1]

                                # step 3 : camera setup
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, cam_coord, [0,0,1])

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.E:
                            if self._is_topview:
                                pass
                            else:
                                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                                if np.sum(self.center - np.array(self._scene.center_of_rotation)) != 0.0:
                                    self.center = np.array(self._scene.center_of_rotation)
                                
                                # step 1 : camera coordination
                                cam_coord = self._scene.scene.camera.get_model_matrix()[:3,3]

                                # step 2 : camera rotation
                                _tmp_coord = cam_coord - self.center
                                _thta = (np.pi)/180 * self._step_rot
                                cam_coord[0] = np.cos(_thta) * _tmp_coord[0] - np.sin(_thta) *_tmp_coord[1] + self.center[0]
                                cam_coord[1] = np.sin(_thta) * _tmp_coord[0] + np.cos(_thta) *_tmp_coord[1] + self.center[1]

                                # step 3 : camera setup
                                self._scene.center_of_rotation = self.center
                                bounds = self.dataset_lidar_pcd.get_axis_aligned_bounding_box()
                                self._scene.setup_camera(self.fov, bounds, self.center)
                                self._scene.look_at(self.center, cam_coord, [0,0,1])

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.M:
                            self.parent._on_menu_toggle_settings_panel()
                        
                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.LEFT : 
                            if self._slider_od.int_value == 1:
                                self.logger.warning("No previous frame exist !!")
                                self.parent._message_dialog('INFO', ['No previous frame exist !!'])
                            else:
                                self._set_prev_plot_od()

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.RIGHT : 
                            if self._slider_od.int_value == self._max_frame:
                                self.logger.warning("No more frame exist !!")
                                self.parent._message_dialog('INFO', ['No more frame exist !!'])
                            else:
                                self._set_next_plot_od()

                        elif ev.type == gui.KeyEvent.DOWN and ( ev.key ==gui.KeyName.ESCAPE ): #ev.key ==gui.KeyName.Q or
                            # gui.Application.instance.quit()
                            # self.parent.stop_sign = True
                            # self.parent.thh.join()
                            # self.logger.info('FINISHED INFO TEXT THREADING')
                            self.window.close()

                        elif ev.type == gui.KeyEvent.DOWN and ev.key == gui.KeyName.G: # and (self.is_alt_pressed == 0):
                            self.is_grid_show = not self.is_grid_show
                            self._scene.scene.show_geometry("boundary", self.is_grid_show)
                            self._scene.scene.show_ground_plane(self.is_grid_show, self.g_plane)
                            if self.is_grid_show:
                                self.is_grid_show, self.parent.axis_label = _draw_coordinate_plot(self._scene, self.g_plane, 1)
                            else:
                                for x, y in zip(self.parent.axis_label[0], self.parent.axis_label[1]):
                                    self._scene.remove_3d_label(x)
                                    self._scene.remove_3d_label(y)
                                self._scene.scene.remove_geometry("boundary")

                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.EQUALS:

                            self.settings.material.point_size += 0.5
                            # self.settings.material_line.line_width += 0.5
                            # self.settings.material_insp.line_width += 0.5    

                            self.settings.material.point_size = np.max([self.settings.material.point_size, self.point_size[0]])
                            self.settings.material_selected.point_size = np.max([self.settings.material.point_size, self.point_size[1]])
                            self.settings.material_line.line_width = np.max([self.settings.material_line.line_width, self.line_width[0]])
                            self.settings.material_insp.line_width = np.max([self.settings.material_insp.line_width, self.line_width[1]])
                            self.settings.material_check_occl.line_width = np.max([self.settings.material_check_occl.line_width, self.line_width[1]])
                            
                            geometry_lists = list(set(self.geometry_strings))
                            for geometry_string_item in geometry_lists:
                                if geometry_string_item.startswith('RADAR_PC'):
                                    self._scene.scene.modify_geometry_material(geometry_string_item, self.settings.material_selected)
                                elif geometry_string_item.startswith('bbox'):
                                    self._scene.scene.modify_geometry_material(geometry_string_item, self.settings.material_line)
                                # elif geometry_string_item.startswith('INSP_BBOX'):
                                #     self._scene.scene.modify_geometry_material(geometry_string_item, self.settings.material_insp)
                                else:
                                    self._scene.scene.modify_geometry_material(geometry_string_item, self.settings.material)
                                    # print(geometry_string_item)

                            self.point_size[0] = self.settings.material.point_size
                            text = "Point Size Up : " + str(self.settings.material.point_size)
                            
                        elif ev.type == gui.KeyEvent.DOWN and ev.key ==gui.KeyName.MINUS:
                            self.settings.material.point_size -= 0.5
                            # self.settings.material_line.line_width -= 0.5
                            # self.settings.material_insp.line_width -= 0.5    

                            self.settings.material.point_size = np.min([self.settings.material.point_size, self.point_size[0]]) if self.settings.material.point_size >= 2.0 else 2.0
                            self.settings.material_selected.point_size = np.max([self.settings.material.point_size, self.point_size[1]])
                            # self.settings.material_line.line_width = np.min([self.settings.material_line.line_width, self.line_width[0]])
                            # self.settings.material_insp.line_width = np.min([self.settings.material_insp.line_width, self.line_width[1]])
                            
                            geometry_lists = list(set(self.geometry_strings))
                            for geometry_string_item in geometry_lists:
                                if geometry_string_item.startswith('RADAR_PC'):
                                    self._scene.scene.modify_geometry_material(geometry_string_item, self.settings.material_selected)
                                elif geometry_string_item.startswith('bbox'):
                                    self._scene.scene.modify_geometry_material(geometry_string_item, self.settings.material_line)
                                else:
                                    self._scene.scene.modify_geometry_material(geometry_string_item, self.settings.material)
                                    # print(geometry_string_item)

                            self.point_size[0] = self.settings.material.point_size
                        
                            text = "Point Size Down : " + str(self.settings.material.point_size)
                            
                        self._scene.force_redraw()
                        
                        return False
                    except Exception as err:
                        error_strs = []
                        error_strs.append('\n[ERROR OCCUR]')
                        error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
                        error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
                        error_strs.append('Error Type :' + str(type(err).__name__))
                        error_strs.append('Error message : ' + str(err) + '\n')
                        self.parent._message_dialog('ERROR', error_strs)
                        self.logger.error('\n[ERROR OCCUR]')
                        self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
                        self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
                        self.logger.error('Error Type :', type(err).__name__)
                        self.logger.error('Error message : ', err, '\n')
                self._scene.set_on_key(press_on_viewer)
                self._scene.set_on_mouse(self.click_on_viewer)
            else:
                self._is_first_plot = False

            if self.cfg.pth_path is not None:
                log_str = '[OD] Show PCD file ( ' + str(ind) + ' \'th frame ) @ score_th : ' + str(self.selected_score_th)
            else:
                log_str = '[OD] Show PCD file ( ' + str(ind) + ' \'th frame )'
            self.logger.info(log_str)

            self.prev_pcd_ind = ind
            # self._scene.force_redraw()
            
        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            error_strs.append('Please contact the developer of this tool.' + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')
        

    def click_on_viewer(self, event):
        # ref: http://www.open3d.org/docs/release/python_example/visualization/index.html#mouse-and-point-coord-py
        try:
            self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            self.current_vertex_index = []
            if self._is_topview:
                if event.type == gui.MouseEvent.Type.BUTTON_UP and event.is_button_down(gui.MouseButton.RIGHT):
                    self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                    eyemat = self._scene.scene.camera.get_view_matrix()[:3,3]
                    self.center[0] = -eyemat[1]
                    self.center[1] = eyemat[0]
                    self._scene.center_of_rotation = self.center
                    return gui.Widget.EventCallbackResult.IGNORED
                else:
                    pass
                
            return gui.Widget.EventCallbackResult.IGNORED

        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')

            

    def _pressed_prev_button(self):
        try:
            if self._slider_od.int_value == 1:
                self.logger.warning("No previous frame exist !!")
                self.parent._message_dialog('INFO', ['No previous frame exist !!'])
            else:
                self._set_prev_plot_od()
                    
        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')

    def _pressed_next_button(self):
        try:
            if self._slider_od.int_value == self._max_frame:
                self.logger.warning("No more frame exist !!!")
                self.parent._message_dialog('INFO', ['No more frame exist !!'])
            else:
                self._set_next_plot_od()
                    
        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')

    def _set_prev_plot_od(self):
        '''
        ### OD 
        ### Plot and Visualize selected frame with GT boxes
        '''
        try:
            frame_idx_json = (self._slider_od.int_value - 1) - 1

            if self._slider_od.int_value == 1:
                self.logger.warning("No previous frame exist !!")
                self.parent._message_dialog('INFO', ['No previous frame exist !!'])
            else:
                self._slider_od.int_value = frame_idx_json + 1
                self._slider_label_od.text = str(frame_idx_json + 1) + " / " + str(self._max_frame)
                # self._set_plot_od(frame_idx_json)

                def update_label():    
                    text = "Changing Frames..."
                    self.info_title.text = text
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.window, update_label)
                self._scene.force_redraw()


                self.new_slider_idx = frame_idx_json
                self.parent.changing_slide_sign = True

        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')

    def _set_next_plot_od(self):
        '''
        ### OD 
        ### Plot and Visualize selected frame with GT boxes
        '''
        try:
            frame_idx_json = (self._slider_od.int_value - 1) + 1
            
            if self._slider_od.int_value == self._max_frame:
                self.logger.warning("No more frame exist !!")
                self.parent._message_dialog('INFO', ['No more frame exist !!'])
            else:
                self._slider_od.int_value = frame_idx_json + 1
                self._slider_label_od.text = str(frame_idx_json + 1) + " / " + str(self._max_frame)
                # self._set_plot_od(frame_idx_json)

                def update_label():    
                    text = "Changing Frames..."
                    self.info_title.text = text
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.window, update_label)
                self._scene.force_redraw()


                self.new_slider_idx = frame_idx_json
                self.parent.changing_slide_sign = True
                
        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')


    ##### 2-1. animation ####
    ### play/stop animation callback functions should be modified to run in real time
    def _on_start_animation_od(self):
        '''
        ### OD
        ### Play and Stop showing frames within selected/loaded sequence
        '''
        try:
            log_str = '[OD] Start playing PCD files as a video'
            self.logger.info(log_str)

            def on_tick():
                return self._on_animate_od()

            self._play_button_od.text = "Stop"
            self._play_button_od.set_on_clicked(self._on_stop_animation_od)
            self._scene.force_redraw()
            self._last_animation_time = 0.0
            self.window.set_on_tick_event(on_tick)

        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')

    ### play/stop animation callback functions should be modified to run in real time
    def _on_animate_od(self):
        '''
        ### OD
        ### Play and Stop showing frames within selected/loaded sequence
        '''
        try:
            now = time.time()
            if now >= self._last_animation_time + self._animation_delay_secs * 5:
                idx = (self._slider_od.int_value + 1) % int(self._slider_od.get_maximum_value)
                
                if self._slider_od.int_value + 1 == int(self._slider_od.get_maximum_value):
                    self._slider_od.int_value = int(self._slider_od.get_maximum_value)
                    # self._on_animation_slider_changed_od(int(self._slider_od.get_maximum_value))
                    self._set_next_plot_od()
                    self._on_stop_animation_od()
                    return False

                self._slider_od.int_value = idx
                # self._on_animation_slider_changed_od(idx)
                self._set_next_plot_od()
                self._last_animation_time = now
                return True
            # else:
            #     self._on_stop_animation()
            return False

        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')

    ### play/stop animation callback functions should be modified to run in real time
    def _on_stop_animation_od(self):
        '''
        ### OD
        ### Play and Stop showing frames within selected/loaded sequence
        '''
        try:
            log_str = '[OD] Stop playing PCD files as a video'
            # self.w_log_history_od.add_log(log_str)
            self.logger.info(log_str)

            self.window.set_on_tick_event(None)
            self._play_button_od.text = "Play"
            self._play_button_od.set_on_clicked(self._on_start_animation_od)

            # def update_label():    
            #     text = "Changing Frames......"
            #     self.info_title.text = text
            #     self.window.set_needs_layout()

            # gui.Application.instance.post_to_main_thread(self.window, update_label)
            self._scene.force_redraw()

        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')
    
    ### slider_change callback function is only runnable
    def _on_animation_slider_changed_od(self, new_value):
        '''
        ### OD
        ### With this slider and mouse control, user can select certain frame which user want to see
        '''
        # self.logger.warning('slideer_changed')
        try:
            def update_label():    
                text = "Changing Frames..."
                self.parent.info_title.text = text
                self.window.set_needs_layout()

            gui.Application.instance.post_to_main_thread(self.window, update_label)
            self._scene.force_redraw()

            self.new_slider_idx = int(new_value) - 1
            self.parent.changing_slide_sign = True


        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self.parent._message_dialog('ERROR', error_strs)
            self.logger.error('\n[ERROR OCCUR]')
            self.logger.error('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            self.logger.error('Error on line : ', str(sys.exc_info()[-1].tb_lineno))
            self.logger.error('Error Type :', type(err).__name__)
            self.logger.error('Error message : ', err, '\n')


    def _on_new_plot_by_slider(self):
        self._slider_od.enabled = False
        self._set_plot_od(self.new_slider_idx)
        self._slider_od.enabled = True

