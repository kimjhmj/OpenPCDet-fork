# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------


import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import yaml
import time
import threading

import argparse
import logging


from detection_utils import Detection_utils
from tool_utils import _shortcut_key_info


isMacOS = (platform.system() == "Darwin")

version_info = 'v0.0.1'
doc_version_info = 'v0.0.1'
labeling_version_info = '1.0.0'
release_date = '2024.01.17'
# is_program_started = False
# global yaml_path

if platform.system() == "Windows":
    # it is necessary to specify paths on Windows since it stores its fonts
    # with a cryptic name, so font name searches do not work on Windows
    # serif = "C:/Windows/Fonts/HyundaiSansTextKRMedium.ttf"  # Times New Roman
    # ko_ttc = "C:/Windows/Fonts/HyundaiSansTextKRMedium.ttf"  # YaHei UI
    # serif = "C:/Windows/Fonts/LSANS.ttf"  # Times New Roman
    # ko_ttc = "C:/Windows/Fonts/LSANS.ttf"  # YaHei UI
    serif = "C:/Windows/Fonts/micross.tff"
    # serif = "c:/windows/fonts/times.ttf"  # Times New Roman
    # ko_ttc = "C:/Windows/Fonts/micross.ttf"  # YaHei UI
    # ko_ttc = "C:/Windows/Fonts/gulim.ttc"  # YaHei UI
    ko_ttc = "C:/Windows/Fonts/malgun.ttf" # 맑은 고딕
    # "C:\Users\7233384\AppData\Local\Microsoft\Windows\Fonts"

class Config:
    def __init__(self):
        self.class_names = 0
        self.det_color_map = 0
        self.root = ''
        self.save_root = ''
        # self.model_path = ''


def init_config_params(BGR = False):
    DB_cfg = yaml.load(open('tools/viz_utils/DB-GT.yaml','r'), Loader=yaml.SafeLoader)
    cfg = Config()
    cfg.class_names = DB_cfg['labels']
    cfg.det_color_map = DB_cfg['color_map']
    
    if BGR == True:
        for v in cfg.det_color_map.values():
            bgr2rgb_tmp = v[0]
            v[0] = v[2]
            v[2] = bgr2rgb_tmp    

    cfg.root = DB_cfg['root_path']

    assert os.path.exists(cfg.root),'Root directory does not exist '+ cfg.root

    return cfg

class Settings:
    UNLIT = "defaultUnlit"
    UNLIT_LINE = "unlitLine"
    UNLIT_INSP = "unlitInsp"
    UNLIT_INSP_CHECKED = "unlitInspCheck"
    UNLIT_OCCL = "unlitOcclCheck"
    UNLIT_INFER = "inferenceResult"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    def __init__(self):
        self.bg_color = gui.Color(0.0, 0.0, 0.0)
        
        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.UNLIT_LINE: rendering.MaterialRecord(),
            Settings.UNLIT_INSP: rendering.MaterialRecord(),
            Settings.UNLIT_INSP_CHECKED: rendering.MaterialRecord(),
            Settings.UNLIT_OCCL: rendering.MaterialRecord(),
            Settings.UNLIT_INFER: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.UNLIT_LINE].shader = Settings.UNLIT_LINE
        self._materials[Settings.UNLIT_INSP].base_color = [1.0, 1.0, 0.141, 1.0]
        self._materials[Settings.UNLIT_INSP].shader = Settings.UNLIT_LINE
        self._materials[Settings.UNLIT_INSP_CHECKED].base_color = [1.0, 0.0, 0.0, 1.0]
        self._materials[Settings.UNLIT_INSP_CHECKED].shader = Settings.UNLIT_LINE
        
        # self._materials[Settings.UNLIT_OCCL].base_color = [0.25, 0.85, 0.95, 0.9]
        self._materials[Settings.UNLIT_OCCL].shader = Settings.UNLIT_LINE
        self._materials[Settings.UNLIT_INFER].shader = Settings.UNLIT_LINE
        self._materials[Settings.UNLIT_INFER].base_color = [0.051, 1.0, 0.051, 1.0]
        
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        # self.material.shader = "defaultUnlit"
        # self.material.base_color = [0.9, 0.9, 0.9, 1.0]
        # c = self._color.color_value
        # self.material.base_color = [c.red, c.green, c.blue, 1.0]
        self.material = self._materials[Settings.LIT]
        self.material_line = self._materials[Settings.UNLIT_LINE]
        self.material_selected = self._materials[Settings.UNLIT]
        self.material_insp = self._materials[Settings.UNLIT_INSP]
        self.material_insp_checked = self._materials[Settings.UNLIT_INSP_CHECKED]
        self.material_check_occl = self._materials[Settings.UNLIT_OCCL]
        self.material_check_infer_result = self._materials[Settings.UNLIT_INFER]

        self.point_size = [2, 10]
        self.line_width = [2, 4] # yss - [2, 4]
        self.material.point_size = self.point_size[0]
        self.material_selected.point_size = self.point_size[1]
        self.material_line.line_width = self.line_width[0]
        self.material_insp.line_width = self.line_width[1]
        self.material_insp_checked.line_width = self.line_width[1]
        self.material_check_occl.line_width = self.line_width[0]
        self.material_check_infer_result.line_width = self.line_width[0]

class AppWindow:
    MENU_LOG = 1
    MENU_LOAD = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_SHOW_OBJ_INFO = 12
    MENU_SHOW_SEG_INFO = 13


    MENU_ABOUT = 21
    MENU_SHORTCUT = 22

    DEFAULT_IBL = "default"
    def __init__(self, width, height, cfg):
        self.settings = Settings()
        self.cfg = cfg
        self.obj_index_label = []
        self.cad_index_label = []
        self.axis_label = []
        self.show_param = False
        self.show_6ch_param = False
        self.fov = 25
        self.cwd_path = os.getcwd()
        self._is_first_quit = True
        self.labeling_version_info = labeling_version_info
        # print('Load config file [%s]'%(self.cfg.view_cfg))

        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL
        
        self.window = gui.Application.instance.create_window(
            "Multi-Modal Data Viz tool: " + version_info, width, height)
        w = self.window  # to make the code more concise
        self.window.set_on_close(self._on_window_close)
        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        # self._scene.enable_scene_caching(True)

        ### Data Variables
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self.point_color_index = np.zeros(10)
        self._scene.scene.set_background(bg_color)
        
        
        self.settings.material.point_size = np.max([self.settings.material.point_size, self.settings.point_size[0]])
        self.settings.material_selected.point_size = np.max([self.settings.material.point_size, self.settings.point_size[1]])
        self.settings.material_line.line_width = np.max([self.settings.material_line.line_width, self.settings.line_width[0]])
        self.settings.material_insp.line_width = np.max([self.settings.material_insp.line_width, self.settings.line_width[1]])


        self.is_point_select_mode = False
        self._store_button = False
        self.current_frame = ''


        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        
        self.info = gui.Label("")
        self.info.visible = False
        self.info_title = gui.Label("Viewer Mode")
        self.info_title.visible = True

        self.data_info = gui.Label("# Sweeps for LiDAR & Radar")
        self.data_info.visible = True

        # taps API is very important
        self.tab_status = "INIT"
        self.tabs = gui.TabControl()

        utils_OD = Detection_utils(self)
        
        self.tab2 = utils_OD.tab2

        self.tabs.add_tab("      Data Viewer      ", self.tab2)

        self._settings_panel.add_child(self.tabs)
        self._settings_panel.add_fixed(separation_height)

        
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        
        w.add_child(self.info)
        w.add_child(self.info_title)
        w.add_child(self.data_info)
        

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            # file_menu.add_item("Load Configurations...", AppWindow.MENU_LOAD)
            # file_menu.add_item("Export Log History...", AppWindow.MENU_LOG)
            # if not isMacOS:
            #     file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()

            settings_menu.add_item("Control Panel",
                                AppWindow.MENU_SHOW_SETTINGS)

            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)

            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)
            help_menu.add_item("ShortCut Key", AppWindow.MENU_SHORTCUT)
            
            
            menu = gui.Menu()
            # if isMacOS:
            #     # macOS will name the first menu item for the running application
            #     # (in our case, probably "Python"), regardless of what we call
            #     # it. This is the application menu, and it is where the
            #     # About..., Preferences..., and Quit menu items typically go.
            #     menu.add_menu("Example", app_menu)
            #     menu.add_menu("File", file_menu)
            #     menu.add_menu("Settings", settings_menu)
            #     # Don't include help menu unless it has something more than
            #     # About...
            # else:
            #     menu.add_menu("File", file_menu)
            #     menu.add_menu("Settings", settings_menu)
            #     menu.add_menu("Help", help_menu)
            menu.add_menu("File", file_menu)
            menu.add_menu("Settings", settings_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        w.set_on_menu_item_activated(AppWindow.MENU_SHORTCUT, self._on_menu_shortcut)

        
        # utils_SEG.update(self)
        utils_OD.update(self)
        self.utils_OD = utils_OD

        

        def thread_main():
            while True:
                # Update geometry
                info_text = self.info_title.text
                def update_label():
                    # print(info_text)
                    self.info_title.text = info_text

                if self.stop_sign:
                    break
                elif self.loading_sign:
                    self.loading_sign = False
                    self.utils_OD.load_dataset()
                    # gui.Application.instance.post_to_main_thread(self.window, self.utils_OD.load_dataset)
                    
                    time.sleep(0.1)
                elif self.changing_slide_sign:
                    self.changing_slide_sign = False
                    self.utils_OD._is_first_plot = False
                    # self.utils_OD._on_new_plot_by_slider()
                    gui.Application.instance.post_to_main_thread(self.window, self.utils_OD._on_new_plot_by_slider)
                    time.sleep(0.1)
                else:
                    gui.Application.instance.post_to_main_thread(self.window, update_label)
                    time.sleep(0.1)
        
        self.stop_sign = False
        self.loading_sign = False
        self.changing_slide_sign = False
        self.thh = threading.Thread(target=thread_main)
        self.thh.start()
        self.cfg.logger.info('START INFO TEXT THREADING')
        
        

##### layout
    def _on_layout(self, layout_context):
        '''
        ### Layout setting
        '''
        r = self.window.content_rect
        self._scene.frame = r
        width = 50 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                            height)

        pref = self.info_title.calc_preferred_size(layout_context,
                                            gui.Widget.Constraints())
        info_height = layout_context.theme.font_size/2
        self.info_title.frame = gui.Rect(r.get_left() + width/17, r.y+pref.height/2, 
                                    pref.width, info_height)

        pref = self.info.calc_preferred_size(layout_context,
                                            gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.get_left() + width/17, r.y+pref.height*2,
                                    pref.width, info_height)
        
        pref = self.data_info.calc_preferred_size(layout_context,
                                            gui.Widget.Constraints())
        self.data_info.frame = gui.Rect(r.get_left() + width/17, r.height -pref.height*1.5,
                                    pref.width, info_height)
        
        
###########################

#########################################
        #### Menu Function ####
#########################################
#### 3. menu of the Application #############
    def _on_menu_log(self):
        try:
            ind = self.tabs.selected_tab_index
            log_str = "Exporting Log History @ " + self.cfg.root
            

        except Exception as err:
            error_strs = []
            error_strs.append('\n[ERROR OCCUR]')
            error_strs.append('Error file : ' + str(sys.exc_info()[-1].tb_frame).split("'")[1])
            error_strs.append('Error on line : ' + str(sys.exc_info()[-1].tb_lineno))
            error_strs.append('Error Type :' + str(type(err).__name__))
            error_strs.append('Error message : ' + str(err) + '\n')
            self._message_dialog('ERROR', error_strs)
            print('\n[ERROR OCCUR]')
            print('Error file : ', str(sys.exc_info()[-1].tb_frame).split("'")[1])
            print('Error on line : ', sys.exc_info()[-1].tb_lineno)
            print('Error Type :', type(err).__name__)
            print('Error message : ', err, '\n')
    
    def _message_dialog(self, info_type, error_strs):
        # try:
        dlg = gui.Dialog(info_type + " Message")

        # Add the message text
        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        
        for ii in error_strs:
            dlg_layout.add_child(gui.Label(ii))

        
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        
        button_layout = gui.Horiz()
        button_layout.add_stretch()
        button_layout.add_child(ok)

        # Add the button layout,
        dlg_layout.add_child(button_layout)
        # ... then add the layout as the child of the Dialog
        dlg.add_child(dlg_layout)
        # ... and now we can show the dialog
        
        if info_type == 'ERROR':
            dlg.background_color = gui.Color(0.5, 0.4, 0.5)
        elif info_type == 'INFO':
            dlg.background_color = gui.Color(0.2, 0.4, 0.5)
        else:
            dlg.background_color = gui.Color(0.4, 0.4, 0.3)
            
        self.window.show_dialog(dlg)


    def _on_menu_quit(self):
        self.stop_sign = True
        self.thh.join()
        self.cfg.logger.info('FINISHED INFO TEXT THREADING')
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)
    
    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("HMC Data Viz Tool"))
        # dlg_layout.add_child(gui.Label("for 기술용역 3단계"))
        dlg_layout.add_child(gui.Label("Version : " + version_info))
        dlg_layout.add_child(gui.Label("Release Date : " + release_date))
        dlg_layout.add_child(gui.Label("Doc Version : " + doc_version_info))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_menu_shortcut(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("ShortCut Key Info")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em))

        shortcut_treeview = gui.ListView()
        sk_list = _shortcut_key_info()
        shortcut_treeview.set_items(sk_list)

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()

        dlg_layout.add_child(shortcut_treeview)
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        dlg.frame.width = em*100
        self.window.show_dialog(dlg)


    def _on_about_ok(self):
        self.window.close_dialog()

    def _on_window_close(self):
        self.stop_sign = True
        self.thh.join()
        self.cfg.logger.info('FINISHED INFO TEXT THREADING')
        gui.Application.instance.quit()
        # self.window.close()


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='[%Y-%m-%d]%H:%M:%S')
    # handler = logging.FileHandler('log.txt', mode='w')
    # handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_path', default=None, type=str, help='[str] dataset/model config file path')
    parser.add_argument('--pth_path', default=None, type=str, help='[str] DL model pth file path to load and visualize prediction results')

    args = parser.parse_args()
    return args

    
def main():
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    args = parse_config()

    cfg = init_config_params(BGR = True) ## BGR=False for OD, BGR=True for SEG
    # if len(sys.argv)==3:
    cfg.cfg_path = args.cfg_path
    cfg.pth_path = args.pth_path
    
    logger = setup_custom_logger('viz_logger')
    cfg.logger = logger


    cfg.logger.info('=============================')
    cfg.logger.info('======    arguments    ======')
    cfg.logger.info('-----------------------------')
    cfg.logger.info('==> config file path    : %s', cfg.cfg_path)
    cfg.logger.info('==> model pth file path : %s', cfg.pth_path)
    cfg.logger.info('=============================')


    ### suppress open3d warning message
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    gui.Application.instance.initialize()
    
    font = gui.FontDescription()
    # font.add_typeface_for_language(ko_ttc, "ko")
    chess = "c:/windows/fonts/seguisym.ttf"  # Segoe UI Symbol
    range = [0x2501, 0x2517] ##   2504: ━  2517 : ┗
    font.add_typeface_for_code_points(chess, range)
    
    if font is not None:
        gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)
    print('check')
    w = AppWindow(2000, 1200, cfg)
    # w = AppWindow(1200, 600, cfg)


    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()
    w.stop_threads = True
    w.thh.join()


if __name__ == "__main__":
    main()
