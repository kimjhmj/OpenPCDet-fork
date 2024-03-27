# HMC Multi-Modality Data Visualization Tool
- This Visualization Tool is built on open3d gui API
- This Visualization Tool cannot be opened in AIP (ai platform)
- This can be run on VDI local platform

## Config Setting
- Please Select config file for DL model without any modification
  - Example path: "projects/HMCBEV/configs/bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"

## Visualization Run Command
```bash
### Visualize only dataset with GT
python tools/visualizer/hmc_db_visualizer.py --cfg_path {config_path}

### Visualize prediction results with GT
python tools/visualizer/hmc_db_visualizer.py --cfg_path {config_path} --pth_path {pth_path}

### Example
python tools/visualizer/hmc_db_visualizer.py --cfg_path 'projects/HMCBEV/configs/bevfusion_lidar-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_largeradar.py' --pth_path '/VDI_BIGDATA2/ADSTD/Personal/junlee/HMCBEV_models/work_dirs/bevfusion_lidar-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_largeradar/epoch_20.pth'
```
- Please note that the **config file path** and **pth file path** might be diffrent with user's setting
- When user want to visualize dataset, config file is required, which contains dataset loader pipeline


## Figures
- Below figure shows how the visualization tool looks like with GT bboxes only

![figure 1](./figs/fig3_gt_viz.png)

- Below figure shows how the visualization tool looks like with prediction results

![figure 2](./figs/fig1_pred_viz.png)

- Prediction results can be varied by score_threshold value
- Thus, in this tool, user can change the score_threshold value with the combobox in the setting panel of the tool

![figure 3](./figs/fig2_pred_viz_score_th05.png)


