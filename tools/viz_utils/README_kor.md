# HMC Multi-Modality Data Visualization Tool
- 이 툴은 open3d gui API 를 사용하여 만들어짐
- 이 툴은 AIP(ai platform)에서 동작 불가 함(gui 사용 불가)
  - gui 를 열 수 있는 xdisplay 가 뚤려있지 않아 발생한 것으로 파악됨
- 이 툴은 VDI local 에서만 돌릴 수 있음

## Config Setting
- 어떠한 수정 없이, project 내에 있는 보고자 하는 데이터셋의 config 파일을 arg로 추가해야 함
  - Example path: "projects/HMCBEV/configs/bevfusion_lidar-cam-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"

## Visualization Run Command
```bash
### Visualize only dataset with GT
python tools/visualizer/hmc_db_visualizer.py --cfg_path {config_path}

### Visualize prediction results with GT
python tools/visualizer/hmc_db_visualizer.py --cfg_path {config_path} --pth_path {pth_path}

### Example 1
python tools/visualizer/hmc_db_visualizer.py --cfg_path 'projects/HMCBEV/configs/hmc_bevfusion/bevfusion_lidar-radar_voxel0075_second_secfpn_2xb8-cyclic-20e_nus-3d_largeradar.py' --pth_path '/VDI_BIGDATA2/ADSTD/Personal/junlee/HMCBEV_models/work_dirs/bevfusion_lidar-radar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_largeradar/epoch_20.pth'

### Example 2
python tools/visualizer/hmc_db_visualizer.py --cfg_path 'projects/HMCBEV/configs/hmc_ppcp/hmcbev_lidar-radar_pillar016_second_secfpn_2xb4-cyclic-24e_nus-3d.py' --pth_path 'work_dirs/hmcbev_lidar-radar_pillar016_second_secfpn_2xb4-cyclic-24e_nus-3d/epoch_19.pth'
```
- **config file path** 와 **pth file path**는 user 세팅과 다를 수 있음
- 데이터셋을 시각화할 때, config file 은 꼭 arg로 넣어줘야 함.
  - 해당 config file 에는 데이터셋 loader pipeline 이 있고, 이를 활용해서 시각화하기 때문


## Figures
- 아래 figure 는 GT bboxes 만 보여주는 visualization tool 스크린샷

![figure 1](./figs/fig3_gt_viz.png)

- 아래 figure 는 prediction/GT bboxes 모두를 보여주는 visualization tool 스크린샷

![figure 2](./figs/fig1_pred_viz.png)

- prediction 결과는 score_threshold 값에 의해 달라질 수 있음
  - 따라서, 이 툴에서는 오른편 setting panel 에 있는 combobox 를 통해 score_threshold 를 변경할 수 있음

![figure 3](./figs/fig2_pred_viz_score_th05.png)


