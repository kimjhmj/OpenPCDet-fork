import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from visual_utils import ros_visualizer

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet (with nuscenes dataset)-------------------------')

    from pcdet.datasets import NuScenesDataset
    nusc_dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        logger=logger,
    )

    logger.info(f'Total number of samples: \t{len(nusc_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=nusc_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        idx = 10
        data_dict = nusc_dataset[idx]
        logger.info(f'Visualized sample index: \t{idx + 1}')
        batch_dict = nusc_dataset.collate_batch([data_dict])
        load_data_to_gpu(batch_dict)
        pred_dicts, _ = model.forward(batch_dict)

        # ros_visualizer.visualize_points(points=batch_dict['points'][:, 1:5])

        score_threshold = 0.5
        score_mask = pred_dicts[0]['pred_scores'] > score_threshold
        pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'][score_mask]
        pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'][score_mask]
        pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'][score_mask]

        V.draw_scenes(
            points=batch_dict['points'][:, 1:], gt_boxes=batch_dict['gt_boxes'][0], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], radar_points=batch_dict['radar_points'][:, 1:],
        )

        if batch_dict.get('camera_imgs', None) is not None:
            from matplotlib import pyplot as plt
            img = torch.cat([batch_dict['camera_imgs'][0][idx] for idx in [2,0,1,5,3,4]], dim=2)
            plt.imshow(img.permute(1, 2, 0).cpu().numpy())
            plt.show()

        if not OPEN3D_FLAG:
            mlab.show(stop=True)

        if False:
            class_names = nusc_dataset.class_names
            det_annos = []

            annos = nusc_dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
            )
            det_annos += annos

            result_str, result_dict = nusc_dataset.evaluation(
                det_annos, class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path='.'
            )

            logger.info(result_str)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
