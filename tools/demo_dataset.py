import os.path as osp
import json
with open('/home/jonghokim/OpenPCDet/data/nuscenes/v1.0-trainval/v1.0-trainval/scene.json') as f:
    scenes = json.load(f)
with open('/home/jonghokim/OpenPCDet/data/nuscenes/v1.0-trainval/v1.0-trainval/sample.json') as f:
    samples = json.load(f)
with open('/home/jonghokim/OpenPCDet/data/nuscenes/v1.0-trainval/v1.0-trainval/sample_data.json') as f:
    sample_datas = json.load(f)
with open('/home/jonghokim/OpenPCDet/data/nuscenes/v1.0-trainval/v1.0-trainval/calibrated_sensor.json') as f:
    calibrated_sensors = json.load(f)
with open('/home/jonghokim/OpenPCDet/data/nuscenes/v1.0-trainval/v1.0-trainval/sensor.json') as f:
    sensors = json.load(f)

scene_token = scenes[0]['token']

sample_token_list = []
for sample in samples:
    if sample['scene_token'] == scene_token:
        sample_token_list.append(sample['token'])

sensor_time_dict_key = {}
sensor_time_dict_sweep = {}
for sensor in sensors:
    sensor_time_dict_key.update({sensor['channel']: []})
    sensor_time_dict_sweep.update({sensor['channel']: []})

s_timestamp = 1531883530449377
for sample_data in sample_datas:
    if sample_data['sample_token'] in sample_token_list:
        for calibrated_sensor in calibrated_sensors:
            if calibrated_sensor['token'] == sample_data['calibrated_sensor_token']:
                sensor_token = calibrated_sensor['sensor_token']

        for sensor in sensors:
            if sensor['token'] == sensor_token:
                sensor_channel = sensor['channel']

        if sample_data['is_key_frame']:
            dt = (int(sample_data['timestamp']) - s_timestamp)/10e5
            sensor_time_dict_key[sensor_channel].append(dt)
        else:
            dt = (int(sample_data['timestamp']) - s_timestamp)/10e5
            sensor_time_dict_sweep[sensor_channel].append(dt)

import numpy as np
import matplotlib.pyplot as plt

key_priority = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'LIDAR_TOP', 'RADAR_FRONT_LEFT', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT','RADAR_BACK_RIGHT', 'RADAR_BACK_LEFT',]
y = 1
for k in key_priority:
    print(k)
    v = sensor_time_dict_key[k]
    if y in [1,2,3,4,5,6]:
        plt.plot(v, [y]*len(v), 'r.', linewidth=3.0)
    elif y in [8,9,10,11,12]:
        plt.plot(v, [y]*len(v), 'r+', linewidth=3.0)
    else:
        plt.plot(v, [y]*len(v), 'r*', linewidth=3.0)

    y = y+1

y = 1
for k in key_priority:
    v = sensor_time_dict_sweep[k]
    if y in [1,2,3,4,5,6]:
        plt.plot(v, [y]*len(v), 'k.', linewidth=3.0)
    elif y in [8,9,10,11,12]:
        plt.plot(v, [y]*len(v), 'k+', linewidth=3.0)
    else:
        plt.plot(v, [y]*len(v), 'k*', linewidth=3.0)
    y = y+1

plt.grid()
plt.show()