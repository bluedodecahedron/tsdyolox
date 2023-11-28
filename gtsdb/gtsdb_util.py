import os
import pathlib


def change_working_dir():
    filepath = pathlib.Path(__file__).parent.parent.resolve()
    os.chdir(filepath)
    print(pathlib.Path().resolve())


videopath = 'datasets/gtsdb01/external_videos/'
singlevideo = videopath + 'back4.mp4'
external_imagepath = 'datasets/gtsdb01/external_images/'
val_imagepath = 'datasets/gtsdb01/val2017/'
singleimage = external_imagepath + 'gtsdbtest21.jpg'
model = 'YOLOX_outputs/yolox_s_gtsdb_final/best_ckpt.pth'
exp = 'yolox/exp/example/custom/yolox_s_gtsdb.py'
demoscript = 'tools/demo.py'
wandb_project_name = 'gtsdb-yolox'