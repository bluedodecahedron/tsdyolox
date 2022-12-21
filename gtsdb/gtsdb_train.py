# Run this script only in an environment that knows build tools (On Windows use the Developer Console for VS)
# Otherwise this will not work!

import os
import wandb
from gtsdb_util import change_working_dir, exp, wandb_project_name


change_working_dir()


def wandb_init():
    wandb.login()


def train():
    os.system(
        "python "
        "tools/train.py "
        f"-f {exp} "
        "-d 1 "
        "-b 4 "
        "--fp16 "
        "-c yolox_s.pth "
        f"--logger wandb wandb-project {wandb_project_name}"
    )


if __name__ == "__main__":
    wandb_init()
    train()
