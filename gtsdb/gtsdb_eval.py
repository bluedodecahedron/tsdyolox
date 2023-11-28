# Run this script only in an environment that knows build tools (On Windows use the Developer Console for VS)
# Otherwise this will not work!

import os
from gtsdb_util import change_working_dir, exp, model


change_working_dir()


def evaluate():
    os.system(
        "python "
        "tools/eval.py "
        "-n yolox-s "
        f"-c {model} "
        f"-f {exp} "
        "-b 1 "
        "-d 1 "
        "--conf 0.75 "
        "--fp16 "
        "--fuse"
    )


if __name__ == "__main__":
    evaluate()
