#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets.gtsdb_classes import GTSDB_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tools.infer_result import InferResult

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    # exp file
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


class PredictorBuilder:
    def __init__(self,
                 exp=None,
                 options=''
                 ):
        self._args = make_parser().parse_args(options.split())
        self._exp = exp

    def build(self):
        file_name = os.path.join(self._exp.output_dir, self._exp.exp_name)

        if not self._args.experiment_name:
            self._args.experiment_name = self._exp.exp_name

        if self._args.trt:
            self._args.device = "gpu"

        logger.info("Args: {}".format(self._args))

        if self._args.conf is not None:
            self._exp.test_conf = self._args.conf
        if self._args.nms is not None:
            self._exp.nmsthre = self._args.nms
        if self._args.tsize is not None:
            self._exp.test_size = (self._args.tsize, self._args.tsize)

        model = self._exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, self._exp.test_size)))

        if self._args.device == "gpu":
            model.cuda()
            if self._args.fp16:
                model.half()  # to FP16
        model.eval()

        if not self._args.trt:
            if self._args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = self._args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if self._args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if self._args.trt:
            assert not self._args.fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        predictor = Predictor(
            model=model,
            exp=self._exp,  # model and experiment
            demo=self._args.demo,
            camid=self._args.camid,
            cls_names=GTSDB_CLASSES,
            trt_file=trt_file,
            decoder=decoder,
            device=self._args.device,
            fp16=self._args.fp16,
            legacy=self._args.legacy
        )

        return predictor


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        demo,
        camid=None,
        cls_names=GTSDB_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.exp = exp
        self.demo = demo
        self.camid = camid
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            torch.cuda.synchronize()
            logger.info("Infer time: {:.4f}s".format(time.perf_counter() - t0))
        return InferResult(outputs, img_info, self.cls_names, self.confthre)

    def warmup(self, num_images):
        logger.info("Warmup inference")
        for i in range(num_images):
            dummy_input = torch.randn(10, 10, 3).numpy()
            self.inference(dummy_input)


class BatchPredictor:
    def __init__(self,
                 predictor,
                 save_result=False,
                 ):
        self._predictor = predictor
        self._save_result = save_result

        file_name = os.path.join(self._predictor.exp.output_dir, self._predictor.exp.exp_name)
        os.makedirs(file_name, exist_ok=True)
        self._vis_folder = None
        if self._save_result:
            self._vis_folder = os.path.join(file_name, "vis_res")
            os.makedirs(self._vis_folder, exist_ok=True)

    def run_demo(self, source_path):
        current_time = time.localtime()
        if self._predictor.demo == "image":
            self._image_demo(source_path, current_time)
        elif self._predictor.demo == "video" or self._predictor.demo == "webcam":
            self._imageflow_demo(source_path, current_time)

    def _image_demo(self, source_path, current_time):
        if os.path.isdir(source_path):
            files = self._get_image_list(source_path)
        else:
            files = [source_path]
        files.sort()
        for image_name in files:
            outputs, img_info = self._predictor.inference(image_name)
            result_image = self._predictor.visual(outputs[0], img_info, self._predictor.confthre)
            if self._save_result:
                save_folder = os.path.join(
                    self._vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                os.makedirs(save_folder, exist_ok=True)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                logger.info("Saving detection result in {}".format(save_file_name))
                cv2.imwrite(save_file_name, result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

    def _imageflow_demo(self, source_path, current_time):
        cap = cv2.VideoCapture(source_path if self._predictor.demo == "video" else self._predictor.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        if self._save_result:
            save_folder = os.path.join(
                self._vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            if self._predictor.demo == "video":
                save_path = os.path.join(save_folder, os.path.basename(source_path))
            else:
                save_path = os.path.join(save_folder, "camera.mp4")
            logger.info(f"video save_path is {save_path}")
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
            )
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                infer_result = self._predictor.inference(frame)
                result_frame = infer_result.visual() # self._predictor.visual(outputs[0], img_info, self._predictor.confthre)
                if self._save_result:
                    vid_writer.write(result_frame)
                else:
                    cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                    cv2.imshow("yolox", result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break

    def _get_image_list(self, path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
        return image_names
