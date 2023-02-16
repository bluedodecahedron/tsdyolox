import unittest
from unittest import TestCase
import os
import yolox.tools.demo as demo
import yolox.exp.example.custom.yolox_s_gtsdb as gtsdb
import cv2
from gtsdb_util import change_working_dir, videopath, singlevideo, external_imagepath, singleimage, model, exp, demoscript, val_imagepath


change_working_dir()


class InferenceTestsCodeFast(TestCase):
    def test_code_warmup_inference(self):
        predictor = demo.PredictorBuilder(
            exp=gtsdb.Exp(),
            options='video '
                    '-n yolox-s '
                    f'-c {model} '
                    '--device gpu '
                    '--tsize 800 '
                    '--conf 0.3'
        ).build()
        predictor.warmup(5)

    def test_code_image_inference(self):
        predictor = demo.PredictorBuilder(
            exp=gtsdb.Exp(),
            options='video '
                    '-n yolox-s '
                    f'-c {model} '
                    '--device gpu '
                    '--tsize 800 '
                    '--conf 0.3'
        ).build()
        image = cv2.imread(singleimage)
        infer_result = predictor.inference(image)
        boxes = infer_result.boxed_images
        self.assertEqual(len(boxes), 9)

    def test_code_image_inference_show_result(self):
        predictor = demo.PredictorBuilder(
            exp=gtsdb.Exp(),
            options='video '
                    '-n yolox-s '
                    f'-c {model} '
                    '--device gpu '
                    '--tsize 800 '
                    '--conf 0.3'
        ).build()
        image = cv2.imread(singleimage)
        infer_result = predictor.inference(image)
        result_image = infer_result.visual()
        boxes = infer_result.boxed_images
        self.assertEqual(len(boxes), 9)

        cv2.imshow("result", result_image)
        cv2.imshow("box content", boxes[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class InferenceTestsCodeSlow(TestCase):
    def test_code_video_inference(self):
        predictor = demo.PredictorBuilder(
            exp=gtsdb.Exp(),
            options='video '
                    '-n yolox-s '
                    f'-c {model} '
                    '--device gpu '
                    '--tsize 800 '
                    '--conf 0.3'
        ).build()
        batch_predictor = demo.BatchPredictor(predictor, save_result=True)
        batch_predictor.run_demo(singlevideo)


class InferenceTestsCommandFast(TestCase):
    def test_command_all_external_images_inference(self):
        os.system(
            f"python {demoscript} image "
            "-n yolox-s "
            f"-c {model} "
            f"--path {external_imagepath} "
            "--save_result "
            "--device gpu "
            "--tsize 800 "
            f"-f {exp} "
            "--conf 0.3"
        )


class InferenceTestsCommandSlow(TestCase):
    def test_command_all_validation_images_inference(self):
        os.system(
            f"python {demoscript} image "
            "-n yolox-s "
            f"-c {model} "
            f"--path {val_imagepath} "
            "--save_result "
            "--device gpu "
            "--tsize 800 "
            f"-f {exp} "
            "--conf 0.3"
        )

    def test_command_video_inference(self):
        os.system(
            f"python {demoscript} video "
            "-n yolox-s "
            f"-c {model} "
            f"--path {singlevideo} "
            "--save_result "
            "--device gpu "
            "--tsize 800 "
            f"-f {exp} "
            "--conf 0.3"
        )

    def test_command_all_videos_inference(self):
        for filename in os.listdir(videopath):
            filepath = videopath + filename
            os.system(
                f"python {demoscript} video "
                "-n yolox-s "
                f"-c {model} "
                f"--path {singlevideo} "
                "--save_result "
                "--device gpu "
                "--tsize 800 "
                f"-f {exp} "
                "--conf 0.3"
            )


if __name__ == '__main__':
    unittest.main()
