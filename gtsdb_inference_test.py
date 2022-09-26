import yolox.tools.demo as demo
import yolox.exp.example.custom.yolox_s_gtsdb as gtsdb
import cv2


def get_tsd_predictor():
    print('Initializing yoloX pytorch model for traffic sign detection')
    predictor = demo.PredictorBuilder(
        exp=gtsdb.Exp(),
        options='image '
                '-n yolox-s '
                '-c YOLOX_outputs/yolox_s_gtsdb/best_ckpt.pth '
                '--device gpu '
                '--tsize 800 '
                '--conf 0.3'
    ).build()
    predictor.warmup(2)
    print('Initialization of yoloX pytorch model complete')
    return predictor


predictor = get_tsd_predictor()


def tsd(image):
    outputs, img_info = predictor.inference(image)
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    boxes = predictor.boxes(outputs[0], img_info, predictor.confthre)

    return boxes, result_image


image_file = cv2.imread("assets/gtsdbtest21.jpg")
boxes, result_image = tsd(image_file)

cv2.imshow("result", result_image)
cv2.imshow("box content", boxes[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
