import torch

from yolox.utils.visualize import vis, boxes


class InferResult:
    def __init__(self, outputs, img_info, cls_names, confthre):
        if outputs[0] is not None:
            self.output = outputs[0].cpu()
        else:
            self.output = None
        self.img = img_info["raw_img"]
        self.ratio = img_info["ratio"]
        self.confthre = confthre
        self.cls_names = cls_names

    def img_copy(self):
        # make a copy so that our operations are done on a new object
        return self.img.copy()

    def get_box_borders(self):
        box_borders = torch.clone(self.output)[:, 0:4]
        # preprocessing: resize
        box_borders /= self.ratio
        return box_borders

    def get_scores(self):
        cls = self.output[:, 6]
        scores = self.output[:, 4] * self.output[:, 5]
        return cls, scores

    def get_boxed_images(self):
        if self.output is None:
            return []

        box_borders = self.get_box_borders()
        cls, scores = self.get_scores()
        images = boxes(self.img_copy(), box_borders, scores, self.confthre)
        return images

    def visual(self):
        if self.output is None:
            return self.img.copy()

        box_borders = self.get_box_borders()
        cls, scores = self.get_scores()
        vis_res = vis(self.img_copy(), box_borders, scores, cls, self.confthre, self.cls_names)
        return vis_res