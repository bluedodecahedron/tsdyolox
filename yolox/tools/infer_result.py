import torch

from yolox.utils.visualize import vis


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
        self.boxed_images, self.box_borders, self.classes, self.scores = self.process_output()

    def img_copy(self):
        # make a copy so that our operations are done on a new object
        return self.img.copy()

    def process_output(self):
        if self.output is None:
            return [], [], [], []

        box_borders = torch.clone(self.output)[:, 0:4]
        classes = self.output[:, 6]
        scores = self.output[:, 4] * self.output[:, 5]

        # preprocessing: resize
        box_borders /= self.ratio

        boxed_images = []
        new_box_borders = []
        new_cls = []
        new_scores = []

        for i in range(len(box_borders)):
            box = box_borders[i]
            cls = classes[i]
            score = scores[i]
            if score < self.confthre:
                continue
            for j, value in enumerate(box):
                if value < 0.0:
                    box[j] = 0
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            boxed_images.append(self.img[y0:y1, x0:x1].copy())
            new_box_borders.append([x0, y0, x1, y1])
            new_cls.append(cls)
            new_scores.append(score)

        return boxed_images, box_borders, classes, scores

    def visual(self):
        if self.output is None:
            return self.img_copy()

        vis_res = vis(self.img_copy(), self.box_borders, self.scores, self.classes, self.confthre, self.cls_names)
        return vis_res
