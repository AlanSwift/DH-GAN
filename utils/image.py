import random
try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

coco_class = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class RandomCrop:
    def __init__(self, size, padding=0):
        self.padding = padding
        self.size = (size, size)

    def crop_image(self, img):
        i, j, h, w = self.get_params(img, self.size)
        return crop(img, i, j, h, w), (i, j, h, w)

    def crop_bbox(self, proposals: np.ndarray, origin_size: (int, int, int, int)):
        i, j, h, w = origin_size
        proposals[:, 1] = proposals[:, 1] - i
        proposals[:, 3] = proposals[:, 3] - i
        proposals[:, 1] = np.clip(proposals[:, 1], 0, h - 1)
        proposals[:, 3] = np.clip(proposals[:, 3], 0, h - 1)

        proposals[:, 0] = proposals[:, 0] - j
        proposals[:, 2] = proposals[:, 2] - j
        proposals[:, 0] = np.clip(proposals[:, 0], 0, w - 1)
        proposals[:, 2] = np.clip(proposals[:, 2], 0, w - 1)
        return proposals


    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


if __name__ == "__main__":
    print(coco_class[15])