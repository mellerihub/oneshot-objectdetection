
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image

class NoLog:
    def __enter__(self):
        logging.disable()

    def __exit__(self, type, val, trace):
        logging.disable(logging.NOTSET)

def set_fewshot_data(query_data):
    images = []
    boxes = []
    classes = []

    for path, box, cl in query_data:
        indices = [i for i, c in enumerate(cl)]
        box = box[indices]
        cl = [cl[i] for i in indices]

        image = Image.open(path).convert('RGB')
        w, h = image.width, image.height

        if (box == np.array([[0, 0, 1.0, 1.0]])).any():
            image = np.pad(np.array(image), ((h, h), (w, w), (0, 0)), constant_values=0)
            w, h = 3 * w, 3 * h
            
            image = Image.fromarray(image)
            box = np.array([[0.333, 0.333, 0.666, 0.666]])

        images.append(image)
        boxes.append(box * np.array([[w, h, w, h]]) / max(w, h))
        classes += cl

    return images, boxes, classes

