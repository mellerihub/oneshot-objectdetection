# self.device = self.args['device']

#         if not gpu_ok():
#             self.device = 'cpu'

#         if self.device == 'cpu':
#             self.asset.save_warning('CUDA not Working!')
#             self.asset.save_warning('It is strongly recommended to use CUDA for fast inference')

#         self.target_class = self.config['target_class']

#         if isinstance(self.target_class, str):
#             if self.target_class == 'ALL':
#                 self.target_class = None

#         else:
#             self.target_class = [str(x) for x in self.target_class]
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

