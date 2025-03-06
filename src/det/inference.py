import os
import logging
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from src.util.visualize import annotate



warnings.filterwarnings('ignore')

#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class NoLog:
    def __enter__(self):
        logging.disable()

    def __exit__(self, type, val, trace):
        logging.disable(logging.NOTSET)


def run_inference(args, image_files, classes, model, embeddings):
    pbar = tqdm(image_files, desc='Inference')
    results = []
    
    threshold = args['threshold'] if 'threshold' in args else 0.7
        
    for path in pbar:
        image = Image.open(path).convert('RGB')
        factor = max(image.width, image.height)
        image_name = path.split('/')[-1]

        pred_boxes, pred_scores, pred_classes = model.batch_predict(image, embeddings,
                                            mode=args['measure'],
                                            classes=classes.tolist(),
                                            calc_score=args['score'],
                                            average_embed=args['average_embed'],
                                            nms_threshold=args['nms_threshold'],
                                            threshold=threshold)
        
        pred_boxes = pred_boxes * factor

        results.append([path, pred_boxes, pred_scores, pred_classes])
        result_image = annotate(image, pred_boxes, pred_scores, pred_classes)

        Image.fromarray(result_image).save(os.path.join(args['output_dir'], image_name))

    del model

    return results
