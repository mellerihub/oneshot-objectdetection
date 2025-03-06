import cv2
import numpy as np
import supervision as sv

def annotate(image, boxes, scores, classes_list):
    class_set = set(classes_list)
    class_to_idx = {cls: i for i, cls in enumerate(class_set)}

    class_id = np.array([class_to_idx[c] for c in classes_list])
    detections = sv.Detections(xyxy=boxes, class_id=class_id)

    labels = [
        f"{cls} {score:.2f}"
        for score, cls
        in zip(scores, classes_list)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS)
    annotated_frame = np.array(image)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame
