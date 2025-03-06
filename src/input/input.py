 
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd  
import numpy as np
import json
from glob import glob
from PIL import Image

def get_image_files(input_path):
    allowed_formats = ['jpg', 'png', 'jpeg']
    image_files = []

    for file_type in allowed_formats:
        for path in glob(os.path.join(input_path, f'*.{file_type}')):
            image_files.append(path)

        for path in glob(os.path.join(input_path, f'*/*.{file_type}')):
            image_files.append(path)

    return image_files


def get_query_data(input_path):

    allowed_formats = ['jpg', 'png', 'jpeg']
    query_data = []
    
    if os.path.isdir(input_path):
        class_folders = [f for f in os.listdir(input_path) 
                       if os.path.isdir(os.path.join(input_path, f))]
        
        for class_name in class_folders:
            class_path = os.path.join(input_path, class_name)
            
            all_images = []
            for file_type in allowed_formats:
                image_pattern = os.path.join(class_path, f'*.{file_type}')
                all_images.extend(glob(image_pattern))
            
            if not all_images:
                continue
                
            origin_images = {}
            crop_images = []
            
            for img_path in all_images:
                file_name = os.path.basename(img_path)
                
                label_path = img_path.replace('.' + img_path.split('.')[-1], '_label.txt')
                if os.path.exists(label_path):
                    boxes, classes = get_bbox(label_path)
                    query_data.append((img_path, boxes, classes))
                    continue
                
                if '_crop' in file_name:
                    crop_images.append(img_path)
                else:
                    origin_id = file_name.split('.')[0]
                    origin_images[origin_id] = img_path
            
            for crop_path in crop_images:
                crop_name = os.path.basename(crop_path)
                
                origin_id = None
                for oid in origin_images.keys():
                    if crop_name.startswith(oid + '_crop'):
                        origin_id = oid
                        break
                
                if origin_id and origin_id in origin_images:
                    origin_path = origin_images[origin_id]
                    
                    enhanced_data = enhance_with_original_context(
                        origin_path, crop_path, class_name)
                    query_data.extend(enhanced_data)
                else:

                    query_data.append(
                        (crop_path, np.array([[0.0, 0.0, 1.0, 1.0]]), [class_name])
                    )
                    
                    image = Image.open(crop_path).convert('RGB')
                    padded_image, padded_box = add_context_to_crops(image)
                    padded_file = crop_path.replace('.' + crop_path.split('.')[-1], '_padded.' + crop_path.split('.')[-1])
                    padded_image.save(padded_file)
                    query_data.append(
                        (padded_file, padded_box, [class_name])
                    )
    

    for file_type in allowed_formats:
        image_pattern = os.path.join(input_path, f'*.{file_type}')
        for image_file in glob(image_pattern):

            is_in_class_folder = False
            for class_name in class_folders if 'class_folders' in locals() else []:
                if image_file.startswith(os.path.join(input_path, class_name)):
                    is_in_class_folder = True
                    break
            
            if is_in_class_folder:
                continue
            
            label_file = image_file.replace('.' + image_file.split('.')[-1], '_label.txt')
            
            if os.path.exists(label_file):
                boxes, classes = get_bbox(label_file)
                query_data.append(
                    (image_file, boxes, classes)
                )
    
    return query_data

def enhance_with_original_context(origin_path, crop_path, class_name):
    result_data = []
    
    try:
        origin_img = Image.open(origin_path).convert('RGB')
        crop_img = Image.open(crop_path).convert('RGB')
        
        result_data.append(
            (crop_path, np.array([[0.0, 0.0, 1.0, 1.0]]), [class_name])
        )
        
        crop_location = find_crop_in_original(origin_img, crop_img)
        
        if crop_location is not None:
            x, y, w, h = crop_location
            context_size = 1.5 
            
            cx, cy = x + w/2, y + h/2  
            new_w, new_h = w * context_size, h * context_size
            new_x, new_y = cx - new_w/2, cy - new_h/2
            
            ow, oh = origin_img.size
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_x2 = min(ow, new_x + new_w)
            new_y2 = min(oh, new_y + new_h)
            

            context_img = origin_img.crop((new_x, new_y, new_x2, new_y2))

            ctx_w, ctx_h = context_img.size
            rel_x = (x - new_x) / ctx_w
            rel_y = (y - new_y) / ctx_h
            rel_w = w / ctx_w
            rel_h = h / ctx_h
            

            x1, y1 = rel_x, rel_y
            x2, y2 = rel_x + rel_w, rel_y + rel_h
            
            context_file = crop_path.replace('.' + crop_path.split('.')[-1], 
                                          '_context.' + crop_path.split('.')[-1])
            context_img.save(context_file)
            
            result_data.append(
                (context_file, np.array([[x1, y1, x2, y2]]), [class_name])
            )
        
    except Exception as e:
        print(f"원본 컨텍스트 처리 중 오류: {e}")
    
    return result_data

def find_crop_in_original(original, crop, threshold=0.7):

    try:
        import cv2
        import numpy as np
        

        orig_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
        crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        
        result = cv2.matchTemplate(orig_cv, crop_cv, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            return (max_loc[0], max_loc[1], crop_cv.shape[1], crop_cv.shape[0])
        else:
            return None
            
    except ImportError:
        print("템플릿 매칭을 위해 OpenCV가 필요합니다.")
        return None
    except Exception as e:
        print(f"템플릿 매칭 중 오류: {e}")
        return None

def add_context_to_crops(image, padding_ratio=0.2):
    w, h = image.size

    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    padded_image = Image.new(
        image.mode, 
        (w + 2*pad_w, h + 2*pad_h),
        (240, 240, 240) 
    )
    
    padded_image.paste(image, (pad_w, pad_h))
    
    new_w, new_h = padded_image.size
    box = np.array([
        [pad_w/new_w, pad_h/new_h, (pad_w+w)/new_w, (pad_h+h)/new_h]
    ])
    
    return padded_image, box


def get_bbox(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    classes = []
    boxes = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 5:  # class, x, y, w, h 형식 확인
            class_name = parts[0].strip()
            # if class_name.startswith("class "):
            #     class_name = class_name[6:].strip()  
            try:
                x, y, w, h = [float(z.strip()) for z in parts[1:5]]
                x1, x2 = x - 0.5 * w, x + 0.5 * w
                y1, y2 = y - 0.5 * h, y + 0.5 * h
                
                boxes.append([x1, y1, x2, y2])
                classes.append(class_name) 
            except (ValueError, IndexError):
                print(f"경고: {path}에서 잘못된 형식의 라인 발견: {line}")
    
    boxes = np.array(boxes) if boxes else np.empty((0, 4))
    return boxes, classes



