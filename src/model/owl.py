from PIL import Image
import torch
import numpy as np
from torchvision import ops
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import box_convert
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from .nms import nms

def gpu_ok():
    return torch.cuda.is_available()

def cosine_similarity(A, B):
    A, B = A.squeeze(), B.squeeze()

    if len(A.shape) == 1:
        A = A.view(1, -1)
        
    if len(B.shape) == 1:
        B = B.view(1, -1)

    A = torch.nn.functional.normalize(A)
    B = torch.nn.functional.normalize(B)

    return torch.einsum('ik, jk->ij', A, B)

class OWLV2:
    def __init__(self, device, model_path):
        try:
            self.device = device
            self.processor = Owlv2Processor.from_pretrained(model_path, local_files_only=True)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_path, local_files_only=True).to(self.device)
        except Exception as e:
            print(e)

    def process(self, image, query_image):
        inputs = self.processor(images=image, query_images=query_image, return_tensors='pt').to(self.device)
        return inputs.pixel_values, inputs.query_pixel_values
    
    def pad_image(self, image):
        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image
    
    @torch.no_grad()
    def get_feature_map(self, pixel_values):
        return self.model.image_embedder(pixel_values)[0]
    
    @torch.no_grad()
    def get_boxes(self, feature_map, return_score=False):
        b, h, w, d = feature_map.shape
        embed = feature_map.reshape(b, h * w, d)

        box = self.model.box_predictor(embed, feature_map).view(-1, 4).cpu()
        box = ops.box_convert(box, 'cxcywh', 'xyxy')

        if return_score:
            objness = self.model.objectness_predictor(embed).cpu().squeeze()
            return box, objness
        
        else:
            return box
    
    def get_class_embed(self, feature_map):
        b, h, w, d = feature_map.shape
        embed = feature_map.reshape(b, h * w, d)

        return self.model.class_predictor(embed)[1]
        
    @torch.no_grad()
    def get_max_embed(self, query_embed, query_pred_box, query_objness, query_box, topk=10):
        query_objness = query_objness.numpy()
        query_box = torch.Tensor(query_box).view(-1, 4)

        ious = ops.box_iou(query_pred_box, query_box).cpu().squeeze().numpy()

        if topk is not None:
            indices = np.argpartition(ious, -topk)[-topk:].reshape(-1)

        else:
            indices = np.argwhere(ious > 0.7).reshape(-1)

        scores = query_objness[indices]
        max_index = indices[np.argmax(scores)]
        return query_embed.squeeze()[max_index].view(1, 1, -1)
    
    @torch.no_grad()
    def extract_query_embed(self, query_images, topk=3, query_boxes=None):
        if isinstance(query_images, Image.Image):
            query_images = [query_images]

        if query_boxes is None:
            query_boxes = [np.array([0, 0, 1.0, 1.0]) for _ in query_images]

        if isinstance(query_boxes, np.ndarray):
            query_boxes = [query_boxes]

        embeddings = []

        for query_image, query_box in zip(query_images, query_boxes):
            pixel_values = self.processor(images=query_image, return_tensors='pt').pixel_values
            feature_map = self.get_feature_map(pixel_values.to(self.device))
            class_embed = self.get_class_embed(feature_map)
            pred_box, objness = self.get_boxes(feature_map, return_score=True)

            for box in query_box:
                embeddings.append(
                    self.get_max_embed(class_embed, 
                                    pred_box,
                                    objness,
                                    box, 
                                    topk=topk
                    )
                )

        embeddings = torch.cat(embeddings, dim=1)

        return embeddings

    @torch.no_grad()
    def batch_predict(self, image, query_embeddings, classes=None, mode='cosine', calc_score='max', average_embed=False, nms_threshold=0.1, threshold=0.7):
        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values
        image_feature_map = self.get_feature_map(pixel_values.to(self.device))

        query_embeddings = torch.Tensor(query_embeddings)

        if classes is None:
            query_embeddings = {0: query_embeddings}
            
        else:
            classes_set = set(classes)
            query_embeddings = {c: 
                                query_embeddings[:, [i for i in range(len(classes)) if classes[i] == c], :] 
                                for c in classes_set}
            idx2class = {i: c for i, c in enumerate(classes_set)}
            
        if average_embed:
            for cls, embed in query_embeddings.items():
                query_embeddings[cls] = torch.mean(embed, dim=1)

        b, h, w, d = image_feature_map.shape
        image_feature_embed = image_feature_map.view(b, h * w, d)
        scores = []

        for cls, embed in query_embeddings.items():
            embed = embed.to(self.device)

            if mode == 'cosine':
                image_class_embed = self.get_class_embed(image_feature_map)
                score = cosine_similarity(image_class_embed, embed).cpu()

                score = torch.log(score / (1 - score))
                bias = 2

            else:
                score = self.model.class_predictor(image_feature_embed, embed)[0]
                bias = 10

            if calc_score == 'max':
                score = torch.max(score, dim=-1)[0].squeeze().cpu()

            else:
                score = torch.mean(score, dim=-1).squeeze().cpu()

            score = torch.sigmoid(score - bias).numpy()
            
            if isinstance(threshold, dict):
                score = score * (score >= threshold[cls])
            else:
                score = score * (score >= threshold)
                
            scores.append(score)

        scores = np.stack(scores, axis=1)
        predict_class = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        if isinstance(threshold, dict):
            indices = np.argwhere(scores > 0).reshape(-1)

        else:
            indices = np.argwhere(scores >= threshold).reshape(-1)

        pred_boxes = self.get_boxes(image_feature_map).cpu().numpy()
        pred_boxes, scores, predict_class = pred_boxes[indices], scores[indices], predict_class[indices]

        if nms_threshold is not None:
            keep = nms(pred_boxes, scores, nms_threshold)
            pred_boxes, scores, predict_class = pred_boxes[keep], scores[keep], predict_class[keep]

        if classes is not None:
            predict_class = [idx2class[x] for x in predict_class]

            return pred_boxes, scores, predict_class

        return pred_boxes, scores
    
