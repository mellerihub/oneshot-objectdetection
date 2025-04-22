import os
import numpy as np
from datetime import datetime, timezone

def output(pipeline: dict):
    logger = pipeline['logger']
    logger.debug("output")

    '''
    The inference results can lead to further applied work.
    '''



def inference(pipeline: dict):
    from src.det.inference import run_inference, NoLog
    from src.model.owl import OWLV2
    
    EMBEDDING_PATH = pipeline['model']['workspace']

    logger = pipeline['logger']

    args = pipeline['inference']['argument']
    device = args['device']

    image_files = pipeline['input']['result']['image_files']

    if "threshold" in args:
        if isinstance(args['threshold'], str):
            ret = {}
            for x in args['threshold'].split(','):
                parts = x.split(':')
                if len(parts) == 2:
                    class_name = parts[0].strip()
                    try:
                        threshold_value = float(parts[1].strip())
                        ret[class_name] = threshold_value
                    except ValueError:
                        print(f"경고: 임계값 '{parts[1].strip()}'를 실수로 변환할 수 없습니다.")
            threshold = ret
        else:
            threshold = args['threshold']

        args['threshold'] = threshold

    args['output_dir'] = pipeline['artifact']['workspace']

    with NoLog():
        base_model = './src/owlv2-base-patch16-ensemble'
        model = OWLV2(device=device, model_path=base_model)
        
    embeddings = np.load(os.path.join(EMBEDDING_PATH, 'query_embed.npy'))
    classes = np.load(os.path.join(EMBEDDING_PATH, 'classes.npy'))

    result = run_inference(args, image_files, classes, model, embeddings)

    print(result)
    return {
            'summary': {
                'note': f'Inference has been completed (date: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")})'
                }
        }

def train(pipeline: dict):
    from src.det.train import set_fewshot_data, NoLog
    from src.model.owl import OWLV2

    logger = pipeline['logger']

    args = pipeline['train']['argument']
    device = args['device']

    query_data = pipeline['input']['result']['query_data']
    
    images, boxes, classes = set_fewshot_data(query_data)
    
    with NoLog():
        base_model = './src/owlv2-base-patch16-ensemble'
        model = OWLV2(device=device, model_path=base_model)

    logger.info('Extracting Query Embeddings')

    query_embed = model.extract_query_embed(images, query_boxes=boxes).cpu().numpy()
    
    del model

    EMBEDDING_PATH = pipeline['model']['workspace']

    np.save(os.path.join(EMBEDDING_PATH, 'query_embed.npy'), query_embed)
    np.save(os.path.join(EMBEDDING_PATH, 'classes.npy'), classes)

    return {
        'summary': {
            'note': f'Embedding of the Fewshot learner has been completed (date: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")})'
            }
    }

def input(pipeline : dict):
    from src.input.input import get_image_files, get_query_data
    logger = pipeline['logger']
    pipeline_name = pipeline['name']
    data = dict()

    data_path = pipeline['dataset']['workspace']

    logger.info(f'{data_path}')
    
    if pipeline_name == 'inference':
        data['image_files'] = get_image_files(data_path)
    else:
        data['query_data'] = get_query_data(data_path)
    
    logger.info(f'Success load {data_path}')

    return data
