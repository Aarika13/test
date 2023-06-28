import os
import json
import pytesseract
from PIL import Image
from pathlib import Path
from uuid import uuid4
import requests
# ^.*\.(jpg|jpeg|png|gif|bmp)$
# ./scripts/serve_local_files.sh /home/aarika/Desktop/test/data/upload/
LOCAL_FILES_DOCUMENT_ROOT=True 
LEVELS = {
    'page_num': 1,
    'block_num': 2,
    'par_num': 3,
    'line_num': 4,
    'word_num': 5
}

# tesseract output levels for the level of detail for the bounding boxes
def create_image_url(filename):
    # return f'http://localhost:8080/{filename}'
    # return f'http://localhost:8080/data/local-files/?d={filename}'
    relative_path = os.path.relpath(filename, "/home/aarika/Desktop/test")
    return f'http://localhost:8080/data/local-files/?d=test/{relative_path}'

def convert_to_ls(image, tesseract_output, per_level='block_num'):
    image_width, image_height = image.size
    per_level_idx = LEVELS[per_level]
    results = []
    all_scores = []

    for i, level_idx in enumerate(tesseract_output['level']):
        if level_idx == per_level_idx:
            bbox = {
                'x': 100 * tesseract_output['left'][i] / image_width,
                'y': 100 * tesseract_output['top'][i] / image_height,
                'width': 100 * tesseract_output['width'][i] / image_width,
                'height': 100 * tesseract_output['height'][i] / image_height,
                'rotation': 0
            }

            words, confidences = [], []
            for j, curr_id in enumerate(tesseract_output[per_level]):
                if curr_id != tesseract_output[per_level][i]:
                    continue
                word = tesseract_output['text'][j]
                confidence = tesseract_output['conf'][j]
                words.append(word)
                if confidence != '-1':
                    confidences.append(float(confidence) / 100.0)

            text = ' '.join(words).strip()
            if not text:
                continue
            region_id = str(uuid4())[:10]
            score = sum(confidences) / len(confidences) if confidences else 0

            bbox_result = {
                'id': region_id,
                'from_name': 'bbox',
                'to_name': 'image',
                'type': 'rectangle',
                'value': bbox
            }

            transcription_result = {
                'id': region_id,
                'from_name': 'transcription',
                'to_name': 'image',
                'type': 'textarea',
                'value': dict(text=[text], **bbox),
                'score': score
            }

            results.extend([bbox_result, transcription_result])
            all_scores.append(score)

    return {
        'data':{
            # 'image': create_image_url(image.filename)
            'ocr': create_image_url(image.filename)
        },
        'predictions':[{
            'result': results,
            'scores': sum(all_scores) / len(all_scores) if all_scores else 0
        }]
        # 'annotations': results,
        # 'scores': [sum(all_scores) / len(all_scores)] if all_scores else [],
        # 'image': create_image_url(image.filename)
    }

tasks = []

# Collect the receipt images from the image directory
image_dir = '/home/aarika/Desktop/test/data/upload'  # Specify the path to the image directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        with Image.open(image_path) as image:
            tesseract_output = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            task = convert_to_ls(image, tesseract_output, per_level='block_num')
            tasks.append(task)

#new one included
output_data = {
    'tasks': tasks
}

output_json = json.dumps(output_data, indent=2)

# Save tasks to JSON file
with open('ocr_tasks.json', mode='w') as f:
    json.dump(tasks, f, indent=2)
    

