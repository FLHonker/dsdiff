import os
import json
import random

COCO_caption = '/data/yaliu/datasets/coco/annotations/captions_val2017.json'
save_file = './coco_caption2017_256.txt'

prompts = []
with open(COCO_caption, 'r', encoding='utf8') as fp:
    json_data = json.load(fp)
    annotations = json_data['annotations']
    for item in annotations:
        prompts.append(item['caption'])

random.shuffle(prompts)
with open(save_file, 'w', encoding='utf8') as fp:
    for i in prompts[:256]:
        fp.write(i.split('\n')[0] + '\n')
    print('Extracted prompts and saved to file!')