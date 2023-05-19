import os
import torch
import numpy as np
import os.path as op
import os
import json
import collections
from tqdm import tqdm
if __name__ == '__main__':
    img_dir = "./data/test/test_images"
    img_paths = []
    img_pids = []
    img_names = []
    female_list = collections.defaultdict(list)
    count = 0
    for i, img_path in enumerate(os.listdir(img_dir)):
        img_pids.append(i)
        img_paths.append(op.join(img_dir, img_path))
        img_names.append(img_path)
        count+=1
    caption_pids = []
    captions = []
    captions_all = []
    file = open("./data/test/test_text.txt","r")
    caption_lines = file.readlines()
    for i, caption in enumerate(caption_lines):
        caption_pids.append(i)
        captions_all.append(caption.strip())
        captions.append(caption.strip())
    similarity1 = torch.load("retrieval1.pt")
    similarity2 = torch.load("retrieval2.pt")
    similarity = similarity1+similarity2
    similarity_argsort = np.argsort(-similarity, axis=1)
    print('similarity_argsort.shape', similarity_argsort.shape)
    topk = 10
    result_list = []
    for i in range (len(similarity_argsort)):
        dic = {'text': captions_all[i], 'image_names': []}
        for j in range(topk):
            dic['image_names'].append(img_names[similarity_argsort[i][j]])
        result_list.append(dic)
    with open('ensemble.json', 'w') as f:
        f.write(json.dumps({'results': result_list}, indent=4))