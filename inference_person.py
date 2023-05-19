from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op
import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import json
from datasets import build_dataloader, build_transforms
from datasets.bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset
import collections
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs
from retri_utils.compute_dist import re_ranking
from tqdm import tqdm


def batch_torch_topk_self(qf, gf, k1, N=1000, query = False):
    # m = qf.shape[0]
    # n = gf.shape[0]
    m = gf.shape[0]
    n = qf.shape[0]
    dist_mat = []
    initial_rank = []
    for j in tqdm(range(n // N + 1)):
        # temp_gf = gf[j * N:j * N + N]
        temp_qf = qf[j * N:j * N + N]
        if len(temp_qf)==0:
            continue
        temp_qd = []
        for i in range(m // N + 1):
            # temp_qf = qf[i * N:i * N + N]
            temp_gf = gf[i * N:i * N + N]
            temp_d = torch.mm(temp_gf, temp_qf.t())
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        temp_qd = temp_qd.t()
        value, rank = torch.topk(temp_qd, k=k1, dim=1, largest=True, sorted=True)
        # rank = torch.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1]
        if query:
            dist_mat.append(value)
        initial_rank.append(rank)
    del value
    del rank
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory

    initial_rank = torch.cat(initial_rank, dim=0)#.cpu().numpy()
    if query:
        dist_mat = torch.cat(dist_mat, dim=0)#.cpu().numpy()
        return dist_mat, initial_rank
    return initial_rank

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    args = parser.parse_args()
    save_name = args.config_file.split("/")[-2]
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)

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
    num_workers = args.num_workers
    test_transforms = build_transforms(img_size=args.img_size,
                                        is_train=False)
    test_img_set = ImageDataset(img_pids, img_paths,
                                test_transforms)
    test_txt_set = TextDataset(caption_pids,
                               captions,
                               text_length=args.text_length)

    test_img_loader = DataLoader(test_img_set,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    test_txt_loader = DataLoader(test_txt_set,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    model.eval()
    qfeats, gfeats, qids, gids = evaluator._compute_embedding(model)
    pos_score = []
    qfeats = F.normalize(qfeats, p=2, dim=1)
    gfeats = F.normalize(gfeats, p=2, dim=1)
    similarity = qfeats @ gfeats.t()
    similarity = similarity.cpu().numpy()
    torch.save(similarity, save_name+".pt")
    # torch.save(similarity, "retrieval2.pt")
    # similarity1 = torch.load("similarity_text_vit-l-bn.pt")
    # similarity2 = torch.load("similarity_text_vit-l.pt")
    # similarity = similarity1+similarity2
    # similarity_argsort = np.argsort(-similarity, axis=1)
    # print('similarity_argsort.shape', similarity_argsort.shape)
    # topk = 10
    # result_list = []
    # for i in range (len(similarity_argsort)):
    #     dic = {'text': captions_all[i], 'image_names': []}
    #     for j in range(topk):
    #         dic['image_names'].append(img_names[similarity_argsort[i][j]])
    #     result_list.append(dic)
    # with open('infer_all_vit-L-bn_ensemble.json', 'w') as f:
    #     f.write(json.dumps({'results': result_list}, indent=4))