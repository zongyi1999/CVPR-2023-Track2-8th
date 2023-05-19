import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset
import collections
import json



class ImageRetri(BaseDataset):
    """
    CUHK-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions: 
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format: 
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = ''

    def __init__(self, root='', verbose=True):
        super(ImageRetri, self).__init__()
        
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = self.dataset_dir

        self.train_annos = op.join(self.dataset_dir, 'train/train_label.txt')
        # self.train_annos = op.join(self.dataset_dir, 'train/all_label.txt')
        self.val_annos = op.join(self.dataset_dir, 'val/val_label.txt')
        self.test_annos = op.join(self.dataset_dir, 'val/val_label.txt')
        self._check_before_run()
        #"/media/backup/competition/image_retrieval/data/data203278/train/train_label.txt"
        self.train, self.train_id_container = self._process_anno(self.train_annos, op.join(self.dataset_dir, 'train/train_images'),training=True)
        # self.train, self.train_id_container = self._process_anno(self.train_annos, op.join(self.dataset_dir, 'train/all'),training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos,op.join(self.dataset_dir, 'val/val_images'),)
        self.val, self.val_id_container = self._process_anno(self.val_annos,op.join(self.dataset_dir, 'val/val_images'),)
        if verbose:
            self.logger.info("=> ImageRetri Images and Captions are loaded")
            self.show_dataset_info()

    def _process_anno(self, ann_path, img_dir, training=False):
        attributes = []
        img_paths = []
        captions = []
        labels = []
        att_dict = collections.defaultdict(int)
        file = open(ann_path,"r")
        lines = file.readlines()
        for line in lines:
            img, att, cap = line.split("$")
            if  img.startswith("0"):continue
            # attributes.append(cap.strip().split(".")[1])
            # att = attribute_dict[img]
            attributes.append(att)
            # attributes.append(att[:19])
        for i, att in enumerate(set(attributes)):
            att_dict[att]= i
        for line in lines:
            img, att, cap = line.split("$")
            if  img.startswith("0"):continue
            # att = attribute_dict[img]
            img_path = op.join(img_dir, img)
            img_paths.append(img_path)
            captions.append(cap.strip())
            # captions.append(cap.strip().split(".")[1])
            labels.append(att_dict[att])
            # labels.append(att_dict[att[:19]])
            # labels.append(att_dict[cap.strip().split(".")[1]])

        if training:
            dataset = []
            pid_container = set()
            image_id = 0
            for pid,  img_path, caption in zip(labels, img_paths, captions):
                dataset.append((pid, image_id, img_path, caption))
                pid_container.add(pid)
                image_id+=1
        else:
            pid_container = set(labels)
            dataset = {}
            img_paths = img_paths
            captions = captions
            image_pids = labels
            caption_pids = labels
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
        return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))