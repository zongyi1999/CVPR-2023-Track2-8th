from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import random

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.loss_mse = nn.CrossEntropyLoss(reduction="mean")
        if 'oim' in self.current_task:
            self.image_oim = objectives.OIMLoss(self.embed_dim,self.num_classes)
            self.text_oim = objectives.OIMLoss(self.embed_dim,self.num_classes)
        if 'supcon' in args.loss_names:
            # self.supcon = objectives.SupConLoss("cuda")
            self.triplet = objectives.TripletLoss()
        if 'id' in args.loss_names:
            self.bottleneck = nn.BatchNorm1d(self.embed_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            self.bottleneck.apply(weights_init_kaiming)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
#('bn', nn.BatchNorm1d(self.embed_dim*2)),
            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim*2, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, 2))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'oim' in self.current_task:
            # oim_img, oim_img_text = self.image_oim(i_feats, t_feats, batch['pids'])
            # oim_text, oim_text_img = self.text_oim(t_feats, i_feats, batch['pids'])
            # ret.update({'oim_img_loss': oim_img})
            # ret.update({'img_text_loss': oim_img_text})
            # ret.update({'oim_text_loss': oim_text})
            # ret.update({'text_img_loss': oim_text_img})
            x_neg = []
            associate_loss=0
            for k in range(len(batch['pids'])):
                ori_asso_ind = torch.nonzero(batch['pids'] != batch['pids'][k]).squeeze(-1)
                x_neg.append(t_feats[random.choice(ori_asso_ind)]) #positive loss
            pos_features = torch.cat((i_feats, t_feats),dim=-1)
            x_neg = torch.vstack(x_neg)
            neg_features = torch.cat((i_feats, x_neg),dim=-1)
            labels_itm = torch.cat((  torch.ones((len(batch['pids']))  ), torch.zeros((len(batch['pids']))) )).cuda()
            combine_features = torch.cat((pos_features, neg_features), dim=0 )
            loss_positive = self.loss_mse(combine_features, labels_itm)


        if 'supcon' in self.current_task:
            # supconloss = self.supcon(i_feats, t_feats,batch['pids'], batch['pids'])
            supconloss = self.triplet(i_feats, t_feats, batch['pids'],)
            ret.update({'supconloss': supconloss})
        if 'id' in self.current_task:
            image_logits = self.classifier(self.bottleneck(i_feats.half())).float()
            text_logits = self.classifier(self.bottleneck(i_feats.half())).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})
            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            # mlm_ids = batch['mlm_ids']

            # mlm_feats = self.base_model.encode_text(mlm_ids)

            # x = self.cross_former(mlm_feats, image_feats, image_feats)

            # x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            # scores = x.float().reshape(-1, self.args.vocab_size)
            # mlm_labels = batch['mlm_labels'].reshape(-1)
            # ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            # pred = scores.max(1)[1]
            # mlm_label_idx = torch.nonzero(mlm_labels)
            # acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            # ret.update({'mlm_acc': acc})


            x_neg = []
            associate_loss=0
            for k in range(len(batch['pids'])):
                ori_asso_ind = torch.nonzero(batch['pids'] != batch['pids'][k]).squeeze(-1)
                x_neg.append(t_feats[random.choice(ori_asso_ind)]) #positive loss
            pos_features = torch.cat((i_feats, t_feats),dim=-1)
            x_neg = torch.vstack(x_neg)
            neg_features = torch.cat((i_feats, x_neg),dim=-1)
            labels_itm = torch.cat( (torch.ones(len(batch['pids'])), torch.zeros(len(batch['pids'])) ) ).long().cuda()
            combine_features = torch.cat((pos_features, neg_features), dim=0 )
            predict_logits = self.mlm_head(combine_features.half())
            loss_positive = self.loss_mse(predict_logits, labels_itm)

            pos_pred = torch.argmax(predict_logits, dim=1)

            mlm_acc = (pos_pred == labels_itm).float().mean()
            ret.update({'mlm_loss': loss_positive})
            ret.update({'mlm_acc': mlm_acc})

        return ret
    def after_bn(self, embedding):
        return self.bottleneck(embedding.half()).float()
    def cal_pos(self, combine_features):
        predict_logits = self.mlm_head(combine_features.half())
        return predict_logits
        

def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
