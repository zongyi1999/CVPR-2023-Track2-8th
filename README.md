# The 8-th solution to CVPR 2023 foundation model challenge-Track2 Competition 
We follow the code described in the CVPR2023 paper titled ["Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval"](https://github.com/anosorae/IRRA)
Requirements
## Usage
### Requirements
we use single RTX3090 24G GPU for training and evaluation. 
```
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```

### Prepare Datasets
Organize your data folder as follows:
```
|-- data/
|       |-- train
|           |-- train_images/
|           |-- train_label.txt
|       |-- test
|           |-- test_images/
|           |-- test_label.txt
|       |-- val
|           |-- val_images/
|           |-- val_label.txt
|-- logs/
|       |-- retrival1
|       |-- retrival2
```
The logs can be download from:""

## Training

```
CUDA_VISIBLE_DEVICES=0 python train.py \
--name iira \
--img_aug \
--batch_size 96 \
--MLM \
--loss_names 'sdm+id' \
--dataset_name 'ImageRetri' \
--root_dir './data' \
--num_epoch 30 \
--output_dir logs/retrival1

CUDA_VISIBLE_DEVICES=0 python train.py \
--name iira \
--img_aug \
--batch_size 96 \
--MLM \
--loss_names 'sdm+id+oim' \
--dataset_name 'ImageRetri' \
--root_dir './data' \
--num_epoch 30 \
--output_dir logs/retrieval2
```
## Testing

```python
CUDA_VISIBLE_DEVICES=0 python inference_person.py --config_file ./logs/retrival1/configs.yaml
CUDA_VISIBLE_DEVICES=0 python inference_person.py --config_file ./logs/retrival2/configs.yaml
python ensemble.py
```