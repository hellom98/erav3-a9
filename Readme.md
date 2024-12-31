# Train ResNet50 from Scratch on Imagenet

## Datasets
### Imagenet mini
https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000

> kaggle datasets download ifigotin/imagenetmini-1000

### Imagenet

https://www.kaggle.com/datasets/mayurmadnani/imagenet-dataset

> kaggle datasets download mayurmadnani/imagenet-dataset

#### Preparation

https://www.kaggle.com/c/imagenet-object-localization-challenge/

> kaggle competitions download -c imagenet-object-localization-challenge

Extract
```bash
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

## Pre-requisites

### Installation
> pip install torch torchvision tqdm torch-lr-finder albumentations boto3 kaggle s3fs

### Configuration

> aws configure

> touch ~/.kaggle/kaggle.json

## Commands

Start training
```bash
nohup python train_resnet50.py \
    --data-dir imagenet \
    --output-dir outputs/imagenet/ \
    --batch-size 512 \
    --epochs 100 \
    --save-freq 5 \
    --s3-bucket training-imagenet \
    --s3-prefix outputs &
```

Resume training
```bash
nohup python train_resnet50.py \
    --resume outputs/checkpoint_epoch_10.pth \
    --data-dir imagenet \
    --output-dir outputs/imagenet/ \
    --s3-bucket training-imagenet \
    --s3-prefix outputs &
```

View Progress
```bash
tail -f nohup.out
```

## Gradio App on HuggingFace
### https://huggingface.co/spaces/KamleshShrama/ERAV3-ResNet50-Imagnet
