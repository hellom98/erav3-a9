# Train ResNet50 from Scratch on Imagenet

## Installation
> pip install torch torchvision tqdm torch-lr-finder albumentations boto3

## Commands

Start training
```
nohup python train_resnet50.py \
    --data-dir imagenet-mini \
    --output-dir outputs \
    --batch-size 128 \
    --epochs 100 \
    --save-freq 5 \
    --s3-bucket training-imagenetmini \
    --s3-prefix outputs &
```

Resume training
```
nohup python train_resnet50.py \
    --resume outputs/checkpoint_epoch_10.pth \
    --data-dir imagenet-mini \
    --output-dir outputs \
    --s3-bucket training-imagenetmini \
    --s3-prefix outputs &
```

View Progress
```
tail -f nohup.out
```
