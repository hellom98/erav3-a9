import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import time
from torch_lr_finder import LRFinder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from albumentations.core.transforms_interface import ImageOnlyTransform
import argparse
import os
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import boto3
import io
import json
from botocore.exceptions import ClientError
import s3fs
import multiprocessing

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available(
) else 'cuda' if torch.cuda.is_available() else 'cpu')

# Initialize scaler based on device type
scaler = None
if device.type in ['cuda', 'mps']:
    scaler = GradScaler(init_scale=1024)


class Cutout(ImageOnlyTransform):
    def __init__(self, num_holes, max_h_size, max_w_size, p=0.5, always_apply=False):
        super().__init__(p, always_apply)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def apply(self, image, **params):
        h, w, _ = image.shape
        mask = np.ones((h, w), dtype=np.float32)

        for _ in range(self.num_holes):
            hole_h = random.randint(1, self.max_h_size)
            hole_w = random.randint(1, self.max_w_size)

            y = random.randint(0, h)
            x = random.randint(0, w)

            y1 = np.clip(y - hole_h // 2, 0, h)
            y2 = np.clip(y + hole_h // 2, 0, h)
            x1 = np.clip(x - hole_w // 2, 0, w)
            x2 = np.clip(x + hole_w // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        image = image * mask[:, :, np.newaxis]
        return image

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")


# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# Data augmentation and normalization
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.1, p=0.5),
    Cutout(num_holes=1, max_h_size=56, max_w_size=56, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Convert grayscale to RGB if needed
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # If grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 1:  # If grayscale with single channel
            img_array = np.concatenate([img_array] * 3, axis=-1)
        return self.transform(image=img_array)['image']


# Load ImageNet-mini dataset
train_dataset = datasets.ImageFolder(
    root='imagenet-mini/train',
    transform=AlbumentationsTransform(train_transform)
)

val_dataset = datasets.ImageFolder(
    root='imagenet-mini/val',
    transform=AlbumentationsTransform(val_transform)
)

train_sampler = torch.utils.data.RandomSampler(train_dataset)

train_loader = None
val_loader = None

# Initialize model
model = models.resnet50(weights=None)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.9,
#                       momentum=0.9, weight_decay=1e-2)
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)


def find_lr(model, train_loader, batch_size, optimizer, criterion, device):
    # Create a deep copy of the model for LR finding
    import copy
    model_copy = copy.deepcopy(model)
    optimizer_copy = copy.deepcopy(optimizer)

    # Create a subset of the training dataset (1%)
    subset_size = int(len(train_loader.dataset) * 0.01)
    subset_indices = np.random.choice(
        len(train_loader.dataset), subset_size, replace=False)
    subset = torch.utils.data.Subset(train_loader.dataset, subset_indices)
    subset_loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    lr_finder = LRFinder(model_copy, optimizer_copy,
                         criterion, device=device, grad_scaler=scaler)
    lr_finder.range_test(subset_loader, end_lr=0.1,
                         num_iter=100, step_mode='exp')

    ax, best_lr = lr_finder.plot(suggest_lr=True)
    ax.figure.savefig('lr_finder_plot.png')

    lr_finder.reset()
    return best_lr

# Setup logging


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ResNet50 on ImageNet-mini')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--data-dir', type=str,
                        default='imagenet', help='Path to dataset')
    parser.add_argument('--output-dir', type=str,
                        default='outputs', help='Directory to save outputs')
    parser.add_argument('--s3-bucket', type=str,
                        help='S3 bucket name (optional)')
    parser.add_argument('--s3-prefix', type=str,
                        default='checkpoints', help='S3 prefix for checkpoints')
    return parser.parse_args()


def save_checkpoint(state, is_best, filename, output_dir, args):
    # First save locally
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint saved locally: {checkpoint_path}")

    # Upload to S3 only if bucket is specified
    if args.s3_bucket:
        s3_path = f"{args.s3_prefix}/{filename}"
        save_torch_to_s3(state, args.s3_bucket, s3_path)

    if is_best:
        best_filename = 'model_best.pth'
        best_path = os.path.join(output_dir, best_filename)
        torch.save(state, best_path)
        logging.info(f"Best model saved locally: {best_path}")

        # Upload best model to S3 only if bucket is specified
        if args.s3_bucket:
            s3_best_path = f"{args.s3_prefix}/{best_filename}"
            save_torch_to_s3(state, args.s3_bucket, s3_best_path)
            logging.info(
                f"Best model saved to s3://{args.s3_bucket}/{s3_best_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, args):
    try:
        if checkpoint_path.startswith('s3://'):
            if not args.s3_bucket:
                raise ValueError(
                    "S3 checkpoint specified but --s3-bucket not provided")
            # Parse S3 URI
            bucket = checkpoint_path.split('/')[2]
            s3_path = '/'.join(checkpoint_path.split('/')[3:])
            return load_checkpoint_from_s3(bucket, s3_path, model, optimizer, scheduler)
        else:
            # Local file loading logic
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}")

            logging.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            return (checkpoint['epoch'], checkpoint['best_acc'],
                    checkpoint['train_acc'], checkpoint['train_loss'])
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise


def get_s3_client():
    """Initialize and return S3 client"""
    return boto3.client('s3')


def upload_to_s3(local_path, bucket, s3_path):
    """Upload a file to S3"""
    try:
        s3_client = get_s3_client()
        s3_client.upload_file(local_path, bucket, s3_path)
        logging.info(
            f"Successfully uploaded {local_path} to s3://{bucket}/{s3_path}")
        return True
    except ClientError as e:
        logging.error(f"Error uploading to S3: {e}")
        return False


def save_torch_to_s3(state, bucket, s3_path):
    """Save PyTorch state directly to S3 without saving to disk first"""
    try:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)

        s3_client = get_s3_client()
        s3_client.upload_fileobj(buffer, bucket, s3_path)
        logging.info(
            f"Successfully uploaded checkpoint to s3://{bucket}/{s3_path}")
        return True
    except ClientError as e:
        logging.error(f"Error uploading to S3: {e}")
        return False


def load_checkpoint_from_s3(bucket, s3_path, model, optimizer, scheduler):
    """Load checkpoint from S3"""
    try:
        s3_client = get_s3_client()
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket, s3_path, buffer)
        buffer.seek(0)

        checkpoint = torch.load(buffer, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logging.info(
            f"Successfully loaded checkpoint from s3://{bucket}/{s3_path}")
        return (checkpoint['epoch'], checkpoint['best_acc'],
                checkpoint['train_acc'], checkpoint['train_loss'])
    except ClientError as e:
        logging.error(f"Error loading checkpoint from S3: {e}")
        raise


def main():
    global model, optimizer, scheduler, train_loader, val_loader, device
    args = parse_args()

    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    # Log S3 configuration if provided
    if args.s3_bucket:
        logging.info(
            f"S3 backup enabled - Bucket: {args.s3_bucket}, Prefix: {args.s3_prefix}")
    else:
        logging.info("Running in local-only mode (no S3 backup)")

    logging.info(f"Training with configuration: {vars(args)}")

    # Update data paths
    global train_dataset, val_dataset
    train_dataset.root = os.path.join(args.data_dir, 'train')
    val_dataset.root = os.path.join(args.data_dir, 'val')

    start_epoch = 0
    best_acc = 0

    # Initialize data loaders with argument batch size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Find the best learning rate
    # logging.info("Finding best learning rate...")
    # best_lr = find_lr(model, train_loader, args.batch_size,
                      # optimizer, criterion, device)
    # logging.info(f"Best learning rate found: {best_lr}")

    # Learning rate schedulers
    # one_cycle_scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=best_lr,
    #     epochs=args.epochs, 
    #     steps_per_epoch=len(train_loader)
    # )
    
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=10  # Warm-up for first 10 epochs
    )


    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - 10  # Decay starts after warm-up
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[10]  # Switch to CosineAnnealingLR at epoch 10
    )

    # Resume from checkpoint if specified
    if args.resume:
        try:
            if args.resume.startswith('s3://'):
                start_epoch, best_acc, train_acc, train_loss = load_checkpoint(
                    args.resume, model, optimizer, scheduler, args)
            else:
                start_epoch, best_acc, train_acc, train_loss = load_checkpoint(
                    args.resume, model, optimizer, scheduler, args)
            logging.info(f"Resumed from epoch {start_epoch + 1}")
            start_epoch += 1
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            sys.exit(1)

    # Training loop
    try:
        checkpoint = None  # Initialize checkpoint variable
        for epoch in range(start_epoch, args.epochs):
            train_acc, train_loss = train_epoch(epoch, args.epochs)
            scheduler.step()
            top1_acc, top5_acc = validate()

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc,
                'train_loss': train_loss,
                'best_acc': best_acc,
                'val_top1_acc': top1_acc,
                'val_top5_acc': top5_acc
            }


            is_best = top1_acc > best_acc
            if is_best:
                best_acc = top1_acc

            # Save checkpoint based on frequency
            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(
                    checkpoint,
                    is_best,
                    f'checkpoint_epoch_{epoch+1}.pth',
                    args.output_dir,
                    args
                )

            logging.info(
                f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, "
                f"Loss={train_loss:.4f}, Top-1={top1_acc:.2f}%, "
                f"Top-5={top5_acc:.2f}%"
            )

    except Exception as e:
        logging.error(f"Training interrupted: {e}")
        # Save emergency checkpoint
        if checkpoint is not None:  # Only save if checkpoint exists
            save_checkpoint(
                checkpoint,
                False,
                'interrupt_checkpoint.pth',
                args.output_dir,
                args
            )
        raise


def train_epoch(epoch, epochs):
    global args
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        # Use AMP for CUDA and MPS devices
        if device.type in ['cuda', 'mps']:
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            # Clip gradients (applied to scaled gradients)
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


        # scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        elapsed_time = time.time() - start_time
        eta = elapsed_time / (batch_idx + 1) * \
            (len(train_loader) - batch_idx - 1)

        pbar.set_postfix({
            'loss': f'{avg_loss:.3f}',
            'acc': f'{accuracy:.2f}%',
            'lr': f'{current_lr:.6f}',
            'eta': f'{eta:.0f}s'
        })
    return accuracy, avg_loss


def validate(top_k=(1, 5)):
    model.eval()
    correct_k = [0] * len(top_k)
    total = 0

    try:
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validating'):
                images, targets = images.to(device), targets.to(device)

                # Use AMP for inference
                if device.type in ['cuda', 'mps']:
                    with autocast(device_type=device.type):
                        outputs = model(images)
                else:
                    outputs = model(images)

                _, pred = outputs.topk(max(top_k), 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))

                for i, k in enumerate(top_k):
                    correct_k[i] += correct[:
                                            k].reshape(-1).float().sum(0).item()
                total += targets.size(0)
    except Exception as e:
        logging.error(f"Error during validation: {e}")
        raise

    return [100. * c / total for c in correct_k]


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    print(f"Using device: {device}")
    main()
