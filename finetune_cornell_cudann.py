import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

# Assuming you have these imports available from your existing code
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output
from utils.dataset_processing import evaluation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_distributed():
    """Initialize distributed training"""
    # Check if using distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.warning("Distributed environment variables not found. Using single GPU.")
        rank = 0
        world_size = 1
        local_rank = 0

    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        logger.info(f"Initialized process group: rank {rank}/{world_size}, local_rank: {local_rank}")
    
    return rank, world_size, local_rank, device

def finetune_nbmod_from_cornell():
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    is_main_process = rank == 0
    
    # Parameters
    input_channels = 1  # Change if using RGB
    batch_size = 48 // world_size  # Scale batch size based on number of GPUs
    val_batches = 3510
    epochs = 1000
    nbmod_dataset_path = '/home/data/maa1446/nbmod/combined/'  # Update with your path
    cornell_model_path = '/home/data/maa1446/ggcnn/output/models/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'  # Update with your model path
    output_dir = '/home/data/maa1446/ggcnn/output/models/cornell_finetuned_models'
    learning_rate = 1e-5  # Lower learning rate for fine-tuning

    # Create output directory (only on main process)
    if is_main_process:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_folder = os.path.join(output_dir, f'cornell_to_nbmod_{timestamp}')
        os.makedirs(save_folder, exist_ok=True)
    else:
        save_folder = None

    # Broadcast save_folder to all processes if needed
    if world_size > 1:
        if is_main_process:
            save_folder_len = torch.tensor(len(save_folder), device=device)
            save_folder_bytes = torch.tensor([ord(c) for c in save_folder], device=device)
        else:
            save_folder_len = torch.tensor(0, device=device)
            save_folder_bytes = None
        
        dist.broadcast(save_folder_len, 0)
        
        if not is_main_process:
            save_folder_bytes = torch.zeros(save_folder_len.item(), dtype=torch.long, device=device)
        
        dist.broadcast(save_folder_bytes, 0)
        
        if not is_main_process:
            save_folder = ''.join([chr(i) for i in save_folder_bytes.cpu().numpy()])
    
    # Load Dataset
    if is_main_process:
        logger.info('Loading NBMOD Dataset...')
    
    Dataset = get_dataset('nbmod')
    
    # Use consistent random seed for dataset splits
    shuffle_seed = 42
    
    # Create datasets with train/val split
    train_dataset = Dataset(nbmod_dataset_path, start=0.0, end=0.9, 
                           shuffle_seed=shuffle_seed,
                           random_rotate=True, random_zoom=True,
                           include_depth=1, include_rgb=0)
    
    val_dataset = Dataset(nbmod_dataset_path, start=0.9, end=1.0, 
                         shuffle_seed=shuffle_seed,
                         random_rotate=True, random_zoom=True,
                         include_depth=1, include_rgb=0)
    
    # Create distributed samplers
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank
        )
    else:
        train_sampler = None
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Only rank 0 process validates
    if is_main_process:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
    else:
        val_loader = None
    
    if is_main_process:
        logger.info('Done loading datasets')
    
    # Initialize network
    if is_main_process:
        logger.info('Loading Network...')
    
    ggcnn = get_network('ggcnn')
    net = ggcnn(input_channels=input_channels)
    
    # Load pre-trained Cornell model (on main process)
    if is_main_process:
        try:
            # Try loading the entire model
            logger.info(f"Loading pre-trained model from {cornell_model_path}")
            pretrained_model = torch.load(cornell_model_path, map_location=device, weights_only=False)
            
            if isinstance(pretrained_model, nn.Module):
                # If we loaded the full model
                net = pretrained_model
            elif isinstance(pretrained_model, dict) and 'model_state_dict' in pretrained_model:
                # If we loaded a checkpoint with state_dict
                net.load_state_dict(pretrained_model['model_state_dict'])
            elif isinstance(pretrained_model, dict) and 'state_dict' in pretrained_model:
                # Another common format
                net.load_state_dict(pretrained_model['state_dict'])
            else:
                # Try direct loading as state_dict
                net.load_state_dict(pretrained_model)
                
            logger.info("Successfully loaded pre-trained model")
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {e}")
            logger.info("Starting with randomly initialized weights")
            return
    
    # Move to device
    net = net.to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank)
        # Sync parameters
        for param in net.parameters():
            dist.broadcast(param.data, 0)
    
    # Setup optimizer with smaller learning rate for fine-tuning
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    
    # Initialize tracking variables
    best_iou = 0.0
    
    # Create arrays to track metrics
    train_losses = []
    val_ious = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        if is_main_process:
            logger.info(f'Beginning Epoch {epoch:02d}')
        
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_epoch(net, device, train_loader, optimizer, world_size, epoch)
        
        # Aggregate train loss across processes
        if world_size > 1:
            train_loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / world_size
        
        if is_main_process:
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_iou = validate_epoch(net, device, val_loader, val_batches, epoch)
            val_losses.append(val_loss)
            val_ious.append(val_iou)
            
            # Update learning rate based on IoU
            scheduler.step(val_iou)
            
            # Save model if it's the best so far
            if val_iou > best_iou:
                best_iou = val_iou
                logger.info(f'New best model with IoU = {best_iou:.4f}')
                
                # Get state dict (handle DDP wrapper)
                if isinstance(net, DDP):
                    model_state_dict = net.module.state_dict()
                else:
                    model_state_dict = net.state_dict()
                # Save model and state dict
                torch.save(net.module, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, best_iou)))
                torch.save(model_state_dict, os.path.join(save_folder, f'epoch_{epoch:02d}_iou_{best_iou:.2f}_statedict.pt'))

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                # Get state dict (handle DDP wrapper)
                if isinstance(net, DDP):
                    model_state_dict = net.module.state_dict()
                else:
                    model_state_dict = net.state_dict()
                
                # Save model and state dict
                torch.save(net.module, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, val_iou)))
                torch.save(model_state_dict, os.path.join(save_folder, f'epoch_{epoch:02d}_iou_{val_iou:.2f}_statedict.pt'))
        
        # Broadcast IoU to all processes for synchronized learning rate scheduling
        if world_size > 1:
            val_iou_tensor = torch.tensor([val_iou if is_main_process else 0.0], device=device)
            dist.broadcast(val_iou_tensor, 0)
            val_iou = val_iou_tensor.item()
            
            # Now update LR scheduler in all processes
            if not is_main_process:
                scheduler.step(val_iou)
        
        # Wait for all processes to reach this point
        if world_size > 1:
            dist.barrier()
    
    # Plot training progress on main process
    if is_main_process:
        logger.info(f'Training complete! Best IoU: {best_iou:.4f}')
    
    # Clean up
    if world_size > 1:
        dist.destroy_process_group()
    
    return best_iou if is_main_process else None


def train_epoch(net, device, train_loader, optimizer, world_size, epoch_idx):
    """Train for one epoch"""
    net.train()
    total_loss = 0
    batch_idx = 0
    
    for x, y, _, _, _ in train_loader:
        batch_idx += 1
        
        # Transfer to device
        xc = x.to(device)
        yc = [yy.to(device) for yy in y]
        
        # Forward pass
        lossd = net(xc, yc)
        loss = lossd['loss']
        
        # Compute gradients and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log batch statistics
        loss_item = loss.item()
        total_loss += loss_item
        
        # Only log from main process to avoid duplicate logs
        if dist.get_rank() == 0 and batch_idx % 100 == 0:
            # Check output values
            pos_pred = lossd['pred']['pos']
            logger.info(f"Batch {batch_idx}: Loss={loss_item:.4f}")
    
    avg_loss = total_loss / batch_idx
    
    if dist.get_rank() == 0:
        logger.info(f"Train Epoch {epoch_idx} Average Loss: {avg_loss:.4f}")
    
    return avg_loss


def validate_epoch(net, device, val_loader, batches_per_epoch, epoch_num=None):
    """Validate network on validation dataset"""
    if val_loader is None:
        return 0.0, 0.0
    
    net.eval()
    
    total_loss = 0
    correct = 0
    failed = 0
    
    with torch.no_grad():
        batch_idx = 0
        for x, y, didx, rot, zoom_factor in val_loader:
            batch_idx += 1
            if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                break
                
            # Transfer to device
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            
            # Forward pass
            lossd = net(xc, yc)
            loss = lossd['loss']
            total_loss += loss.item()
            
            # Post-process output
            q_out, ang_out, w_out = post_process_output(
                lossd['pred']['pos'], 
                lossd['pred']['cos'],
                lossd['pred']['sin'], 
                lossd['pred']['width']
            )
            
            # Get ground truth bounding boxes
            gt_bbs = val_loader.dataset.get_gtbb(didx[0].item(), rot[0].item(), zoom_factor[0].item())
            
            # Visualize first batch of each epoch
            vis_q_img = (batch_idx == 1)
            
            # Calculate IoU
            s = evaluation.calculate_iou_match(
                q_out, 
                ang_out,
                gt_bbs,
                no_grasps=1,
                grasp_width=w_out,
                epoch_num=None, 
                vis_q_img=vis_q_img
            )
            
            if s:
                correct += 1
            else:
                failed += 1
    
    # Calculate metrics
    avg_loss = total_loss / batch_idx if batch_idx > 0 else 0
    iou = correct / (correct + failed) if (correct + failed) > 0 else 0
    
    logger.info(f"Validation: Loss={avg_loss:.4f}, IoU={iou:.4f} ({correct}/{correct+failed})")
    return avg_loss, iou

if __name__ == '__main__':
    finetune_nbmod_from_cornell()