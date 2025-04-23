import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
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

def finetune_nbmod_from_cornell():
    # Parameters
    input_channels = 1  # Change if using RGB
    batch_size = 32
    val_batches = 1350
    epochs = 100
    nbmod_dataset_path = '/home/data/maa1446/nbmod/Simple-Single_Subset/'  # Update with your path
    cornell_model_path = '/home/data/maa1446/ggcnn/output/models/ggcnn_weights_cornell'  # Update with your model path
    output_dir = '/home/data/maa1446/ggcnn/output/models/cornell_finetuned_models'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    learning_rate = 1e-5  # Lower learning rate for fine-tuning
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = os.path.join(output_dir, f'cornell_to_nbmod_{timestamp}')
    os.makedirs(save_folder, exist_ok=True)
    
    # Load Dataset
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
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    logger.info('Done loading datasets')
    
    # Initialize network
    logger.info('Loading Network...')
    ggcnn = get_network('ggcnn')
    net = ggcnn(input_channels=input_channels)
    
    # Load pre-trained Cornell model
    try:
        # Try loading the entire model
        logger.info(f"Loading pre-trained model from {cornell_model_path}")
        pretrained_model = torch.load(cornell_model_path, map_location=device)
        
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
    
    # Move to device
    net = net.to(device)
    
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
        logger.info(f'Beginning Epoch {epoch:02d}')
        
        # Train for one epoch
        train_loss = train_epoch(net, device, train_loader, optimizer)
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': best_iou
            }, os.path.join(save_folder, f'best_model_iou_{best_iou:.4f}.pth'))
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': val_iou
            }, os.path.join(save_folder, f'checkpoint_epoch_{epoch}.pth'))
        
        # Debug output quality maps periodically
        if epoch % 10 == 0:
            debug_quality_maps(net, device, val_dataset, save_folder, epoch)
    
    # Plot training progress
    plot_training_progress(train_losses, val_losses, val_ious, save_folder)
    
    logger.info(f'Training complete! Best IoU: {best_iou:.4f}')
    return best_iou


def train_epoch(net, device, train_loader, optimizer):
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
        
        if batch_idx % 10 == 0:
            # Check output values
            pos_pred = lossd['pred']['pos']
            logger.info(f"Batch {batch_idx}: Loss={loss_item:.4f}, Pos max={pos_pred.max().item():.4f}, mean={pos_pred.mean().item():.4f}")
    
    avg_loss = total_loss / batch_idx
    logger.info(f"Train Epoch Average Loss: {avg_loss:.4f}")
    return avg_loss


def validate_epoch(net, device, val_loader, batches_per_epoch, epoch_num=None):
    """Validate network on validation dataset"""
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
    avg_loss = total_loss / batch_idx
    iou = correct / (correct + failed) if (correct + failed) > 0 else 0
    
    logger.info(f"Validation: Loss={avg_loss:.4f}, IoU={iou:.4f} ({correct}/{correct+failed})")
    return avg_loss, iou


def debug_quality_maps(net, device, dataset, save_folder, epoch):
    """Debug function to visualize quality maps"""
    os.makedirs(os.path.join(save_folder, 'debug'), exist_ok=True)
    
    net.eval()
    with torch.no_grad():
        for i in range(5):  # Debug first 5 samples
            # Get sample
            x, y, idx, rot, zoom = dataset[i]
            x = x.unsqueeze(0).to(device)
            
            # Forward pass
            pos_pred, cos_pred, sin_pred, width_pred = net(x)
            
            # Convert to numpy
            q_img = pos_pred.cpu().squeeze().numpy()
            
            # Get ground truth
            gt_bbs = dataset.get_gtbb(idx, rot, zoom)
            pos_gt, ang_gt, width_gt = gt_bbs.draw((dataset.output_size, dataset.output_size))
            
            # Apply different thresholds
            # thresholds = [0.0, 0.1, 0.2, 0.3]
            thresholds = [0.2]

            fig, axes = plt.subplots(2, len(thresholds)+1, figsize=(5*(len(thresholds)+1), 10))
            
            # Input and ground truth
            axes[0, 0].imshow(x.cpu().squeeze(), cmap='gray')
            axes[0, 0].set_title("Input")
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(pos_gt, cmap='viridis')
            axes[1, 0].set_title("Ground Truth")
            axes[1, 0].axis('off')
            
            # Quality map at different thresholds
            for i, thresh in enumerate(thresholds):
                from skimage.feature import peak_local_max
                local_max = peak_local_max(q_img, min_distance=20, 
                                          threshold_abs=thresh, 
                                          num_peaks=5)
                
                # Show quality map
                im = axes[0, i+1].imshow(q_img, cmap='viridis')
                axes[0, i+1].set_title(f"Quality Map\nMax: {q_img.max():.4f}, Mean: {q_img.mean():.4f}")
                fig.colorbar(im, ax=axes[0, i+1])
                
                # Show detected grasp points
                axes[1, i+1].imshow(q_img, cmap='viridis')
                axes[1, i+1].set_title(f"Threshold: {thresh}, Peaks: {len(local_max)}")
                
                # Mark peak points
                for point in local_max:
                    y, x = point
                    axes[1, i+1].plot(x, y, 'r+', markersize=10)
                
                axes[1, i+1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, 'debug', f'epoch_{epoch}_sample_{i}.png'))
            plt.close()


def plot_training_progress(train_losses, val_losses, val_ious, save_folder):
    """Plot and save training progress"""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot IoU
    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'training_progress.png'))
    plt.close()

if __name__ == '__main__':
    finetune_nbmod_from_cornell()