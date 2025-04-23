import datetime
import os
import sys
import argparse
import logging

import cv2

import torch
import torch.utils.data
import torch.optim as optim

from torchsummary import summary

import tensorboardX

from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output

logging.basicConfig(level=logging.INFO)

def list_shape(list_):
    if not isinstance(list_, list):
        return ()  # Not a list, return empty tuple
    elif not list_:
        return (0,)  # Empty list, shape is (0,)
    elif not isinstance(list_[0], list):
        return (len(list_),)  # 1D list, shape is (length,)
    else:
        # Nested list, get shape of first sublist and check if consistent
        sublist_shape = list_shape(list_[0])
        for sublist in list_:
            if list_shape(sublist) != sublist_shape:
                raise ValueError("Inconsistent list shape")
        return (len(list_),) + sublist_shape
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default = 'nbmod', help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default = None, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.8, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=7, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=15, help='Validation Batches')

    # Distributed Training
    parser.add_argument('--world-size', type=int, default=6, help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank of the current process')
    parser.add_argument('--init-method', type=str, default='env://', help='Method to initialize distributed training')

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch, epoch_num=None):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {}
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net(xc, yc)

                loss = lossd['loss'].mean() if isinstance(lossd['loss'], torch.Tensor) else lossd['loss']

                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                gt_bbs = val_data.dataset.get_gtbb(didx, rot, zoom_factor)
                # logging.info(f'Number of ground truth bounding boxes: {gt_bbs.num_grasps}')
                if batch_idx == 1: # only visualize the q_img for the first validation image
                    vis_q_img = True
                else:
                    vis_q_img = False
                s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   gt_bbs,
                                                   no_grasps=1,
                                                   grasp_width=w_out,epoch_num=epoch_num, vis_q_img=vis_q_img)

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, rank=0, world_size=1, vis=True):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param rank: Rank of the current process
    :param world_size: Total number of processes
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {}
    }

    net.train()
    batch_idx = 0
    while batch_idx < batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net(xc, yc)

            loss = lossd['loss']
            if isinstance(loss, torch.Tensor):
                if world_size > 1:
                    torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    loss /= world_size
                loss_item = loss.item()
            else:
                loss_item = loss

            logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss_item:0.4f}')

            results['loss'] += loss_item
            for ln, l_tensor in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                if isinstance(l_tensor, torch.Tensor):
                    if world_size > 1:
                        torch.distributed.reduce(l_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
                        l_tensor /= world_size
                    results['losses'][ln] += l_tensor.item()
                else:
                    results['losses'][ln] += l_tensor

            optimizer.zero_grad()
            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()

            # Display the images (only on rank 0)
            if vis and rank == 0:
                gt_img = x[idx,].cpu().numpy().squeeze()
                # gt_bbs = 
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].cpu().numpy().squeeze()] + [yi[idx,].cpu().numpy().squeeze() for yi in y] + [
                        x[idx,].cpu().numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()
    logging.info(f"--- STARTING run() with rank: {args.rank}, local_rank: {args.local_rank}, world_size: {args.world_size} ---")
    
    # Print CUDA device information
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    # When using torch.distributed.launch, these environment variables are set
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"Using environment variables: WORLD_SIZE={args.world_size}, RANK={args.rank}, LOCAL_RANK={args.local_rank}")
    
    # Setup device based on local_rank for distributed training
    if args.world_size > 1:
        logging.info(f"Setting up distributed training with {args.world_size} processes")
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        try:
            # Initialize the process group
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=args.init_method,
                world_size=args.world_size,
                rank=args.rank
            )
            logging.info(f"Successfully initialized distributed process group")
            logging.info(f"Process rank: {args.rank}, local rank: {args.local_rank}, world size: {args.world_size}")
            logging.info(f"Using GPU: {torch.cuda.get_device_name(args.local_rank)}")
        except Exception as e:
            logging.error(f"Failed to initialize distributed training: {e}")
            logging.info("Falling back to single GPU training.")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            args.world_size = 1
            args.rank = 0
    else:
        logging.info("Setting up single GPU training")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using single GPU: {device}")
        args.world_size = 1
        args.rank = 0

    # Vis window
    if args.vis and args.rank == 0:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    # Set-up output directories
    if args.rank == 0:
        dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
        net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
        save_folder = os.path.join(args.outdir, net_desc)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info(f'Loading {args.dataset.title()} Dataset...')
    Dataset = get_dataset(args.dataset)
    if args.dataset_path is None:
        dataset_path = '/home/data/maa1446/nbmod/a_bunch_of_bananas'
    else:
        dataset_path = args.dataset_path

    # Use the same random seed for both dataset instances to ensure consistent shuffling
    shuffle_seed = 42  # Or any fixed value, or add as a command-line argument

    train_dataset = Dataset(dataset_path, start=0.0, end=args.split, shuffle_seed=shuffle_seed,
                        random_rotate=True, random_zoom=True,
                        include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_dataset = Dataset(dataset_path, start=args.split, end=1.0, shuffle_seed=shuffle_seed,
                        random_rotate=True, random_zoom=True,
                        include_depth=args.use_depth, include_rgb=args.use_rgb)

    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.world_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )
    for abc in train_data:
        print("abc: ",list_shape(abc))
        for itr in abc:
            print("shapes: ", list_shape(itr))
        break
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        pin_memory=True
    )
    logging.info('Done loading dataset.')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    ggcnn = get_network(args.network)
    net = ggcnn(input_channels=input_channels)
    net = net.to(device)

    if args.world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = optim.Adam(net.parameters())
    logging.info('Done loading network.')

    # Print model architecture.
    if args.rank == 0:
        summary(net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net,
                (input_channels, 300, 300))
        with open(os.path.join(save_folder, 'arch.txt'), 'w') as f:
            sys.stdout = f
            summary(net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net,
                    (input_channels, 300, 300))
            sys.stdout = sys.__stdout__

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info(f'Beginning Epoch {epoch:02d}')
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch,
                              rank=args.rank, world_size=args.world_size, vis=args.vis)

        # Log training losses to tensorboard
        if args.rank == 0:
            tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
            for n, l in train_results['losses'].items():
                tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        if args.rank == 0:
            logging.info('Validating...')
            if epoch % 100 == 0:
                epoch_num = epoch
            else:
                epoch_num = None
            test_results = validate(net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net,
                                    device, val_data, args.val_batches, epoch_num=epoch_num)
            iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
            logging.info(f'Validation IOU: {iou:.4f} ({test_results["correct"]}/{test_results["correct"] + test_results["failed"]})')

            # Log validation results to tensorboard
            tb.add_scalar('loss/IOU', iou, epoch)
            tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
            for n, l in test_results['losses'].items():
                tb.add_scalar('val_loss/' + n, l, epoch)

            # Save best performing network
            state_dict = net.module.state_dict() if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net.state_dict()
            cur_net = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
            if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
                torch.save(cur_net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
                torch.save(state_dict, os.path.join(save_folder, f'epoch_{epoch:02d}_iou_{iou:.2f}_statedict.pt'))
                best_iou = iou

        if args.world_size > 1:
            torch.distributed.barrier()

    if args.rank == 0:
        tb.close()
        logging.info('Training complete. Best validation IOU: {:.4f}'.format(best_iou))

logging.info("--- BEFORE if __name__ == '__main__': ---")
if __name__ == '__main__':
    logging.info("--- INSIDE if __name__ == '__main__': ---")
    run()
    logging.info("--- AFTER run() call ---")
