import numpy as np
import matplotlib.pyplot as plt
from .grasp import GraspRectangles, detect_grasps

def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None, ground_truth_bbs=None):
    """
    Plot the output of a GG-CNN with optional ground truth visualization
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :param ground_truth_bbs: (optional) Ground truth bounding boxes
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    
    fig = plt.figure(figsize=(10, 10))
    
    # Plot RGB image with grasps
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax, color='r')  # Plot predicted grasps in red
    
    # If ground truth is provided, plot it in green
    if ground_truth_bbs is not None:
        if not isinstance(ground_truth_bbs, GraspRectangles):
            gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
        else:
            gt_bbs = ground_truth_bbs
            
        for g in gt_bbs:
            g.plot(ax, color='g')  # Plot ground truth in green
            
    ax.set_title('RGB')
    ax.axis('off')
    
    # Plot depth image with grasps
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax, color='r')  # Plot predicted grasps in red
        
    # If ground truth is provided, plot it in green
    if ground_truth_bbs is not None:
        if not isinstance(ground_truth_bbs, GraspRectangles):
            gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
        else:
            gt_bbs = ground_truth_bbs
            
        for g in gt_bbs:
            g.plot(ax, color='g')  # Plot ground truth in green
            
    ax.set_title('Depth')
    ax.axis('off')
    
    # Plot Q output
    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)
    
    # Plot angle output
    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    
    # Add legend
    if ground_truth_bbs is not None:
        # Add rectangle patches for the legend
        from matplotlib.patches import Patch
        red_patch = Patch(color='red', label='Predicted Grasps')
        green_patch = Patch(color='green', label='Ground Truth')
        fig.legend(handles=[red_patch, green_patch], loc='lower center', ncol=2)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display IoU match if ground truth is available
    if ground_truth_bbs is not None:
        success = calculate_iou_match(grasp_q_img, grasp_angle_img, ground_truth_bbs, no_grasps, grasp_width_img)
        print(f"Grasp prediction success: {success}")
        return success
    
    return None

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """
    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    
    # Modified to return more detailed information
    matched_grasps = []
    iou_values = []
    
    for g in gs:
        curr_iou = g.max_iou(gt_bbs)
        iou_values.append(curr_iou)
        if curr_iou > 0.25:
            matched_grasps.append(g)
    
    # Print detailed metrics
    # print(f"IoU values: {iou_values}")
    # print(f"Matched grasps: {len(matched_grasps)}/{len(gs)}")
    
    return len(matched_grasps) > 0