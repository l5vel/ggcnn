import os
from utils.data import get_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import cv2
from PIL import Image
import numpy as np

# --- Configuration ---    
dataset_path = '/home/data/maa1446/nbmod/Simple-Single_Subset'

output_size = 300  # Ensure this matches your desired crop size

# --- Load the Dataset ---
Dataset = get_dataset('nbmod')
shuffle_seed = 42  # Or any fixed value, or add as a command-line argument

dataset = Dataset(dataset_path, start=0.0, end=1.0, ds_rotate=0,
                        random_rotate=True, random_zoom=True,
                        include_depth=1, include_rgb=0, shuffle_seed=shuffle_seed)
    

# --- Function to add rotated rectangle (already in your class) ---
def add_rotated_rectangle(ax, center, width, height, angle, edgecolor='red'):
    # print("center: ", center)
    cx, cy = center
    lower_left = (cx - width / 2, cy - height / 2)
    rect = patches.Rectangle(lower_left, width, height, fill=False, edgecolor=edgecolor, linewidth=2)
    transform = transforms.Affine2D().rotate_deg_around(cx, cy, angle)
    rect.set_transform(transform + ax.transData)
    ax.add_patch(rect)

# --- Verification Loop ---
num_samples_to_verify = [10195]  # Adjust as needed

def add_to_img_axes(axes,bbox,idx):
    for bb in bbox:
        # print(bbox.center)
        add_rotated_rectangle(axes[idx], bb.center, bb.width, bb.length, bb.angle)
    return axes[idx]

for i in num_samples_to_verify:
    try:
        # --- Load Original Data ---
        original_rgb = cv2.imread(dataset.rgb_files[i])
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
        original_depth_path = dataset.depth_files[i]
        original_depth_pil = Image.open(original_depth_path)
        original_depth = np.array(original_depth_pil)
        gt_bboxes_original = dataset.get_gtbb_org(i, rot=0, zoom=1.0)
        gt_bboxes_cropped = dataset.get_gtbb(i, rot=0, zoom=1.0)
        # print("shape of original gttbs: ", gt_bboxes_original.num_grasps)
        # --- Load Cropped Data (before resizing) ---
        cropped_rgb = dataset.get_rgb(i)
        cropped_depth = dataset.get_depth(i)
        center, left, top = dataset._get_crop_attrs(i)

        # --- Visualize ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sample {i}")

        # 1. Original RGB with GTBBs
        axes[0].imshow(original_rgb)
        add_to_img_axes(axes,gt_bboxes_original,0)
        axes[0].set_title("Original RGB with GTBBs")
        axes[0].axis('off')

        # 2. Cropped RGB
        if cropped_rgb.ndim == 2:
            # If the cropped RGB is grayscale
            axes[1].imshow(cv2.cvtColor(cropped_rgb, cv2.COLOR_GRAY2RGB))
            add_to_img_axes(axes,gt_bboxes_cropped,1)
            axes[1].set_title(f"Cropped RGB (Grayscale, Top:{top}, Left:{left}, Size:{cropped_rgb.shape})")
        elif cropped_rgb.shape[0] == 1:  # Check the first dimension for single channel
            axes[1].imshow(cv2.cvtColor(cropped_rgb[0], cv2.COLOR_GRAY2RGB)) # Access the single channel
            add_to_img_axes(axes,gt_bboxes_cropped,1)
            axes[1].set_title(f"Cropped RGB (Single Channel, Top:{top}, Left:{left}, Size:{(cropped_rgb.shape[1], cropped_rgb.shape[2])})")
        elif cropped_rgb.shape[0] == 3:  # If channels are the first dimension
            axes[1].imshow(cropped_rgb.transpose(1, 2, 0))
            add_to_img_axes(axes,gt_bboxes_cropped,1)
            axes[1].set_title(f"Cropped RGB (Transposed, Top:{top}, Left:{left}, Size:{(cropped_rgb.shape[1], cropped_rgb.shape[2])})")
        else:
            axes[1].imshow(cropped_rgb)
            add_to_img_axes(axes,gt_bboxes_cropped,1)
            axes[1].set_title(f"Cropped RGB (Top:{top}, Left:{left}, Size:{cropped_rgb.shape[:2]})")
        axes[1].axis('off')

        # 3. Cropped Depth (visualized with a colormap)
        axes[2].imshow(cropped_depth, cmap='viridis')  # You can try other colormaps
        add_to_img_axes(axes,gt_bboxes_cropped,2)
        axes[2].set_title(f"Cropped Depth (Top:{top}, Left:{left}, Size:{cropped_depth.shape[:2]})")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error loading file for sample {i}: {e}")
    except Exception as e:
        print(f"An error occurred during verification of sample {i}: {e}")