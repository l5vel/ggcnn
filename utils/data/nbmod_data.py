import os
import glob
import torch

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image
from utils.data import get_dataset

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

class NBModDataset(GraspDatasetBase):
    """
    Dataset wrapper for the NBMOD dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: NBMOD Dataset directory.
        :param start: If splitting timg.shapehe dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(NBModDataset, self).__init__(**kwargs)
        # print("file_path: ",file_path)
        grasp_path = os.path.join(file_path, 'label')
        # print("grasp_path: ",grasp_path)
        graspf = glob.glob(os.path.join(grasp_path, '*r.xml'))
        graspf.sort()
        l = len(graspf)

        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        # Corrected the path replacement logic to match the files
        depthf = [f.replace('label', 'img').replace('r.xml', 'd.tiff') for f in graspf]
        rgbf = [f.replace('label', 'img').replace('r.xml', 'r.png') for f in graspf]
        
        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_xml_file(self.grasp_files[idx])
        c = self.output_size//2
        rot = rot.item() if torch.is_tensor(rot) else float(rot)
        zoom = zoom.item() if torch.is_tensor(zoom) else float(zoom)
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])


    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_xml_file(self.grasp_files[idx])

        center = gtbbs.center  # Center of the grasp
        crop_size_half = self.output_size // 2

        # Calculate top-left corner of the crop
        left = int(center[0] - crop_size_half)
        top = int(center[1] - crop_size_half)

        # Ensure the crop stays within the image boundaries
        left = max(0, min(left, 640 - self.output_size))
        top = max(0, min(top, 480 - self.output_size))

        return center, round(left), round(top)

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        cropped_img = depth_img.img[round(top):round(min(480, top + self.output_size)), round(left):round(min(640, left + self.output_size))]
        depth_image = image.DepthImage(cropped_img) # Wrap the numpy array back into a DepthImage object for normalization and resizing
        depth_image.normalise()
        depth_image.zoom(zoom)
        depth_image.resize((self.output_size, self.output_size))
        return depth_image.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=False):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rot = rot.item() if torch.is_tensor(rot) else float(rot)
        zoom = zoom.item() if torch.is_tensor(zoom) else float(zoom)
        rgb_img.rotate(rot, center)
        cropped_img = rgb_img.img[round(top):round(min(480, top + self.output_size)), round(left):round(min(640, left + self.output_size))]
        rgb_image = image.Image(cropped_img) # Wrap the numpy array back into an Image object for zooming and resizing
        rgb_image.zoom(zoom)
        rgb_image.resize((self.output_size, self.output_size))
        if normalise:
            rgb_image.normalise()
            # Check the shape after normalization and transpose to (H, W, C)
            if rgb_image.img.shape == (self.output_size, 3, self.output_size):
                return rgb_image.img.transpose(0, 2, 1)
            elif rgb_image.img.shape[0] == 3:
                return rgb_image.img.transpose(1, 2, 0)
            else:
                return rgb_image.img
        else:
            return rgb_image.img# Already likely (H, W, C) after loading and cropping

    # def get_jname(self, idx):
    #     return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])

    
    def add_rotated_rectangle(self, ax, center, width, height, angle):
        cx, cy = center
        lower_left = (cx - width / 2, cy - height / 2)
        rect = patches.Rectangle(lower_left, width, height, fill=False, edgecolor='red', linewidth=2)
        transform = transforms.Affine2D().rotate_deg_around(cx, cy, angle)
        rect.set_transform(transform + ax.transData)
        ax.add_patch(rect)

    def show_image_with_gtbbs(self, idx, rot=0, zoom=1.0):
        image = cv2.imread(self.rgb_files[idx])
        if image is None:
            print(f"Error: Could not load image: {self.rgb_files[idx]}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_bboxes = self.get_gtbb(idx, rot, zoom)

        fig, ax = plt.subplots()
        ax.imshow(image)
        for bbox in gt_bboxes:
            # Draw the rotated rectangle using center, width, height, and angle.
            self.add_rotated_rectangle(ax, bbox.center, bbox.width, bbox.length, bbox.angle)
        ax.axis('off')
        plt.show()

    
    def get_cropped_depth(self, idx):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        cropped_img = depth_img.img[top:min(480, top + self.output_size), left:min(640, left + self.output_size)]
        return cropped_img

    def get_cropped_rgb(self, idx):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        cropped_img = rgb_img.img[top:min(480, top + self.output_size), left:min(640, left + self.output_size)]
        return cropped_img
    
    def show_tiff_with_gtbbs(self, idx, rot=0, zoom=1.0):
        """
        Loads a TIFF image, obtains the grasp bounding boxes, overlays a rotated rectangle
        using grasp parameters, and displays the result.
        """
        try:
            img = Image.open(self.depth_files[idx])
            img.seek(0)  # Use the first frame if multi-page
        except Exception as e:
            print(f"Error: Could not load TIFF file {self.depth_files[idx]} ({e})")
            return

        # Determine colormap for grayscale images.
        cmap = 'gray' if img.mode in ['L', 'I'] else None
        gt_bboxes = self.get_gtbb(idx, rot, zoom)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap=cmap)
        for bbox in gt_bboxes:
            # Draw the rectangle overlay using bbox attributes.
            self.add_rotated_rectangle(ax, bbox.center, bbox.width, bbox.length, bbox.angle)
        ax.axis('off')
        plt.title("TIFF with Grasp Rectangle")
        plt.show()