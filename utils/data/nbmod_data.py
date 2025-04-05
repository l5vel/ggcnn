import os
import glob
import torch

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image


class NBModDataset(GraspDatasetBase):
    """
    Dataset wrapper for the NBMOD dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: NBMOD Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(NBModDataset, self).__init__(**kwargs)
        grasp_path = os.path.join(file_path, 'label')
        print(grasp_path)
        graspf = glob.glob(os.path.join(grasp_path, '*r.xml'))
        graspf.sort()
        l = len(graspf)

        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        # Corrected the path replacement logic to match the files
        depthf = [f.replace('label', 'img').replace('r.xml', 'd.tiff') for f in graspf]
        rgbf = [f.replace('label', 'img').replace('r.xml', 'r.png') for f in depthf]
        
        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_xml_file(fname = self.grasp_files[idx])
        c = self.output_size//2
        rot = rot.item() if torch.is_tensor(rot) else float(rot)
        zoom = zoom.item() if torch.is_tensor(zoom) else float(zoom)
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        rot = rot.item() if torch.is_tensor(rot) else float(rot)
        zoom = zoom.item() if torch.is_tensor(zoom) else float(zoom)
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rot = rot.item() if torch.is_tensor(rot) else float(rot)
        zoom = zoom.item() if torch.is_tensor(zoom) else float(zoom)
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])