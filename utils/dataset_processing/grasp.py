import numpy as np

import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage.feature import peak_local_max

import xml.etree.ElementTree as ET

def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """
    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __len__(self):
        return len(self.grs)

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta/180.0*np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    @classmethod
    def load_from_xml_file(cls, fname):
        """
        Load grasp rectangles from an XML dataset file.
        :param fname: Path to the XML file.
        :return: GraspRectangles()
        """
        grs = []

        # Parse the XML file
        tree = ET.parse(fname)
        root = tree.getroot()

        grs = cls.xml_to_gr(root)
        grs = cls(grs)
        return grs
    
    @classmethod
    # ref: https://github.com/kmittle/Grasp-Detection-NBMOD/blob/main/data_preprocess/data_augmentation/label/original_img.py
    def nbmod_center_to_vertice(cls, x, y, w, h, angle):  # 将抓取参数转化为抓取框用以显示
        theta = angle
        vertice = np.zeros((4, 2))
        vertice[0] = (x - w / 2 * np.cos(theta) + h / 2 * np.sin(theta), y - w / 2 * np.sin(theta) - h / 2 * np.cos(theta))
        vertice[1] = (x + w / 2 * np.cos(theta) + h / 2 * np.sin(theta), y + w / 2 * np.sin(theta) - h / 2 * np.cos(theta))
        vertice[2] = (x + w / 2 * np.cos(theta) - h / 2 * np.sin(theta), y + w / 2 * np.sin(theta) + h / 2 * np.cos(theta))
        vertice[3] = (x - w / 2 * np.cos(theta) - h / 2 * np.sin(theta), y - w / 2 * np.sin(theta) + h / 2 * np.cos(theta))
        for i in range(0, 2):
            for j in range(0, 4):
                vertice[j][i] = round(vertice[j][i], 3)
        return vertice
    
    @classmethod
    # ref: https://github.com/kmittle/Grasp-Detection-NBMOD/blob/main/data_preprocess/data_augmentation/label/original_img.py
    def xml_to_gr(cls, root):
        grs = []
        for obj2 in root.iter('object'):
            # current = list()
            # class_num = class_names.index(name)
            xmlbox1 = obj2.find('robndbox')
            x = xmlbox1.find('cx').text
            y = xmlbox1.find('cy').text
            width = xmlbox1.find('w').text
            height = xmlbox1.find('h').text
            angle = xmlbox1.find('angle').text
            x = float(x)
            x = x * 1
            y = float(y)
            y = y * 1
            # print("prior: ", width, height)
            width = float(width)
            width = width
            height = float(height)
            height = height
            # print(post: ", width, height)
            angle = float(angle)
            if height > width:
                exchange = width
                width = height
                height = exchange
                angle = angle + 3.1415926/2
                if angle >= 3.1415926:
                    angle = angle - 3.1415926
            # convert to vertex
            gr_vertices = cls.nbmod_center_to_vertice(x, y, width, height, angle)
            grs.append(GraspRectangle(np.array(gr_vertices)))
        return grs


    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
           if pad_to > len(self.grs):
               a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int32)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int32)
    
    @property
    def num_grasps(self):
        """
        :return: Number of grasps
        """
        num = 0
        for itr in self.grs:
            num+= 1
        return num

class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int32)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, self.angle, self.length/3, self.width).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi/6):
        # Print angles for debugging
        angle_diff = abs((self.angle - gr.angle + np.pi/2) % np.pi - np.pi/2)
        # print(f"Self angle: {self.angle}, GR angle: {gr.angle}, Diff: {angle_diff}")
        
        if angle_diff > angle_threshold:
            # print("Rejected due to angle threshold")
            return 0
        # Debug rectangle coordinates
        # print("Rectangle 1 points:")
        # print(self.points)
        # print("Rectangle 2 points:")
        # print(gr.points)
       
        # Get coordinates and print their shape
        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])
        # print(f"Polygon 1 coords: {len(rr1)} points, Polygon 2 coords: {len(rr2)} points")
        
        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except Exception as e:
            print(f"Exception in IOU calculation: {e}")
            return 0
            
        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        
        union = np.sum(canvas > 0)
        intersection = np.sum(canvas == 2)
        
        # print(f"Intersection: {intersection}, Union: {union}, IOU: {intersection/union if union > 0 else 0}")
        return intersection/union if union > 0 else 0

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array([
                [np.cos(-angle), np.sin(-angle)],
                [-np.sin(-angle), np.cos(-angle)]
            ])
        # R = R.squeeze(-1)  # Removes the last dimension if it's size 1
        assert R.shape == (2, 2), f"Rotation matrix R has incorrect shape: {R}"
        c = np.array(center).reshape((1, 2))
        # print((self.points).shape)
        # print(R.shape)
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int32)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1/factor, 0],
                [0, 1/factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int32)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """
    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(np.array(
            [
             [y1 - self.width/2 * xo, x1 - self.width/2 * yo],
             [y2 - self.width/2 * xo, x2 - self.width/2 * yo],
             [y2 + self.width/2 * xo, x2 + self.width/2 * yo],
             [y1 + self.width/2 * xo, x1 + self.width/2 * yo],
             ]
        ).astype(np.float64))

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (self.center[1]*scale, self.center[0]*scale, -1*self.angle*180/np.pi, self.length*scale, self.width*scale)


def debug_quality_map(q_img, save_path=None):
    """Debug the quality map to see why no grasps are being detected"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show the quality map
    im1 = ax1.imshow(q_img, cmap='viridis')
    ax1.set_title("Quality Map (q_img)")
    plt.colorbar(im1, ax=ax1, label="Quality Value")
    
    # Show a histogram of values
    ax2.hist(q_img.flatten(), bins=50)
    ax2.set_title("Distribution of Quality Values")
    ax2.set_xlabel("Quality Value")
    ax2.set_ylabel("Frequency")
    ax2.axvline(x=0.2, color='r', linestyle='--', label="Threshold (0.2)")
    ax2.legend()
    
    # Add stats
    min_val = np.min(q_img)
    max_val = np.max(q_img)
    mean_val = np.mean(q_img)
    median_val = np.median(q_img)
    above_threshold = np.sum(q_img >= 0.2) / q_img.size * 100
    
    stats_text = (
        f"Min: {min_val:.4f}\n"
        f"Max: {max_val:.4f}\n"
        f"Mean: {mean_val:.4f}\n"
        f"Median: {median_val:.4f}\n"
        f"Values ≥ 0.2: {above_threshold:.2f}%"
    )
    
    plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10,
                bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Quality map debug saved to {save_path}")
    else:
        plt.show()
        
    # Try different thresholds to see if we get any peaks
    # thresholds = [0.0, 0.05, 0.1, 0.15, 0.2]
    thresholds = [0.2]
    for thresh in thresholds:
        peaks = peak_local_max(q_img, min_distance=20, threshold_abs=thresh, num_peaks=5)
        print(f"Threshold {thresh}: {len(peaks)} peaks found")
        if len(peaks) > 0:
            print(f"  Highest peak value: {q_img[tuple(peaks[0])]:.4f}")

def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1, epoch_num=None, vis_q_img=False):
    """
    Detect grasps in a GG-CNN output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    # if (epoch_num is not None) and (vis_q_img): # print the q_img after a certain epoch
    #     print(f"Epoch {epoch_num}: Q image shape: {q_img.shape}")
    #     debug_quality_map(q_img)
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)
    # print(len(local_max))
    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length/2

        grasps.append(g)
    return grasps
