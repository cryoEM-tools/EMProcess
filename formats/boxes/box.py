import matplotlib.patches as patches
import numpy as np

# lightweight box loader
class box:
    def __init__(self, x=None, y=None, x_dim=None, y_dim=None):
        self.x = x
        self.y = y
        self.x_dim = x_dim
        self.y_dim = y_dim
        
    @property
    def loc(self):
        return (self.x, self.y)
    
    @property
    def arr(self):
        return [self.x, self.y, self.x_dim, self.y_dim]
    
    def patch_box(self, linewidth=1, edgecolor='r', facecolor='none', **kwargs):
        rect = patches.Rectangle(
            (self.loc), self.x_dim, self.y_dim, linewidth=linewidth,
            edgecolor=edgecolor, facecolor=facecolor, **kwargs)
        return rect


class boxes:
    def __init__(self, filename=None, data=None):
        if filename:
            self.load(filename)
        elif data is not None:
            self.data = np.array(data)
        else:
            self.data = None
        
    @property
    def array_data(self):
        return [d.arr for d in self.data]
    
    @property
    def n_boxes(self):
        return len(self.data)
    
    def __repr__(self):
        output = "boxes(n_boxes=%d)" % self.n_boxes
        return output
    
    def __getitem__(self, iis):
        return boxes(data=self.data[iis])
    
    def load(self, filename):
        box_file_contents = np.loadtxt(filename)
        self.data = []
        for box_data in box_file_contents:
            self.data.append(box(*box_data))
        self.data = np.array(self.data)
        return
    
    def save(self, output):
        np.savetxt(output, self.array_data, fmt='%0.1f\t%0.1f\t%0.1f\t%0.1f')

