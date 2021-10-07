import copy
import matplotlib.pylab as plt
import numpy as np
from collections import OrderedDict

# common star file labels and their type
LABELS = {
    'OpticsGroupName': str,
    'OpticsGroup': int,
    'MtfFileName': str, 
    'MicrographOriginalPixelSize': float,
    'ImagePixelSize': float,
    'ImageSize': int,
    'ImageDimensionality' : int,
    'Voltage': float,
    'DefocusU': float,
    'DefocusV': float,
    'DefocusAngle': float,
    'SphericalAberration': float,
    'DetectorPixelSize': float,
    'CtfFigureOfMerit': float,
    'Magnification': float,
    'AmplitudeContrast': float,
    'ImageName': str,
    'OriginalName': str,
    'CtfImage': str,
    'CoordinateX': float,
    'CoordinateY': float,
    'CoordinateZ': float,
    'NormCorrection': float,
    'MicrographName': str,
    'GroupName': str,
    'GroupNumber': str,
    'OriginX': float,
    'OriginY': float,
    'OriginXAngst': float,
    'OriginYAngst': float,
    'AngleRot': float,
    'AngleTilt': float,
    'AnglePsi': float,
    'ClassNumber': int,
    'LogLikeliContribution': float,
    'RandomSubset': int,
    'ParticleName': str,
    'OriginalParticleName': str,
    'NrOfSignificantSamples': int,
    'NrOfFrames': int,
    'MaxValueProbDistribution': float
}

class Label():
    """Label class
    
    Attributes
    ----------
    data : nd.array, shape=(n_items, )
        The list of data associated with the label
    type : str
        The data type for the label, i.e. string, int, or float
    """
    def __init__(self, labelName, data=None):
        """
        Inputs
        ----------
        labelName : str
            The name of the label
        data : array-like, shape=(n_items,), default=None,
            Optionally populate label with a set of data upon initilization
        """
        self.name = labelName
        # Get the type from the LABELS dict, assume str by default
        self.type = LABELS.get(labelName, str)
        self.clear()
        if data is not None:
            self._data = list(data)
    
    @property
    def data(self):
        return np.array(self._data, dtype=self.type)
    
    @property
    def _n_items(self):
        return len(self._data)
    
    def clear(self):
        self._data = []

    def __repr__(self):
        output = "Label(labelName=%s, n_items=%d)" % (self.name, self._n_items)
        return output 
        
    def __str__(self):
        return self.name

    def __cmp__(self, other):
        return self.name == str(other)
    
    def __getitem__(self, iis):
        new_data = self.data[iis].reshape(-1)
        new_label = Label(self.name, data=new_data)
        return new_label
    
    def __setitem__(self, iis, value):
        new_data = self.data
        new_data[iis] = value
        self._data = list(new_data)
        return
    
    def __len__(self):
        return len(self._data)
    
    def append(self, other, inplace=True):
        self._data.append(self.type(other))


class MetaData():
    """MetaData class containing an organized set of labels and their data.
    Attributes are populated from found Labels.

    Attributes
    ----------
    labels : dict, shape=(n_labels),
        A dictionary containing each label class.
    label_order : dict, shape=(n_labels)
        A dictionary to keep track of label ordering.
    label_names : list, shape=(n_labels)
        The name of each label.
    """
    def __init__(self, group_name, labels=None, label_order=None):
        self.name = group_name
        self._clear()
        if labels:
            self._labels = labels
            self.__dict__.update(self._labels)
        if label_order:
            self._label_order = label_order
        self._assert_consistent()
    
    def __getitem__(self, iis):
        new_labels = OrderedDict()
        for key,value in self._labels.items():
            new_labels[key] = value[iis]
        return MetaData(
            group_name=self.name, labels=new_labels,
            label_order=self._label_order)
    
    def __setitem__(self, iis, other):
        assert np.all(np.sort(self.label_names) == np.sort(other.label_names))
        new_labels = OrderedDict()
        for key,value in self._labels.items():
            new_label = value
            new_label[iis] = other._labels[key].data
            new_labels[key] = new_label
        return
            
    
    def _assert_consistent(self):
        try:
            assert self._consistent_num_items
        except:
            logging.warning('inconsistent number of items between labels!')
        return
    
    def __repr__(self):
#         self._assert_consistent()
        output = "%s(n_labels=%d, n_items=%d)" % \
            (
                self.name, self._n_labels,
                np.average(self._n_label_items))
        return output
    
    @property
    def _n_items(self):
        return np.average(self._n_label_items)
    
    @property
    def label_names(self):
        return [name for name in self.labels.keys()]
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def _n_labels(self):
        return len(self._labels)
    
    @property
    def _n_label_items(self):
        return [l._n_items for l in self._labels.values()]
    
    @property
    def _consistent_num_items(self):
        consistent = False
        if (np.unique(self._n_label_items).shape[0] == 1) or \
                (np.unique(self._n_label_items).shape[0] == 0):
            consistent = True
        return consistent
    
    @property
    def _sorted_label_indices(self):
        return np.argsort(list(self._label_order.keys()))
    
    def _clear(self):
        self._labels = OrderedDict()
        self._label_order = OrderedDict()

    def _add_label(self, label_name, order_num):
        self._labels[label_name] = Label(label_name)
        self._label_order[order_num] = label_name
        self.__dict__.update(self._labels)

    def copy(self):
        return copy.deepcopy(self)
    
    def _header_lines(self):
        header = [
            "",
            "# verion 30001",
            ""]
        header.append(self.name)
        header.append("")
        header.append("loop_")

        keys = []
        values = []
        for k,v in self._label_order.items():
            keys.append(k)
            values.append(v)
        
        for n in self._sorted_label_indices:
            label_name = "_rln" + self.labels[values[n]].name
            order = keys[n]
            new_line = label_name + " #%d" % (order + 1)
            header.append(new_line)
        return header
    
    def _get_new_line(self, line_number, label_names):
        new_line = []
        for n in self._sorted_label_indices:
            label = self.labels[label_names[n]]
            value = label._data[line_number]
            if label.type is str:
                new_line.append(value)
            elif label.type is int:
                new_line.append('{0: >12}'.format(value))
            elif label.type is float:
                if np.abs(value) >= 100000:
                    new_line.append('{0: >#012.6e}'.format(value))
                else:
                    new_line.append('{0: >#012.6f}'.format(value))
            else:
                print("Error! not recognized label format for %s" % label.name)
        return " ".join(new_line)
    
    def _data_lines(self):
        n_items = int(self._n_items)
        label_names = self.label_names
        new_lines = []
        for n in np.arange(n_items):
            new_lines.append(self._get_new_line(n, label_names))
        return new_lines


class StarFile(object):
    """Class to parse Relion star file. Contains all the MetaData found
    within a file. Attributes are dynamically populated with found MetaData.

    Attributes
    ----------
    
    """
    def __init__(self, input_star=None):
        if input_star:
            self.read(input_star)
        else:
            self.clear()
            
    def __repr__(self):
        output = "StarFile(n_groups=%d)" % self._n_groups
        return output

    @property
    def _n_groups(self):
        return len(self._data)
    
    @property
    def group_names(self):
        return [name for name in self._data.keys()]
    
    @property
    def _sorted_group_indices(self):
        return np.argsort(list(self._data_order.keys()))
        
    def clear(self):
        self._data = OrderedDict()
        self._data_order = OrderedDict()

    def _add_metadata(self, group_name, order):
        self._data[group_name] = MetaData(group_name)
        self._data_order[order] = group_name

    def read(self, input_star):
        self.clear()

        with open(input_star, "r") as f:
            n_metadata = 0
            n_label = 0
            found_label = False
            found_group = False
            t = 0
            for l in f:

                # strip values
                values = l.strip().split()

                # if no values *after* extracting metadata, reset counter
                if not values and found_group and found_label:
                    found_group = False
                    found_label = False
                    n_metadata += 1
                    n_label = 0
                    continue
                elif not values and found_group and not found_label:
                    continue
                elif not values and not found_group and not found_label:
                    continue
                elif values[0].startswith("#"):
                    continue
                elif values[0].startswith("loop"):
                    assert found_group and not found_label
                    continue

                if values[0].startswith("_rln"):
                    assert found_group
                    label_name = values[0].split("_rln")[-1]
                    self._data[group_name]._add_label(label_name, order_num=n_label)
                    n_label += 1
                    found_label = True
                elif found_label:
                    for label, value in zip(self._data[group_name]._labels.values(), values):
                        label.append(value)
                else:
                    assert not found_group
                    found_group = True
                    group_name = values[0]
                    self._add_metadata(group_name, n_metadata)
        self.__dict__.update(self._data)
        
    def write(self, filename):
        file_output = []
        for n in self._sorted_group_indices:
            group = self._data[particles.group_names[n]]
            file_output.append(group._header_lines())
            file_output.append(group._data_lines())
            file_output.append("")
        file_output = np.hstack(file_output)
        with open(filename, "w") as f:
            for line in file_output:
                f.write(line+"\n")

    def copy(self):
        return copy.deepcopy(self)


class Particles(StarFile):
    """Class to parse Relion star file. Contains all the MetaData found
    within a file. Attributes are dynamically populated with found MetaData.

    Attributes
    ----------
    
    """
    def __init__(self, filename=None, data=None, data_order=None):
        if filename:
            self.read(filename)
        elif data is not None:
            self.clear()
            self._data = data
            if data_order is not None:
                self._data_order = data_order
        
        try:
            assert hasattr(self, 'data_particles')
        except:
            logging.warning(
                'no data_particles group found! ' +\
                'Is you sure this is a particles file?')
        return
    
    @property
    def _n_particles(self):
        return self.data_particles._n_items
    
    @property
    def _unique_classes(self):
        unique_classes = None
        if hasattr(self.data_particles, 'ClassNumber'):
            unique_classes = np.unique(self.data_particles.ClassNumber.data)
        return unique_classes
    
    @property
    def _n_classes(self):
        n_classes = 0
        if self._unique_classes is not None:
            n_classes = len(self._unique_classes)
        return n_classes
    
    @property
    def _class_counts(self):
        class_counts = []
        if self._unique_classes is not None:
            class_counts = np.bincount(self.data_particles.ClassNumber.data)
        return class_counts
    
    def __getitem__(self, iis):
        new_particles = self.copy()
        old_data = new_particles._data.pop('data_particles')
        new_particles._data['data_particles'] = old_data[iis]
        new_particles.__dict__.update(new_particles._data)
        return new_particles
    
    def __setitem__(self, iis, other):
        old_data_particles = self._data.pop('data_particles')
        old_data_particles[iis] = other._data['data_particles']
        self._data['data_particles'] = old_data_particles
        self.__dict__.update(self._data)
    
    def __repr__(self):
        output = 'Particles(n_particles=%d, n_classes=%d)' % \
            (self._n_particles, self._n_classes)
        return output
    
    def copy(self):
        return copy.deepcopy(self)

    def concatenate(particle_list):
        new_particles = particle_list[0].copy()
        for key,item in particle_list[0].data_particles._labels.items():
            for n in np.arange(1, len(particle_list)):
                new_particles.data_particles._labels[key]._data.extend(
                    particle_list[n].data_particles._labels[key]._data)
        return new_particles
