import copy
import matplotlib.pylab as plt
import numpy as np
import sys
from collections import OrderedDict

# common star file labels and their type
LABELS = {
    'OpticsGroupName': str,
    'OpticsGroup': int,
    'MtfFileName': str, 
    'MicrographOriginalPixelSize': float,
    'Voltage': float,
    'SphericalAberration': float,
    'AmplitudeContrast': float,
    'ImagePixelSize': float,
    'ImageSize': int,
    'ImageDimensionality' : int,
    'BeamTiltX' : float,
    'BeamTiltY' : float,
    'CtfDataAreCtfPremultiplied' : int,
    'CoordinateX': float,
    'CoordinateY': float,
    'CoordinateZ': float,
    'ImageName': str,
    'MicrographName': str,
    'CtfMaxResolution' : float,
    'DefocusU': float,
    'DefocusV': float,
    'DefocusAngle': float,
    'CtfBfactor' : float, 
    'CtfScalefactor' : float, 
    'PhaseShift' : float,
    'GroupNumber': str,
    'DetectorPixelSize': float,
    'CtfFigureOfMerit': float,
    'Magnification': float,
    'OriginalName': str,
    'CtfImage': str,
    'NormCorrection': float,
    'GroupName': str,
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
    'MaxValueProbDistribution': float,
    'AutopickFigureOfMerit': float
}

# preferred ordering of some data_optics labels
DATA_OPTICS_LABEL_ORDER = [
    'OpticsGroupName',
    'OpticsGroup',
    'MtfFileName',
    'MicrographOriginalPixelSize',
    'Voltage',
    'SphericalAberration',
    'AmplitudeContrast',
    'ImagePixelSize',
    'ImageSize',
    'ImageDimensionality',
    'BeamTiltX',
    'BeamTiltY',
    'CtfDataAreCtfPremultiplied'
]

# preferred ordering of some data_particles labels
DATA_PARTICLES_LABEL_ORDER = [
    'CoordinateX',
    'CoordinateY',
    'CoordinateZ',
    'ImageName',
    'MicrographName',
    'OpticsGroup',
    'CtfMaxResolution',
    'CtfFigureOfMerit',
    'DefocusU',
    'DefocusV',
    'DefocusAngle',
    'CtfBfactor',
    'CtfScalefactor', 
    'PhaseShift',
    'GroupNumber',
    'AngleRot',
    'AngleTilt',
    'AnglePsi',
    'OriginX',
    'OriginY',
    'OriginXAngst',
    'OriginYAngst',
    'ClassNumber',
    'NormCorrection',
    'LogLikeliContribution',
    'MaxValueProbDistribution',
    'NrOfSignificantSamples',
    'RandomSubset'
]


def join_stars(star_list, add_filename=False):
    """Joins star files into a single star file.

    Inputs
    ----------
    star_list : array-like, (str or StarFile), shape=(n_filenames,),
        The STAR objects or filenames to join.
    add_filename : bool, default=False,
        Optionally add the file that the data came from as a new entry
        in each group.

    Returns
    ----------
    new_star : StarFile, obj,
        A StarFile object containing the concatenated entries for each
        group in the star_list.    
    """
    stars = []
    for s in star_list:
        if (type(s) is str) or (type(s) is np.str_):
            stars.append(StarFile(s))
        elif type(s) is EMProcess.formats.star.starfile.StarFile:
            stars.append(s)
        else:
            raise

    # ensure the same groups and labels
    group_names = stars[0].group_names
    labels_per_group = [stars[0].__getattribute__(g).label_names for g in group_names]
    for s in stars:
        for n,g in enumerate(group_names):
            assert g in s.group_names
            assert np.all(np.sort(labels_per_group[n]) == np.sort(s.__getattribute__(g).label_names))

    # extract star data and concatenate
    stars_data = [
        [
            [
                s.__getattribute__(g).labels[l]._data
                for l in labels_per_group[n]
            for g in group_names]]
        for s in stars]
    stars_data = np.concatenate(stars_data, axis=2)

    new_star = StarFile(groups=group_names, label_names=labels_per_group, data=stars_data)

    # optionally add the filename data came from as a new parameter to each group
    if add_filename:
        for s in star_list:
            assert (type(s) is str) or (type(s) is np.str_)
        lengths = [[s.__getattribute__(g)._n_items for s in stars] for g in group_names]
        filenames = [
            np.concatenate([[s]*l for s,l in zip(star_list, lengths_inner)])
            for lengths_inner in lengths]
        for n,g in enumerate(group_names):
            new_star.__getattribute__(g)._add_label('Filename', data=filenames[n])
        
    return new_star


def append_particles(particles0, particles1):
    """Appends non-redundant labels from particles 1 to particles 0.

    Inputs
    ----------
    particles0 : Particles, shape=(n_particles,),
        Particle object to append new labels to.
    particles1 : Particles, shape=(n_particles,),
        Particle object contining unique labels to append to particles0.
    """

    # obtain label names
    label_names0 = particles0.data_particles.label_names
    label_names1 = particles1.data_particles.label_names

    # determine unique labels
    labels_to_add = np.setdiff1d(label_names1, label_names0)

    # append to new particle set
    new_particles = particles0.copy()
    for l in labels_to_add:
        new_particles.data_particles._add_label(
            l, data=particles1.data_particles._labels[l].data)

    return new_particles


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
    def __init__(self, group_name, labels=None, label_order=None, label_names=None, data=None):
        self.name = group_name
        self._clear()
        if labels:
            self._labels = labels
            self.__dict__.update(self._labels)
        elif (label_names is not None) and (data is not None):
            for n,l in enumerate(label_names):
                self._labels[l] = Label(l,data=data[n])
                self._label_order[n] = l
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
        return np.max(self._n_label_items)
    
    @property
    def label_names(self):
        return [name for name in self._label_order.values()]
    
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
    def _last_label_num(self):
        if len(self._label_order) == 0:
            last_label_num = -1
        else:
            last_label_num = max([k for k in self._label_order.keys()])
        return last_label_num
    
    @property
    def _sorted_label_indices(self):
        return np.argsort(list(self._label_order.keys()))
    
    def _clear(self):
        self._labels = OrderedDict()
        self._label_order = OrderedDict()

    def _add_label(self, label_name, order_num=None, data=None):
        if order_num is None:
            order_num = self._last_label_num + 1
        self._labels[label_name] = Label(label_name)
        self._label_order[order_num] = label_name
        if data is not None:
            self._labels[label_name]._data = list(data)
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
        header.append("loop_ ")

        keys = []
        values = []
        for k,v in self._label_order.items():
            keys.append(k)
            values.append(v)
        
        for n in self._sorted_label_indices:
            label_name = "_rln" + self.labels[values[n]].name
            order = keys[n]
            new_line = label_name + " #%d " % (order + 1)
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
                value = float(value)
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
    def __init__(
            self, input_star=None, groups=None, labels=None,
            label_names=None, data=None):
        if input_star:
            self.read(input_star)
        else:
            self.clear()

        # if data is input, fill star file
        if (groups is not None) and (labels is not None):
            for n,group in enumerate(groups):
                self._data[group] = MetaData(groups[n], labels=labels[n])
            self.__dict__.update(self._data)
        elif (groups is not None) and (label_names is not None) and (data is not None): 
            for n,group in enumerate(groups):
                self._data[group] = MetaData(groups[n], label_names=label_names[n], data=data[n])
                self._data_order[n] = groups[n]
            self.__dict__.update(self._data)

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
            group = self._data[self.group_names[n]]
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
        
#        try:
#            assert hasattr(self, 'data_particles')
#        except:
#            logging.warning(
#                'no data_particles group found! ' +\
#                'Is you sure this is a particles file?')
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
        new_particles = Particles()
        new_particles.clear()
        for n, group_name in enumerate(self.group_names):
            new_particles._add_metadata(group_name, n)
            if group_name == 'data_particles':
                new_particles._data[group_name] = self._data[group_name][iis]
            else:
                new_particles._data[group_name] = self._data[group_name]
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
    
    def relion_label_ordering(self):

        # determine the labels that don't fit the mold (add them at the end)
        unordered_data_particles_labels = np.setdiff1d(
            self.data_particles.label_names, DATA_PARTICLES_LABEL_ORDER)
        unordered_data_optics_labels = np.setdiff1d(
            self.data_optics.label_names, DATA_OPTICS_LABEL_ORDER)

        # reorder data particles to fit DATA_PARTICLES_LABEL_ORDER
        n = 0
        updated_data_particles_label_order = OrderedDict()
        for label in DATA_PARTICLES_LABEL_ORDER:
            if label in self.data_particles.label_names:
                updated_data_particles_label_order[n] = label
                n += 1
        for label in unordered_data_particles_labels:
            updated_data_particles_label_order[n] = label
            n += 1
        self.data_particles._label_order = updated_data_particles_label_order

        # reorder data particles to fit DATA_OPTICS_LABEL_ORDER
        n = 0
        updated_data_optics_label_order = OrderedDict()
        for label in DATA_OPTICS_LABEL_ORDER:
            if label in self.data_optics.label_names:
                updated_data_optics_label_order[n] = label
                n += 1
        for label in unordered_data_optics_labels:
            updated_data_optics_label_order[n] = label
            n += 1
        self.data_optics._label_order = updated_data_optics_label_order
        return
