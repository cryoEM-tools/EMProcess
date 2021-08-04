import numpy as np

def load_particles(filename):
    params = _load_particles_params(filename)
    return particles_star(**params)

def _load_particles_params(filename):
    optics_columns = [
        ['_rlnOpticsGroupName', '#1'],
        ['_rlnOpticsGroup', '#2'],
        ['_rlnMtfFileName', '#3'],
        ['_rlnMicrographOriginalPixelSize', '#4'],
        ['_rlnVoltage', '#5'],
        ['_rlnSphericalAberration', '#6'],
        ['_rlnAmplitudeContrast', '#7'],
        ['_rlnImagePixelSize', '#8'],
        ['_rlnImageSize', '#9'],
        ['_rlnImageDimensionality', '#10']]

    particle_column_possibilities = np.array(
        [
            ["_rlnCoordinateX"],
            ["_rlnCoordinateY"],
            ["_rlnImageName"],
            ["_rlnMicrographName"],
            ["_rlnOpticsGroup"],
            ["_rlnCtfMaxResolution"],
            ["_rlnCtfFigureOfMerit"],
            ["_rlnDefocusU"],
            ["_rlnDefocusV"],
            ["_rlnDefocusAngle"],
            ["_rlnCtfBfactor"],
            ["_rlnCtfScalefactor"],
            ["_rlnPhaseShift"],
            ["_rlnGroupNumber"],
            ["_rlnAngleRot"],
            ["_rlnAngleTilt"],
            ["_rlnAnglePsi"],
            ["_rlnOriginXAngst"],
            ["_rlnOriginYAngst"],
            ["_rlnClassNumber"],
            ["_rlnNormCorrection"],
            ["_rlnLogLikeliContribution"],
            ["_rlnMaxValueProbDistribution"],
            ["_rlnNrOfSignificantSamples"],
            ["_rlnRandomSubset"]])

    optics_vars = np.array(
        [
            ["optics_group_name"],
            ["optics_group"],
            ["mtf_filename"],
            ["mic_orig_pix_size"],
            ["voltage"],
            ["spherical_aberration"],
            ["amplitude_contrast"],
            ["image_pix_size"],
            ["image_size"],
            ["image_dimensionality"]]).reshape(-1)

    particle_vars = np.array(
        [
            ["x_coords"],
            ["y_coords"],
            ["image_names"],
            ["mic_names"],
            ["optics_groups"],
            ["max_res"],
            ["fig_of_merit"],
            ["defocus_U"],
            ["defocus_V"],
            ["defocus_angle"],
            ["ctf_bfactor"],
            ["ctf_scale_factor"],
            ["phase_shift"],
            ["group_number"],
            ["ang_rot"],
            ["ang_tilt"],
            ["ang_psi"],
            ["origin_x_angst"],
            ["origin_y_angst"],
            ["class_num"],
            ["norm_correction"],
            ["ll_contrib"],
            ["max_val_prob_dist"],
            ["n_sig_samples"],
            ["rand_subset"]]).reshape(-1)

    pull_data_optics = False
    pull_data_particles = False
    with open(filename) as f:
        data = f.readlines()

        for n in np.arange(len(data)):
            ln = data[n]
            if (ln.find("data_optics") == 0):
                pull_data_optics = True
                pull_data_particles = False
            elif (ln.find("data_particles") == 0):
                pull_data_optics = False
                pull_data_particles = True
            if (ln.find("loop_") == 0) and pull_data_optics:
                try:
                    observed_optics_columns = data[n+1:n+11]
                    for i in np.arange(len(observed_optics_columns)):
                        assert observed_optics_columns[i].split() == optics_columns[i]
                except:
                    print(observed_optics_columns[i].split(), optics_columns[i])
                    print(
                        "unrecognized organization of data optics."
                        "Perhaps i don't understand the format perfectly?")
                    raise

                data_optics = []
                i = n+11
                ln = data[i]
                while ln.split() != [] :
                    data_optics.append(ln.split())
                    i += 1
                    ln = data[i+1]
            elif (ln.find("loop_") == 0) and pull_data_particles:
                try:
                    observed_data_columns = data[n+1:n+26]
                    column_ids = []
                    for i in np.arange(26):
                        col_val = data[n+i].split()[0]

                        col_id = np.where(particle_column_possibilities == col_val)[0]
                        if col_id.size == 0:
                            col_id = [-np.inf]
                        column_ids.append(col_id[0])
                    column_ids = np.array(column_ids)
                    column_ids = np.array(column_ids[np.isfinite(column_ids)], dtype=int)
                except:
                    pass
                data_particles = []
                i = n+1+column_ids.shape[0]
                ln = data[i]
                while ln.split() != [] :
                    data_particles.append(ln.split())
                    i += 1
                    ln = data[i+1]
    data_particles = np.array(data_particles)
    data_optics = np.array(data_optics)
    params = {}
    for n in np.arange(optics_vars.shape[0]):
        params[optics_vars[n]] = data_optics[:,n]
    for n in np.arange(column_ids.shape[0]):
        params[particle_vars[column_ids][n]] = data_particles[:,n]
    return params

class particles_star:
    def __init__(
            self, optics_group_name=None, optics_group=None, mtf_filename=None,
            mic_orig_pix_size=None, voltage=None, spherical_aberration=None,
            amplitude_contrast=None, mic_pix_size=None, image_pix_size=None, image_size=None,
            image_dimensionality=None, x_coords=None, y_coords=None, image_names=None,
            mic_names=None, optics_groups=None, max_res=None, fig_of_merit=None,
            defocus_U=None, defocus_V=None, defocus_angle=None, ctf_bfactor=None,
            ctf_scale_factor=None, phase_shift=None, group_number=None, ang_rot=None,
            ang_tilt=None, ang_psi=None, origin_x_angst=None, origin_y_angst=None,
            class_num=None, norm_correction=None, ll_contrib=None, max_val_prob_dist=None,
            n_sig_samples=None, rand_subset=None):
        # optics group info
        self._optics_group_name = optics_group_name
        self._optics_group = optics_group
        self._mtf_filename = mtf_filename
        self._mic_orig_pix_size = mic_orig_pix_size
        self._voltage = voltage
        self._spherical_aberration = spherical_aberration
        self._amplitude_contrast = amplitude_contrast
        self._image_pix_size = image_pix_size
        self._image_size = image_size
        self._image_dimensionality = image_dimensionality
        
        # data particles info
        self._x_coords = x_coords
        if x_coords is not None:
            self._x_coords = np.array(x_coords, dtype=float)
        
        self._y_coords = y_coords
        if y_coords is not None:
            self._y_coords = np.array(y_coords, dtype=float)
        
        self._image_names = image_names
        if image_names is not None:
            self._image_names = np.array(image_names, dtype=str)
            
        self._mic_names = mic_names
        if mic_names is not None:
            self._mic_names = np.array(mic_names, dtype=str)
        
        self._max_res = max_res
        if max_res is not None:
            self._max_res = np.array(max_res, dtype=float)
        
        self._fig_of_merit = fig_of_merit
        if fig_of_merit is not None:
            self._fig_of_merit = np.array(fig_of_merit, dtype=float)
            
        self._defocus_U = defocus_U
        if defocus_U is not None:
            self._defocus_U = np.array(defocus_U, dtype=float)
            
        self._defocus_V = defocus_V
        if defocus_V is not None:
            self._defocus_V = np.array(defocus_V, dtype=float)
            
        self._defocus_angle = defocus_angle
        if defocus_angle is not None:
            self._defocus_angle = np.array(defocus_angle, dtype=float)
            
        self._ctf_bfactor = ctf_bfactor
        if ctf_bfactor is not None:
            self._ctf_bfactor = np.array(ctf_bfactor, dtype=float)
            
        self._ctf_scale_factor = ctf_scale_factor
        if ctf_scale_factor is not None:
            self._ctf_scale_factor = np.array(ctf_scale_factor, dtype=float)
            
        self._phase_shift = phase_shift
        if phase_shift is not None:
            self._phase_shift = np.array(phase_shift, dtype=float)
            
        self._group_number = group_number
        if group_number is not None:
            self._group_number = np.array(group_number, dtype=int)
            
        self._ang_rot = ang_rot
        if ang_rot is not None:
            self._ang_rot = np.array(ang_rot, dtype=float)
            
        self._ang_tilt = ang_tilt
        if ang_tilt is not None:
            self._ang_tilt = np.array(ang_tilt, dtype=float)
            
        self._ang_psi = ang_psi
        if ang_psi is not None:
            self._ang_psi = np.array(ang_psi, dtype=float)
            
        self._origin_x_angst = origin_x_angst
        if origin_x_angst is not None:
            self._origin_x_angst = np.array(origin_x_angst, dtype=float)
            
        self._origin_y_angst = origin_y_angst
        if origin_y_angst is not None:
            self._origin_y_angst = np.array(origin_y_angst, dtype=float)
            
        self._class_num = class_num
        if class_num is not None:
            self._class_num = np.array(class_num, dtype=int)
            
        self._norm_correction = norm_correction
        if norm_correction is not None:
            self._norm_correction = np.array(norm_correction, dtype=float)
            
        self._ll_contrib = ll_contrib
        if ll_contrib is not None:
            self._ll_contrib = np.array(ll_contrib, dtype=float)
            
        self._max_val_prob_dist = max_val_prob_dist
        if max_val_prob_dist is not None:
            self._max_val_prob_dist = np.array(max_val_prob_dist, dtype=float)
            
        self._n_sig_samples = n_sig_samples
        if n_sig_samples is not None:
            self._n_sig_samples = np.array(n_sig_samples, dtype=int)
            
        self._rand_subset = rand_subset
        if rand_subset is not None:
            self._rand_subset = np.array(rand_subset, dtype=int)
        
    @property
    def angles(self):
        ang_output = np.vstack([self._ang_rot, self._ang_tilt, self._ang_psi]).T
        return ang_output

    def _optics_data_line(self, n):
        return " ".join(
            [
                new_particles._optics_group_name[n],
                '{0: >12}'.format(new_particles._optics_group[n]),
                new_particles._mtf_filename[n],
                '{0: >12}'.format(new_particles._mic_orig_pix_size[n]),
                '{0: >12}'.format(new_particles._voltage[n]),
                '{0: >12}'.format(new_particles._spherical_aberration[n]),
                '{0: >12}'.format(new_particles._amplitude_contrast[n]),
                '{0: >12}'.format(new_particles._image_pix_size[n]),
                '{0: >12}'.format(new_particles._image_size[n]),
                '{0: >12}'.format(new_particles._image_dimensionality[n]),
                '\n'
            ])
    
    @property
    def optics_data(self):
        return [self._optics_data_line(n) for n in np.arange(len(self._optics_group_name))]
    
    def _particles_data_line(self, n):
        string = []

        try:
            string.append('{0: >#012.6f}'.format(new_particles._x_coords[n]))
        except:
            pass 

        try:
            string.append('{0: >#012.6f}'.format(new_particles._y_coords[n]))
        except:
            pass

        try:
            string.append(new_particles._image_names[n])
        except:
            pass

        try:
            string.append(new_particles._mic_names[n])
        except:
            pass

        try:
            string.append('{0: >12}'.format(new_particles._optics_group[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._max_res[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._fig_of_merit[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._defocus_U[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._defocus_V[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._defocus_angle[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._ctf_bfactor[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._ctf_scale_factor[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._phase_shift[n]))
        except:
            pass

        try:
            string.append('{0: >12}'.format(new_particles._group_number[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._ang_rot[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._ang_tilt[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._ang_psi[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._origin_x_angst[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._origin_y_angst[n]))
        except:
            pass

        try:
            string.append('{0: >12}'.format(new_particles._class_num[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._norm_correction[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6e}'.format(new_particles._ll_contrib[n]))
        except:
            pass

        try:
            string.append('{0: >#012.6f}'.format(new_particles._max_val_prob_dist[n]))
        except:
            pass

        try:
            string.append('{0: >12}'.format(new_particles._n_sig_samples[n]))
        except:
            pass

        try:
            string.append('{0: >12}'.format(new_particles._rand_subset[n]))
        except:
            pass
        
        string.append("\n")

        return " ".join(string)
        
    
    @property
    def particles_data(self):
        return [self._particles_data_line(n) for n in np.arange(len(self._x_coords))]
    
    def set_angles(self, angles):
        angles = np.array(angles)
        self._ang_rot, self._ang_tilt, self._ang_psi = angles.T
        return
