'''
Created on Mar 7, 2016

@author: anton
'''

import numpy as np
import matplotlib.pyplot as plt
import abc
from SavableObject import SavableThing, ThisComputer, FileHandler
import ImageDithering.imgDither as dither
import pickle as pkl
import time
import re


'''On Site Function Potentials '''


class SiteFunction(object):
    '''
    base class for individual site potentials
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.amplitude = kwargs.pop('amplitude', 1) # principal x axis in degrees
        self.center = kwargs.pop('center', [0, 0]) # center of the function
        self.offset = kwargs.pop('offset', [0, 0]) # center of the function
        self.radii = kwargs.pop('radii', [1, 1]) # radius in x, y, units are pixels
        self.x_angles = kwargs.pop('x_angle', 0) # principal x axis in degrees

    def transform_coordinates(self, X, Y):
        theta = self.x_angles * np.pi/180.0
        Xprime = np.divide(np.cos(theta)*(X-self.center[0]-self.offset[0])-np.sin(theta)*(Y-self.center[1]-self.offset[1]),
                           self.radii[0])
        Yprime = np.divide(np.sin(theta)*(X-self.center[0]-self.offset[0])+np.cos(theta)*(Y-self.center[1]-self.offset[1]),
                           self.radii[1])
        return Xprime, Yprime

    @abc.abstractmethod
    def evaluate(self, X, Y):
        """
        :param meshgrids - X,Y
        :return the function evaluated over X,Y
        note that the function is real valued. The binarization happens later
        """
        return


class BoxFunction(SiteFunction):

    def __init__(self, **kwargs):
        super(BoxFunction, self).__init__(**kwargs)

    def evaluate(self, X, Y):
        X, Y = self.transform_coordinates(X, Y)
        return self.amplitude*((np.abs(X)<1)*(np.abs(Y)<1))


class CircFunction(SiteFunction):
    
    def __init__(self, **kwargs):
        super(CircFunction, self).__init__(**kwargs)
    
    def evaluate(self, X, Y):
        X, Y = self.transform_coordinates(X, Y)
        R = np.sqrt(np.square(X)+np.square(Y))
        return self.amplitude*(R<1)


class GaussFunction(SiteFunction):
    
    def __init__(self, **kwargs):
        super(GaussFunction, self).__init__(**kwargs)
    
    def evaluate(self, X, Y):
        X, Y = self.transform_coordinates(X, Y)
        R = np.sqrt(np.square(X)+np.square(Y))
        return self.amplitude*np.exp(-np.square(R))


class Parabola_BoxDip(SiteFunction):
    def __init__(self, **kwargs):
        super(Parabola_BoxDip, self).__init__(**kwargs)
        self.parab = kwargs.pop('parab', [1, 1])
        self.box = kwargs.pop('box', [40, 40])
        self.box_height = kwargs.pop('box_height', 0)

    def evaluate(self, X, Y):
        X, Y = self.transform_coordinates(X, Y)
        R = np.sqrt(np.square(X/self.parab[0]) + np.square(Y/self.parab[1]))
        bool_box = (np.abs(X) < self.box[0]) * (np.abs(Y) < self.box[1])
        return (self.amplitude-R**2) * (R**2<self.amplitude) * (1 - bool_box) + bool_box * self.box_height

    
class Parabola_EllipseDip(SiteFunction):
    def __init__(self, **kwargs):
        super(Parabola_EllipseDip, self).__init__(**kwargs)
        self.parab = kwargs.pop('parab', [1, 1])
        self.ell = kwargs.pop('ell', [40, 40])
        self.ell_height = kwargs.pop('ell_height', 0)

    def evaluate(self, X, Y):
        X, Y = self.transform_coordinates(X, Y)
        R = np.sqrt(np.square(X/self.parab[0]) + np.square(Y/self.parab[1]))
        R_ell = np.sqrt(np.square(X/self.ell[0]) + np.square(Y/self.ell[1]))
        bool_box = np.abs(R_ell < 1)
        # 'Batman with a crew cut'
        return (self.amplitude-R**2) * (R**2<self.amplitude) * (1 - bool_box) + bool_box * self.ell_height


class GradientParabolaMinusGauss_GaussDipFloor(SiteFunction):
    def __init__(self, **kwargs):
        super(GradientParabolaMinusGauss_GaussDipFloor, self).__init__(**kwargs)
        self.parab = kwargs.pop('parab', [100, 100])
        self.gausscomp = kwargs.pop('gausscomp', [50, 50]) # 1 sigma width
        self.gausscomp_height = kwargs.pop('gausscomp_height', 0)
        self.gauss = kwargs.pop('gauss', [40, 40]) # 1 sigma width
        self.gauss_height = kwargs.pop('gauss_height', 0)
        self.gradient = kwargs.pop('gradient', [0, 0])
        self.level = kwargs.pop('level', 0)

    def evaluate(self, NE, NW):
        NE, NW = self.transform_coordinates(NE, NW)
        R = np.sqrt(np.square(NE/self.parab[0]) + np.square(NW/self.parab[1]))
        R_gausscomp = np.sqrt(np.square(NE/self.gausscomp[0])/2.0 + np.square(NW/self.gausscomp[1])/2.0)
        bool_boxcomp = np.abs(R_gausscomp < 2)
        R_gauss = np.sqrt(np.square(NE/self.gauss[0])/2.0 + np.square(NW/self.gauss[1])/2.0)
        bool_box = np.abs(R_gauss < 2)
        # 'Smoothed potential'
        val = (self.amplitude-R**2) * (R**2<self.amplitude) - \
                bool_boxcomp * np.exp(-R_gausscomp**2) * self.gausscomp_height - \
                bool_box * np.exp(-R_gauss**2) * self.gauss_height + \
                self.gradient[0]*NE + self.gradient[1]*NW
        val += bool_box*(val < self.level) * (self.level-val)
        print val
        return val


class Parabola_ParabolaDip(SiteFunction):
    def __init__(self, **kwargs):
        super(Parabola_ParabolaDip, self).__init__(**kwargs)
        self.parab = kwargs.pop('parab', [1, 1])
        self.ell = kwargs.pop('ell', [40, 40])
        self.ell_height = kwargs.pop('ell_height', 0)
        self.parab_fac = kwargs.pop('ell_height', 1)

    def evaluate(self, X, Y):
        X, Y = self.transform_coordinates(X, Y)
        R = np.sqrt(np.square(X/self.parab[0]) + np.square(Y/self.parab[1]))
        R_ell = np.sqrt(np.square(X/self.ell[0]) + np.square(Y/self.ell[1]))
        bool_box = np.abs(R_ell < 1)
        # 'Batman potential'
        return (self.amplitude-self.parab_fac*R**2) * (R**2<self.amplitude) - bool_box * self.ell_height


class Parabola_GaussDip(SiteFunction):
    def __init__(self, **kwargs):
        super(Parabola_GaussDip, self).__init__(**kwargs)
        self.parab = kwargs.pop('parab', [1, 1])
        self.gauss = kwargs.pop('gauss', [40, 40]) # 1 sigma width
        self.gauss_height = kwargs.pop('gauss_height', 0)

    def evaluate(self, X, Y):
        X, Y = self.transform_coordinates(X, Y)
        R = np.sqrt(np.square(X/self.parab[0]) + np.square(Y/self.parab[1]))
        R_gauss = np.sqrt(np.square(X/self.gauss[0])/2.0 + np.square(Y/self.gauss[1])/2.0)
        bool_box = np.abs(R_gauss < 2)
        # 'Smoothed potential'
        return (self.amplitude-R**2) * (R**2<self.amplitude) - bool_box * np.exp(-R_gauss**2) * self.gauss_height


class ParabolaMinusGauss_GaussDip(SiteFunction):
    def __init__(self, **kwargs):
        super(ParabolaMinusGauss_GaussDip, self).__init__(**kwargs)
        self.parab = kwargs.pop('parab', [100, 100])
        self.gausscomp = kwargs.pop('gausscomp', [50, 50]) # 1 sigma width
        self.gausscomp_height = kwargs.pop('gausscomp_height', 0)
        self.gauss = kwargs.pop('gauss', [40, 40]) # 1 sigma width
        self.gauss_height = kwargs.pop('gauss_height', 0)

    def evaluate(self, X, Y):
        X, Y = self.transform_coordinates(X, Y)
        R = np.sqrt(np.square(X/self.parab[0]) + np.square(Y/self.parab[1]))
        R_gausscomp = np.sqrt(np.square(X/self.gausscomp[0])/2.0 + np.square(Y/self.gausscomp[1])/2.0)
        bool_boxcomp = np.abs(R_gausscomp < 2)
        R_gauss = np.sqrt(np.square(X/self.gauss[0])/2.0 + np.square(Y/self.gauss[1])/2.0)
        bool_box = np.abs(R_gauss < 2)
        # 'Smoothed potential'
        return (self.amplitude-R**2) * (R**2<self.amplitude) - \
                bool_boxcomp * np.exp(-R_gausscomp**2) * self.gausscomp_height - \
                bool_box * np.exp(-R_gauss**2) * self.gauss_height


class GradientParabolaMinusGauss_GaussDip(SiteFunction):
    def __init__(self, **kwargs):
        super(GradientParabolaMinusGauss_GaussDip, self).__init__(**kwargs)
        self.parab = kwargs.pop('parab', [100, 100])
        self.gausscomp = kwargs.pop('gausscomp', [50, 50]) # 1 sigma width
        self.gausscomp_height = kwargs.pop('gausscomp_height', 0)
        self.gauss = kwargs.pop('gauss', [40, 40]) # 1 sigma width
        self.gauss_height = kwargs.pop('gauss_height', 0)
        self.gradient = kwargs.pop('gradient', [0, 0])

    def evaluate(self, NE, NW):
        NE, NW = self.transform_coordinates(NE, NW)
        R = np.sqrt(np.square(NE/self.parab[0]) + np.square(NW/self.parab[1]))
        R_gausscomp = np.sqrt(np.square(NE/self.gausscomp[0])/2.0 + np.square(NW/self.gausscomp[1])/2.0)
        bool_boxcomp = np.abs(R_gausscomp < 2)
        R_gauss = np.sqrt(np.square(NE/self.gauss[0])/2.0 + np.square(NW/self.gauss[1])/2.0)
        bool_box = np.abs(R_gauss < 2)
        # 'Smoothed potential'
        return (self.amplitude-R**2) * (R**2<self.amplitude) - \
                bool_boxcomp * np.exp(-R_gausscomp**2) * self.gausscomp_height - \
                bool_box * np.exp(-R_gauss**2) * self.gauss_height + \
                self.gradient[0]*NE + self.gradient[1]*NW


class ProjectedPotential(SavableThing):

    def __init__(self, **kwargs):
        """
        TODO: PROPERLY DOCUMENT THIS
        """
        self.resolution = kwargs.pop('resolution', [1920, 1080])
        self.zero_point = kwargs.pop('zero_pixel', [960, 540])
        self.lvecs = kwargs.pop('lvecs', [[1, 0], [0, 1]]) #list of 2 vectors expressed in dmd pixels corresponding to the lattice vectors
        

        #  dictionary converting tuples, which represent sites on the grid (units of sites):site_functions
        self.sites = kwargs.pop('sites', [])

        self.rescaling_algorithm = kwargs.pop('rescaling_algorithm', 'clip')
        '''
        valid values =    'clip'
                          'rescale'
        before the real valued potential is binarized, it is brought into the range 0, 1. this can be either done by scaling & shifting max, (min)
        to 1, (0), or by clipping any values above (below) 1 (0).
        '''

        self.binarization_algorithm = kwargs.pop('binarization_algorithm', 'randomize')
        '''
        the binarization algorithm is what takes us from the real valued function to the 
        values on the DMD. Prior to running this, the real valued function is rescaled such that its range = [0, 1].
            'threshold' = everything above binarization_parameters[0] -> 1, else ->0
            'randomize' = pixels are assigned 1's with the probability equal to the function value
            'error_diffusion' = pixels are assigned 1's with error diffusion
        '''
        #the mask is and-ed to the binarized function after the binarization
        self.binarization_mask = kwargs.pop('binarization_mask', np.ones(self.resolution,dtype = np.bool))

        name = kwargs.get('name', '')
        super(ProjectedPotential, self).__init__(name, **kwargs)

        self.real_valued_potential = np.zeros(self.resolution[::-1])
        self.binarized_potential = np.zeros(self.resolution[::-1])
        self.real_valued_potential = self.evaluate_potential()
        self.binarized_potential = self.binarize()

    def evaluate_potential(self):
        """
        evaluates the real valued potential
        """
        X, Y = np.meshgrid(np.arange(self.resolution[0]), np.arange(self.resolution[1]))
        real_valued_potential = np.zeros(self.resolution[::-1])
        for site_coord, site_function in self.sites:
            print site_coord
            print site_function
            site_function.center = np.array(self.zero_point) + site_coord[0]*np.array(self.lvecs[0]) + site_coord[1]*np.array(self.lvecs[1])
            real_valued_potential += site_function.evaluate(X, Y)
        return real_valued_potential

    def binarize(self, **kwargs):
        """
        rescale and binarize the real valued potential
        """
        renormalized_potential = rescaling_fun_dict[self.rescaling_algorithm](self.real_valued_potential)

        if self.binarization_algorithm in binarization_dict.keys():
            dithering_alg, dithering_settings = binarization_dict[self.binarization_algorithm]
        elif self.binarization_algorithm == 'custom':
            dithering_alg = kwargs.get('binarization_algorthm')
            dithering_settings = kwargs.get('binarization_settings')
        else:
            raise Exception("Unknown binarization settings/algorithm")

        dithering = dithering_alg(renormalized_potential,**dithering_settings)
        return dithering.processed_img

    @classmethod
    def _load_from_pkl(cls, filepath):
        init_time = time.time()
        saved_dict = pkl.load(open(filepath, 'rb'))
        tmp = ProjectedPotential()
        tmp.__dict__.update(saved_dict)
        tmp.comp = ThisComputer()
        final_time = time.time()
        print "Loaded pickle in %s seconds" % (final_time - init_time)
        return tmp

    @classmethod
    def find_and_load(cls, name=None, year=99, month=99, day=99, hour=99, minute=99, identifier_snippet=None):
        """
        Saved DMD files have the format of <NAME (arb length)>_<YEAR (4 char length)><MONTH (2)><DAY (2)>_<HR 2><MIN 2>_<UUID 36>.pkl
        :param name: name of file
        :param year:
        :param month:
        :param day:
        :param hour:
        :param minute:
        :param identifier_snippet:
        Note that 99 is an escape sequence
        :return: Projected potential loaded from matching file, or error if !=1 file was matched.
        """
        # info takes form of (value of thing, length in string)
        esc_seq = re.compile(".*9{2,}")
        is_escape = lambda x: esc_seq.match(x) is not None
        info = [(name, '.*'), (None, '_'), ('%04d' % year, '\d'*4), ('%02d' % month, '\d'*2), ('%02d' % day, '\d'*2),
                (None, '_'), ('%02d' % hour, '\d'*2), ('%02d' % minute, '\d'*2), (None, '_'),
                (identifier_snippet+'.{%s}'%(36-len(identifier_snippet)), '.{,36}'), (None, '\.pkl')]
        assert any(map(lambda x: not((x[0] is None) or (is_escape(x[0]))), info)) # ensure that SOME information was supplied
        reg = ''.join(map(lambda x: x[0] if (x[0] is not None) and (not is_escape(x[0])) else x[1], info))
        filematch_reg = re.compile(reg)
        is_filematch = lambda x: filematch_reg.match(x) is not None
        comp = ThisComputer()
        matched_files = FileHandler.find_file_regex(comp.dmd_data_dir, reg)
        if len(matched_files) != 1:
            raise Exception('Found %s files that satisfied the given constraints! Can only proceed if 1 and only 1 file is found' % len(matched_files))
        else:
            return ProjectedPotential._load_from_pkl('%s\\%s' % (comp.dmd_data_dir, matched_files[0]))


'''Helper Functions for evaluating potential'''

def clip_real_potential(potential, **kwargs):
    return np.clip(potential, 0, 1)


def rescale_real_potential(potential, **kwargs):
    potential -= np.min(potential)
    return potential/np.max(potential)


rescaling_fun_dict = {'clip': clip_real_potential, 'rescale': rescale_real_potential}


binarization_dict = {'threshold': (dither.ThresholdDithering, {'randomness': 0.0}),
                     'randomize': (dither.ThresholdDithering, {'randomness': 0.5}),
                     'error_diffusion': (dither.ErrorDiffusion, {'randomness': 0.05,
                                                                 'error_diffusion_type': 'floyd_steinberg'}),
                     }

'''
test functions
'''


def test_potential_generation():
    site_rad = 40
    site_radii = [site_rad, site_rad]
    sites_dict = {((0, 0), GaussFunction(radii=site_radii)),
                  ((100, 0), GaussFunction(radii=site_radii)),
                  ((0, 100), GaussFunction(radii=site_radii)),
                  ((-50, -50), GaussFunction(radii=[10, 10]))
                  }
    settings = {'sites': sites_dict,
                'binarization_algorithm': 'error_diffusion'}
    potential = ProjectedPotential(**settings)

    potential.save_pkl()

    plt.gray()
    
    plt.subplot(211)
    plt.imshow(potential.real_valued_potential, interpolation='none')

    plt.subplot(212)
    plt.imshow(potential.binarized_potential, interpolation='none')
    plt.show()

def test_potential_receiving():
    #st = ProjectedPotential._load_from_pkl('V:\\dmd_patterns\\_20160714_1638_6b221421-1b08-4225-ab5f-f1fc595605ba.pkl')
    pot = ProjectedPotential.find_and_load('0714_1638')

if __name__ == '__main__':
    #test_potential_generation()
    #test_potential_receiving()
    ProjectedPotential.find_and_load(identifier_snippet='98ce')
