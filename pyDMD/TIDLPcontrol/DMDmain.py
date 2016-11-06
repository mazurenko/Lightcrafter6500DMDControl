'''
Created on Mar 3, 2016

@author: anton
'''
from UsbControl import LightCrafter6500Device as lc
import UsbControl.PatternCreation as pc
#import TIDLPImage_Plane.ImagePlanePotentialGenerator as ippg
import PotetialGeneration.ImagePlanePotentialGenerator as ippg
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave, imread
import pickle as pkl

class BoxCancellation(object):
    l_NW = 2* 12 #central size/2 in NE direction
    l_NE = 5 * 12 #central size/2 in NW direction
    l_long = 20 * 12
    w_long = 3 * 12
    w_short = 10 * 12
    center = (-31, 4)
    amp1 = 0.5
    amp2 = 1
    amp3 = 1
    amp4 = 1
    angle_long = 1
    angle_short = 96

    @classmethod
    def center1(cls):
        dist = np.floor((cls.l_NW + cls.w_short))
        return (cls.center[0], cls.center[1] - dist)

    @classmethod
    def center2(cls):
        dist = np.floor((cls.l_NE + cls.w_long))
        return (cls.center[0] - dist, cls.center[1])

    @classmethod
    def center3(cls):
        dist = np.floor((cls.l_NW + cls.w_short))
        return (cls.center[0], cls.center[1] + dist)

    @classmethod
    def center4(cls):
        dist = np.floor((cls.l_NE + cls.w_long))
        return (cls.center[0] + dist, cls.center[1])

    @classmethod
    def get_boxes(cls):
        box_dict = {
                    cls.center1():ippg.BoxFunction(radii = [cls.l_NE, cls.w_short], x_angle=cls.angle_long,
                                                   amplitude=cls.amp1),
                    cls.center2():ippg.BoxFunction(radii = [cls.w_long, cls.l_long], x_angle=cls.angle_long,
                                                   amplitude=cls.amp2),
                    cls.center3():ippg.BoxFunction(radii = [cls.l_NE, cls.w_short], x_angle=cls.angle_long,
                                                   amplitude=cls.amp3),
                    cls.center4():ippg.BoxFunction(radii = [cls.w_long, cls.l_long], x_angle=cls.angle_long,
                                                   amplitude=cls.amp4),
                    }
        return box_dict

def raw_input_test():
    lc_dmd = lc.LC6500Device()
    lc_dmd.raw_input_test_sequence()


def standard_circle_test():
    lc_dmd = lc.LC6500Device()

    # Prepare Pattern
    settings = {'function':pc.circle_fun_2,
                'compression':'rle'
                }
    dmd_pattern = pc.DMDPattern(**settings)
    dmd_pattern.compute_pattern()
    lc_dmd.upload_image(dmd_pattern)

    #command_sequence = dmd_pattern.compress_pattern()
    #lc_dmd.upload_image(command_sequence)
    
def video_pattern_test():
    lc_dmd = lc.LC6500Device()
    lc_dmd.video_pattern_mode_test()

def real_pattern_test():

    lc_dmd = lc.LC6500Device()


    #Prepare Pattern
    #settings = {'function':pc.stripes_2,#pc.circle_fun_2,
    settings = {'function':pc.circle_fun_2,
                'compression':'rle',
                'exposure_time':500000
                }
    dmd_pattern = pc.DMDPattern(**settings)

    
    #dmd_pattern.compute_pattern()
    site_rad = 6
    site_radii = [site_rad, site_rad]
    sites_dict = {#(0,0):ippg.CircFunction(radii = site_radii),
                  #(1,0):ippg.CircFunction(radii = site_radii),
                  #(3,1):ippg.CircFunction(radii = site_radii),
                  #(0,0):ippg.GaussFunction(radii = [1000,1000])
                  (0,0):ippg.GaussFunction(radii = [500,500])
                  }
   #settings = {'sites':sites_dict,
    #            'binarization_algorithm':'error_diffusion'}

    # sites_dict = {(-100,-158):ippg.CircFunction(radii = [4,4], x_angle=1, amplitude=1)}
    #sites_dict = {(-188,-264):ippg.CircFunction(radii = [4,4], x_angle=1, amplitude=1)}

    settings = {'sites':sites_dict,
                'binarization_algorithm':'threshold'}
                #'binarization_algorithm':'error_diffusion'}
    potential = ippg.ProjectedPotential(**settings)
    evaluated_matrix = potential.evaluate_potential()
    binary_matrix = potential.binarize()
 
    dmd_pattern.pattern = binary_matrix 
    #dmd_pattern.show_pattern()
    
    lc_dmd.upload_image(dmd_pattern)

    #command_sequence = dmd_pattern.compress_pattern()
    #lc_dmd.upload_image(command_sequence)
    
def real_pattern_series_test():

    lc_dmd = lc.LC6500Device()
    print lc_dmd.dev
    #Prepare Pattern
    #settings = {'function':pc.stripes_2,#pc.circle_fun_2,
    settings = {'function':pc.circle_fun_2,
                'compression':'rle',
                'exposure_time':500000 # in us
                }
    dmd_pattern = pc.DMDPattern(**settings)
    dmd_pattern_2 = pc.DMDPattern(**settings)

    
    #dmd_pattern.compute_pattern()
    site_rad = 6
    site_radii = [site_rad, site_rad]
    sites_dict = {(-25,-15):ippg.GaussFunction(radii = [1000,200])}
    settings = {'sites':sites_dict,
                'binarization_algorithm':'randomize'}

    potential = ippg.ProjectedPotential(**settings)
    evaluated_matrix = potential.evaluate_potential()
    binary_matrix = potential.binarize()
    dmd_pattern.pattern = binary_matrix 


    sites_dict = {(-25,-15):ippg.GaussFunction(radii = [200,1000])}

    settings = {'sites':sites_dict,
                'binarization_algorithm':'threshold'}

    potential = ippg.ProjectedPotential(**settings)
    evaluated_matrix = potential.evaluate_potential()
    binary_matrix = potential.binarize()
    
    dmd_pattern_2.pattern = binary_matrix
    
    dmd_patterns = {'patterns':[dmd_pattern, dmd_pattern_2, dmd_pattern]}
    lc_dmd.upload_image_sequence(dmd_patterns)
    
    #lc_dmd.upload_image(dmd_pattern)

    #command_sequence = dmd_pattern.compress_pattern()
    #lc_dmd.upload_image(command_sequence)
    lc_dmd.release()

def manual_read():
    lc_dmd = lc.LC6500Device()


def real_pattern_print():

    #Prepare Pattern
    lc_dmd = lc.LC6500Device()

    settings = {
                'compression':'rle',
                'exposure_time':5000000 # in us
                }
    dmd_pattern = pc.DMDPattern(**settings)

    is_debug = False

   #TEST STUFF
    sites_dict = [((0, 0), ippg.GradientParabolaMinusGauss_GaussDip(parab=[600*1.5, 900*1.5], amplitude=1,
                                                  gausscomp=[120, 180], gausscomp_height=0.2,
                                                  gauss=[60, 90], gauss_height=1.0,
                                                  #gradient=[0.0, 0.0]
                                                  gradient=[-0.0003, 0.0002] # [++ -> upper right on atoms, ++ -> upper left]
                                                                     ))
                                         ]

    # settings = {'zero_pixel': (824, 429),
    settings = {'zero_pixel': (762, 586),
                'sites': sites_dict,
                'binarization_algorithm': 'threshold'}#''error_diffusion'}

    potential = ippg.ProjectedPotential(**settings)

    # uncomment this to get a saved copy of the potential
    # imsave('out.png', evaluated_matrix)

    # uncomment this to use the painted potential
    # potential.real_valued_potential = np.array(imread('out.png', mode='L')).astype('float')
    # potential.real_valued_potential /= float(np.max(potential.real_valued_potential))

    binary_matrix =potential.binarized_potential

    if is_debug:
        plt.gray()
        plt.imshow(binary_matrix,interpolation='none')
        # plt.imshow(potential.real_valued_potential,interpolation='none')
        plt.figaspect(1)
        pkl.dump(binary_matrix, open('diffused.pkl', 'wb'), protocol=2)

        plt.show()

    dmd_pattern.pattern = binary_matrix

    dmd_patterns = {'patterns':[dmd_pattern]}
    lc_dmd.upload_image_sequence(dmd_patterns)

def real_pattern_show():

    lc_dmd = lc.LC6500Device()
    print lc_dmd.dev

    #Prepare Pattern
    settings = {
                'compression':'rle',
                'exposure_time':5000000 # in us
                }
    dmd_pattern = pc.DMDPattern(**settings)

    is_debug = False

    # sites_dict={}
    # sites_dict = {(-31,4):ippg.BoxFunction(radii = [30,10], x_angle=1),
    #               (120, 72):ippg.
    # BoxFunction(radii=[30,3], x_angle=2.5),
    #               (-31, 4):ippg.BoxFunction(radii=[30,10], x_angle=1)
    #               }

    # sites_dict = {(0, 0):ippg.CircFunction(radii = [50,50], x_angle=1, amplitude=1)}
    # # sites_dict = BoxCancellation.get_boxes()
    # sites_dict = {(0,-0):ippg.GaussFunction(radii = [100,100])}

    # sites_dict = {(0, 0): ippg.Parabola_BoxDip(parab=[600, 900], amplitude=1,
                                              # box=[1, 1], box_height=0.0,
                                              # )}


    #sites_dict = {(0, 0): ippg.Parabola_EllipseDip(parab=[600, 900], amplitude=1,
                                               # ell=[40, 60], ell_height=0.0,
                                               # )}
    # sites_dict = {(0, 0): ippg.Parabola_GaussDip(parab=[600, 900], amplitude=1,
    #                                            gauss=[60, 90], gauss_height=1.,
    #                                            )}

    #LAB STUFF
    '''
    sites_dict = {(0, 0): ippg.GradientParabolaMinusGauss_GaussDip(parab=[600*1.5, 900*1.5], amplitude=1,
                                                  gausscomp=[120, 180], gausscomp_height=0.2,
                                                  gauss=[60, 90], gauss_height=1.0,
                                                  #gradient=[0.0, 0.0]
                                                  gradient=[-0.0003, 0.0002] # [++ -> upper right on atoms, ++ -> upper left]
                                         )}
    '''
    #TEST STUFF
    sites_dict = {(0, 0): ippg.GradientParabolaMinusGauss_GaussDip(parab=[600*1.5, 900*1.5], amplitude=1,
                                                  gausscomp=[120, 180], gausscomp_height=0.2,
                                                  gauss=[60, 90], gauss_height=1.0,
                                                  #gradient=[0.0, 0.0]
                                                  gradient=[-0.0003, 0.0002] # [++ -> upper right on atoms, ++ -> upper left]
                                         )}

    # settings = {'zero_pixel': (824, 429),
    settings = {'zero_pixel': (762, 586),
                'sites': sites_dict,
                'binarization_algorithm': 'threshold'}

    potential = ippg.ProjectedPotential(**settings)
    evaluated_matrix = potential.evaluate_potential()

    # uncomment this to get a saved copy of the potential
    # imsave('out.png', evaluated_matrix)

    # uncomment this to use the painted potential
    # potential.real_valued_potential = np.array(imread('out.png', mode='L')).astype('float')
    # potential.real_valued_potential /= float(np.max(potential.real_valued_potential))

    binary_matrix = potential.binarize()

    if is_debug:
        plt.gray()
        plt.imshow(binary_matrix,interpolation='none')
        # plt.imshow(potential.real_valued_potential,interpolation='none')
        plt.figaspect(1)
        plt.show()

    dmd_pattern.pattern = binary_matrix

    dmd_patterns = {'patterns':[dmd_pattern]}
    lc_dmd.upload_image_sequence(dmd_patterns)


def manual_test():

    lc_dmd = lc.LC6500Device()
    lc_dmd.start_stop_sequence('stop')
    lc_dmd.set_pattern_on_the_fly()

def test_linux_ctl():

    lc_dmd = lc.LC6500Device()

def video_normal_mode():
    lc_dmd = lc.LC6500Device()
    lc_dmd.set_video_mode()

if __name__ == '__main__':
    real_pattern_print()
    #video_normal_mode()
    #manual_test()
    #standard_circle_test()
 
    #real_pattern_test()
    #real_pattern_series_test()
    #standard_circle_test()
    #real_pattern_show()
