'''
Created on Mar 3, 2016

@author: anton
'''
from UsbControl import LightCrafter6500Device as lc
import UsbControl.PatternCreation as pc

from dmd_reg import DMDBeamCorrector
#import TIDLPImage_Plane.ImagePlanePotentialGenerator as ippg
import PotetialGeneration.ImagePlanePotentialGenerator as ippg
import matplotlib.pyplot as plt

import numpy as np
from random import random

def real_pattern_show():

    lc_dmd = lc.LC6500Device()


    #Prepare Pattern
    settings = {'function':pc.circle_fun_2,
                'compression':'rle',
                'exposure_time':100000000 # in us
                }
    dmd_pattern = pc.DMDPattern(**settings)

    is_debug = False

    sites_dict={}

    circle_dict = {
        (0, 0):ippg.BoxFunction(radii=[200,200], x_angle=0, amplitude=1),
        (0, 0):ippg.CircFunction(radii=[500,500], x_angle=0, amplitude=1),
    }

    entire_beam_dict = {
        (0, 0):ippg.CircFunction(radii=[5000,5000], x_angle=0, amplitude=1),
    }

    multipoint_dict = {
        (-100, -100):ippg.CircFunction(radii=[5,5], x_angle=0, amplitude=1),
        (100, -100):ippg.CircFunction(radii=[5,5], x_angle=0, amplitude=1),
        (-100, 100):ippg.CircFunction(radii=[5,5], x_angle=0, amplitude=1),
        (100, 100):ippg.CircFunction(radii=[5,5], x_angle=0, amplitude=1),
        (-150, -150):ippg.CircFunction(radii=[5,5], x_angle=0, amplitude=1),
        (150, -150):ippg.CircFunction(radii=[5,5], x_angle=0, amplitude=1),
        (50, -50):ippg.CircFunction(radii=[5,5], x_angle=0, amplitude=1),
    }

    sites_dict = {(0, 0): ippg.Parabola_EllipseDip(parab=[400, 600], amplitude=1,
                                               ell=[80, 120], ell_height=0.0,
                                               )}
    #93
    # settings = {'zero_pixel': (824, 429),
    #'zero_pixel': (762, 429),
    settings = {'zero_pixel': (762, 586),
                'sites': sites_dict,
                'binarization_algorithm':'randomize'}

    potential = ippg.ProjectedPotential(**settings)
    evaluated_matrix = potential.evaluate_potential()
    binary_matrix = potential.binarize()

    if True:
        corrector = DMDBeamCorrector('points.pgm',
            'profile.pgm', center=(772, 576))
        arr = np.zeros((1080,1920))
        y, x = np.ogrid[0:1080, 0:1920]

        arr = corrector.homogenize(evaluated_matrix,
            w_center=(762, 586),
            w_size=(300,300))

        if is_debug:
            plt.gray()
            plt.imshow(arr,interpolation='none')
            plt.show()

        for y in range(arr.shape[1]):
            print y
            for x in range(arr.shape[0]):
                # custom floyd-steinberg
                op = arr[x][y]
                # arr[x][y] = 1 if random() < op else 0
                if random() < 0:
                    newp = 1 if random() < op else 0
                else:
                    newp= 1 if 0.5 < op else 0
                arr[x][y] = newp
                err = op - newp
                if x<arr.shape[0]-1:
                    arr[x+1][y] = arr[x+1][y] + err * 7/16
                    if y<arr.shape[1]-1:
                        arr[x+1][y+1] = arr[x+1][y+1] + err * 1/16
                if y<arr.shape[1]-1:
                    arr[x-1][y+1] = arr[x-1][y+1] + err * 3/16
                    arr[x][y+1] = arr[x][y+1] + err * 5/16

        binary_matrix = arr

    if is_debug:
        plt.gray()
        plt.imshow(binary_matrix,interpolation='none')
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

if __name__ == '__main__':
    real_pattern_show()
