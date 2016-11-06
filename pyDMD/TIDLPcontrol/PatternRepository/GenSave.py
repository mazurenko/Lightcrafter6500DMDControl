import PotetialGeneration.ImagePlanePotentialGenerator as ippg
from SavedPattern import SavedPattern

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl



def save_manual_pattern(dmd_settings={}):
    """
    Manually save a pickle and a png showing the pattern
    :param dmd_settings: optional dict of dmd settings. required if you want to use pattern on the fly mode
    :return:
    """

    name = "TEST"
    sites = [
        ((0, 0), ippg.BoxFunction(radii=[20, 10], x_angle=1)),
        ((0, 0), ippg.GaussFunction(radii=[100, 500], x_angle=1))
    ]

    pattern_settings = {
                'zero_pixel': (762, 429),
                'sites': sites,
                'binarization_algorithm': 'error_diffusion'}

    potential = ippg.ProjectedPotential(**pattern_settings)
    SavedPattern(potential.binarized_potential, potential.real_valued_potential, name, metadata=pattern_settings)

def save_redist_pattern_figure(level = 0):
    name = "redist_figure"

    entropy_redist_potential = [((70, -110), ippg.GradientParabolaMinusGauss_GaussDipFloor(
        #parab=[600, 900], amplitude=1,
        parab=[600/2., 900/2.], amplitude=1,
        gausscomp=[120, 180],
        gausscomp_height=0.2,
        level=level,
        # gauss=[60*.75*1.0, 60*.75*1.0], gauss_height=1,
        gauss=[60*1.0, 60*1.0], gauss_height=1,
        #gradient=[0.0000, 0.0002]# [++ -> upper right on atoms, ++ -> upper left]
        gradient=[0.0006, -0.0008]# [++ -> upper right on atoms, ++ -> upper left]
    ))]

    pattern_settings = {'zero_pixel': (702, 816),
                'sites': entropy_redist_potential,
                'binarization_algorithm':'error_diffusion'}

    potential = ippg.ProjectedPotential(**pattern_settings)
    binarized, real_valued = potential.binarized_potential, potential.real_valued_potential

    return binarized, real_valued

def make_figure_patterns(levels = [0, 0.4]):
    output = {}
    for level in levels:
        print "making figure for level {}".format(level)
        output[level] = save_redist_pattern_figure(level = level)
    pkl.dump(output, open("paper_figs_dict.pkl", "wb"))
    return 0


def save_redist_pattern():
    name = "redist"

    entropy_redist_potential = [((70, -110), ippg.GradientParabolaMinusGauss_GaussDipFloor(
        #parab=[600, 900], amplitude=1,
        parab=[600/2., 900/2.], amplitude=1,
        gausscomp=[120, 180],
        gausscomp_height=0.2,
        level=0.00,
        # gauss=[60*.75*1.0, 60*.75*1.0], gauss_height=1,
        gauss=[60*1.0, 60*1.0], gauss_height=1,
        #gradient=[0.0000, 0.0002]# [++ -> upper right on atoms, ++ -> upper left]
        gradient=[0.0006, -0.0008]# [++ -> upper right on atoms, ++ -> upper left]
    ))]

    # gradient=[0.0036, 0.0012]# [++ -> upper right on atoms, ++ -> upper left]
    # entropy_redist_potential = [((30, -60), ippg.GradientParabolaMinusGauss_GaussDipFloor(
    #                                               #parab=[600, 900], amplitude=1,
    #                                               parab=[600/2., 900/2.], amplitude=1,
    #                                               gausscomp=[120, 180],
    #                                               gausscomp_height=0.2,
    #                                               level=0.0, #0.42, #0,
    #                                               gauss=[60*.75*1.0, 60*.75*1.0], gauss_height=1,
    #                                               #gradient=[0.0000, 0.0002]# [++ -> upper right on atoms, ++ -> upper left]
    #                                               gradient=[0.0014, -0.0006]# [++ -> upper right on atoms, ++ -> upper left]

                   #                      ))]

    pattern_settings = {'zero_pixel': (702, 816),
                'sites': entropy_redist_potential,
                'binarization_algorithm':'error_diffusion'}

    potential = ippg.ProjectedPotential(**pattern_settings)
    SavedPattern(potential.binarized_potential, potential.real_valued_potential, name, metadata=pattern_settings)




def save_focusing_pattern():
    name = "focusing"

    entropy_redist_potential = [((30, -60), ippg.BoxFunction(radii=[20, 10])),
                                ((80, 60), ippg.BoxFunction(radii=[20, 10])),
                                ((30, 60), ippg.BoxFunction(radii=[20, 10])),
                                ((80, -60), ippg.BoxFunction(radii=[20, 10]))]

    pattern_settings = {'zero_pixel': (702, 816),
                'sites': entropy_redist_potential,
                'binarization_algorithm':'threshold '}

    potential = ippg.ProjectedPotential(**pattern_settings)
    SavedPattern(potential.binarized_potential, potential.real_valued_potential, name, metadata=pattern_settings)



if __name__ == "__main__":
    #save_manual_pattern()
    #save_redist_pattern_figure()
    #save_focusing_pattern()
    make_figure_patterns()