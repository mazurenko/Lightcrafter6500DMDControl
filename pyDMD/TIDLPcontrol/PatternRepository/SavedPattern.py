import PotetialGeneration.SavableObject as so
import os
import matplotlib.pyplot as plt

"""
NOTE: PATTERNS LIVE UNDER V:\\DMD\\ !!!
"""

class SavedPattern(so.SavableThing):

    DMD_pattern_save_dir = 'V:\\DMD\\'
    """
    Class for easy saving of patterns
    """
    def __init__(self, pattern, real_valued_pattern, name, **kwargs):
        print kwargs
        kwargs['save_directory'] = SavedPattern.DMD_pattern_save_dir
        kwargs['skip_uuid'] = True
        is_save_png = kwargs.pop('is_save_png_fig', True)
        self.pattern = pattern
        self.real_valued_pattern = pattern
        super(SavedPattern, self).__init__(name, **kwargs)
        self.save_pkl(**kwargs)
        if is_save_png:
            self.save_figure(pattern, real_valued_pattern)

    def save_figure(self, pattern, real_valued_pattern):
        """
        Saves a user friendly figure alongside the pickle, to make it clear what it is
        :param potential:
        :return:
        """
        fig = plt.figure(figsize=(10, 10), dpi=1000)
        plt.gray()
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2)

        ax0.imshow(real_valued_pattern, vmin=0, vmax=1, interpolation='none')
        ax1.imshow(pattern, vmin=0, vmax=1, interpolation='none')

        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        # plt.savefig("%s\\%s.png" % (self.save_directory, self.name))

        plt.savefig("%s\\%s_%s.png" % (self.save_directory, self.name, self.identifier))
        plt.show()


def load_pattern(name):
    """
    finds pattern with the correct name.
    :param name: identifier to search for
    :return: binary matrix corresponding to the pattern
    """
    files = os.listdir(SavedPattern.DMD_pattern_save_dir)
    if name in files:
        return so.SavableThing.load_from_pkl(os.path.join(SavedPattern.DMD_pattern_save_dir, name)).pattern
    else:
        raise Exception('File {} not found'.format(name))

if __name__ == "__main__":
    #pattern = [1, 2, 3]
    #SavedPattern(pattern, 'test')
    print load_pattern('TEST_20161012_2217.pkl')#'test_20161012_2119.pkl')
