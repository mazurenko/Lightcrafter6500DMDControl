'''
Created on Mar 3, 2016

@author: anton
'''
import numpy as np
import matplotlib.pyplot as plt
import HexHelper as hh


def mesh_radius(X,Y):
    return np.sqrt(np.square(X)+np.square(Y))


class DMDPattern(object):
    
    def __init__(self, **kwargs):
        self.function = kwargs.pop('function', None)
        self.resolution = kwargs.pop('resolution', (1920, 1080))
        self.zero_point = kwargs.pop('zero_pixel', (960, 540))
        self.compression = kwargs.pop('compression', 'none')
        self.exposure_time = kwargs.pop('exposure_time', 105)
        self.dark_time = kwargs.pop('dark_time', 0)
        self.bit_depth = kwargs.pop('bit_depth', 1) 

        self.X, self.Y = np.meshgrid(np.arange(self.resolution[0]) - self.zero_point[0],
                                     np.arange(self.resolution[1]) - self.zero_point[1])
        self.pattern = np.zeros(self.resolution)
        
        self.compressed_pattern = None
        self.compressed_pattern_length = None
        
    def compute_pattern(self):
        """
        returns the compressed pattern and its' length
        """
        self.pattern = self.function(self.X, self.Y)

    def compress_pattern(self):
        """
        :return the pattern as a 1d array of uint8's, including the appropriate image header
        """
        #make 1d, apply compression
        self.pattern_3d = 0xff*np.dstack((self.pattern,self.pattern,self.pattern))
        compressed_pattern, pattern_length = compression_function_dict[self.compression](self.pattern_3d)

        #get the header
        header = make_header(pattern_length, self.compression)
        command_sequence = header+compressed_pattern
        
        print "Compressed to %s bytes" %len(command_sequence) 
        cs_uint = np.array(command_sequence,dtype=np.uint8)
        #print "end sequences %s"% np.where(cs_uint == 0x01)
        return cs_uint

    def print_pattern(self):
        print self.pattern

    def show_pattern(self):
        plt.gray()
        plt.imshow(self.pattern,interpolation='none')
        plt.show()


'''Compression Math'''


def no_comp_image(pattern):
    """
    :param (M, N) matrix of uint8 that represents a pattern
    :return 1D 24bit compressed bitmap
    """
    compressed_pattern = np.packbits(np.ravel(pattern))
    
    return compressed_pattern, np.size(compressed_pattern)


def erle_comp_image(pattern):
    """
    :param (M, N, 3) matrix of uint8 that represents a pattern
    :return 1D 24bit compressed bitmap
    """
    pass


def standard_rle_compression(x, zero_index = 0):
    """
    :param (N,3) array that corresponds to part or all of one row of an image.
    :param zero index is an offset of the indexes.
    :returns first indexes of runs, lengths of runs, vals of runs. 
    """
    assert np.shape(x)[1] == 3
    x = np.transpose(x)
    #diffs = np.where(np.sum(np.abs(np.diff(x)),axis=1)!=0)
    diff_idx = np.hstack((np.zeros(1,dtype=np.int64),np.where(np.sum(np.abs(np.diff(x)),axis=0)!=0)[0]+1)) + zero_index   # returns list of indexes of new runs
    diff_idx_hi = np.roll(diff_idx,-1)
    diff_idx_hi[-1] = np.shape(x)[1]
    run_lengths = diff_idx_hi - diff_idx   # compute run lengths
    vals = x[:,diff_idx]  # compute values

    return diff_idx, run_lengths, vals


def parse_run(run_length, run_val):
    run_val = np.ravel(run_val)
    if run_length == 0:
        raise Exception()
    if run_length<256:
        if run_length == 1:
            #return [0x00, 0x01, run_val[0],run_val[1],run_val[2]]
            return [0x01, run_val[0],run_val[1],run_val[2]]
        else:
            return [np.uint8(run_length),run_val[0],run_val[1],run_val[2]]
    else:
        commands = []
        while run_length > 255:
            commands+=parse_run(0xff, run_val)
            run_length-=255
        commands+=parse_run(run_length, run_val)
        return commands
            

def rle_row_comp(row):
    run_idxs, run_lengths, vals = standard_rle_compression(row, 0)
    
    command_sequence = []
    for idx, run_idx in np.ndenumerate(run_idxs):
        command_sequence+=parse_run(run_lengths[idx], vals[:,idx])
    return command_sequence
    

def rle_comp_image(pattern):
    """
    :param (M, N) matrix of uint8 that represents a pattern
    :return 1D 24bit compressed bitmap
    """
    command_sequence = []
    for row_idx in np.arange(np.shape(pattern)[0]):
        command_sequence+=rle_row_comp(pattern[row_idx,:])
    command_sequence+=[0x00, 0x01]

    #print np.where(command_sequence == 0x01)
    return command_sequence, len(command_sequence)
    

def cleanup_zero_length_runs(run_lengths, array_to_clean):
    """
    :param run_lengths - 1d numpy array
    :param array to clean
    :return same as array to clean, but all elements in the same position where run_lengths == 0, are gone
    """
    bad_idx = np.where(run_lengths == 0)[0]
    return np.delete(array_to_clean, bad_idx)


def row_comparison(row0, row1):
    """
    :param 2 rows
    :return indexes of starts of runs where they are the same, the lengths of those runs lengths, then same thing, but where they are different
    """
    assert np.shape(row0)[0] == 3
    assert np.shape(row1)[0] == 3
    assert np.shape(row0) == np.shape(row1)
    x = np.sum(np.abs(row0-row1),axis=0) #0 if rows match, nonzero if they don't

    diff_idx = np.hstack((np.zeros(1,dtype=np.int64),np.where(np.diff(x)!=0)[0]+1))   # returns list of indexes of new runs
    diff_idx_hi = np.roll(diff_idx,-1)
    diff_idx_hi[-1] = np.size(x)

    run_lengths = diff_idx_hi - diff_idx   # compute run lengths
    is_rows_same = 1*(x[diff_idx] == 0)  # 1 if same, 0 if difft, later add 2 if make uncompressed
    run_indexes = np.arange(np.size(run_lengths))

    '''
    deal with the fact that uncompressed sequences can't have length 1
    first, check the last element of row. if its a run of length 1, then make it uncompressed, by cutting into the previous run
    1 difft from prev row, cut into the next run, delete that run if it is of length 1
    2 or more difft from prev row, let rle handle it
    '''
    if run_lengths[-1] == 1:
        diff_idx[-1] = diff_idx[-1] - 1
        is_rows_same[-1] = 2
        run_lengths[-1] = 2
        run_lengths[-2] = run_lengths[-2] - 1 #  Remember to implement check to never insert control sequences corresponding to zero run length runs

    short_run_indexes = run_indexes[run_lengths==1]
    short_runs_difft_from_row_indexes = short_run_indexes[is_rows_same[short_run_indexes] == 0]
    
    diff_idx = cleanup_zero_length_runs(run_lengths, diff_idx)
    run_lengths = cleanup_zero_length_runs(run_lengths, run_lengths)
    is_rows_same = cleanup_zero_length_runs(run_lengths, is_rows_same)
    
    for short_difft_run_index in short_runs_difft_from_row_indexes:
        is_rows_same[short_difft_run_index] = 2
        if short_difft_run_index != np.size(diff_idx)-1:
            if run_lengths[short_difft_run_index + 1] == 1:
                is_rows_same[short_difft_run_index + 1] = 2
        else:
            run_lengths[short_difft_run_index] = run_lengths[short_difft_run_index] - 1
            if short_difft_run_index != np.size(diff_idx)-1:
                run_lengths[short_difft_run_index + 1] = run_lengths[short_difft_run_index + 1] - 1 
                diff_idx[short_difft_run_index + 1] = diff_idx[short_difft_run_index] + 1
        
    # Concatenate runs of uncompressed bytes
    # find runs of 2's
    #uncompressed_run_idxs = np.where(is_rows_same == 2)[0] #indexes in the runs array
    #uncompressed_run_lengths = run_lengths[uncompressed_run_idxs]

    #print "uncompressed run idxs %s"%uncompressed_run_idxs
    #print "uncompressed run lengths %s"%uncompressed_run_lengths
    #print uncompressed_run_idxs[np.where(uncompressed_run_lengths == 1)[0]] # run indexes where the uncompressed run lengths are 1

    
    '''
    print "run indexes %s" %run_indexes
    print "short run indexes %s"%short_run_indexes
    print "short run difft from row %s" %short_runs_difft_from_row_indexes
    '''

    return diff_idx, run_lengths, is_rows_same


def erle_compress_row(row,prev_row = None):
    """
    Note: this is not finished
    :param row to be encoded
    :param previous row
    :return command sequences corresponding to the row encoding
    pick out regions that are the same as the previous row.
    run length encode the regions that are not 
    """
    if prev_row is not None:
        row_equal_idx, row_equal_run_lengths, row_difft_idx, row_difft_run_length = row_comparison(row, prev_row)
        erle_encoding_type = np.zeros_like(row_equal_idx)

        __dummy_vals = np.zeros_like(row_equal_idx)
        #Todo: the following could be parallelized, by rewriting the rle function:
        rle_idx = None; rle_run_length = None;
        for idx, row_idx in np.ndenumerate(row_difft_idx):
            temp_diff_idx, temp_run_length, temp_vals = standard_rle_compression(row[row_idx:row_idx+row_difft_run_length[idx]], row_idx)
            
        rle_encoding_type = np
    else:
        rle_idx, rle_run_lengths, rle_vals = standard_rle_compression(row)
    

compression_function_dict = {'none': no_comp_image, 'rle': rle_comp_image, 'erle': erle_comp_image}


def make_header(compressed_pattern_length,compression_type):
    """
    :return header for sending the images
    """
    #Note: taken directly from sniffer of the TI GUI
    header = [0x53, 0x70, 0x6C, 0x64, #Signature
              0x80, 0x07, 0x38, 0x04, #Width and height: 1920, 1080
              0x60, 0x11, 0x00, 0x00, #Number of bytes in encoded image_data
              0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, #reserved
              0x00, 0x00, 0x00, 0x00, # BG color BB, GG, RR, 00
              0x01, #0x01 reserved - manual says 0, sniffer says 1 
              0x02, #encoding 0 none, 1 rle, 2 erle
              0x01, #reserved 
              0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00] #reserved
        
    encoded_length_bytes_header = hh.int2reversed_hex_array(compressed_pattern_length, 4)
    compression_dictionary = {'none': 0, 'rle': 1, 'erle': 2}

    for i in range(4):
        header[8 + i] = encoded_length_bytes_header[i]
    header[25] = compression_dictionary[compression_type]

    return np.array(header, dtype=np.uint8).tolist()
 
'''Handy math functions'''


def circle_fun(X,Y):
    return 1*(mesh_radius(X, Y) < 8)


def circle_fun_2(X,Y):
    return 1*(mesh_radius(X, Y) <540)#300


def stripes(X,Y):
    return 1*(Y%40>1)


def stripes_2(X,Y):
    return 1*(Y%40>20)
    #return 1*(Y%40>20)


if __name__ == '__main__':
    #test_row =  np.array([0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff,0x00,0x00,0xff,0x00,0x00,0xff,0xff])
    #test_row2 = np.array([0xff,0xff,0xff,0xff,0x00,0x00,0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0xff])
    test_row =  np.array([0x00,0x00,0x00,0xff,0xff,0xff,0xff,0x00,0x00,0xff,0x00,0x00,0xff,0xff,0x00])
    test_row2 = np.array([0xff,0xff,0x00,0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff])
    #test_row = np.array([0xff,0xff,0x00,0xff,0xff,0x00,0x00,0xff,0x00])
    #test_row2 = np.array([0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00])
    test_row_pix = np.vstack((test_row,test_row,test_row))
    test_row_pix2 = np.vstack((test_row2,test_row2,test_row2))
    
    #print rle_row_comp(test_row_pix)

    #print standard_rle_compression(test_row_pix)
    #diff_idx, run_length, is_rows_same = row_comparison(test_row_pix, test_row_pix2)
    #print "diff_idx  %s"%diff_idx
    #print "run length %s"%run_length
    #print "compressio %s"%is_rows_same
    
    #creating the pattern
    settings = {'function':stripes,
                'resolution':(1920,1080),
                'compression':'rle',
                'zero_pixel':(50,10)}
    dmd_pattern = DMDPattern(**settings)
    dmd_pattern.compute_pattern()
    dmd_pattern.show_pattern()
    compressed = dmd_pattern.compress_pattern()

    #print "the starting size is is %s bytes"%
    print "the compressed size is %s bytes"%len(compressed)
    
    #compressing the pattern
    