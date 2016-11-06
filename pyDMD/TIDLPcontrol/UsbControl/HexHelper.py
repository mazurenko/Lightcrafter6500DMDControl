'''
Created on Mar 3, 2016

@author: anton
'''
import numpy as np

def int2reversed_hex_array(number, n_bytes = 2):
    """
    :param positive integer
    :return the integer, as a n_bytes long array of uint8's, starting from lsb
    """
    hex_array = []
    for n in range(n_bytes):
        hex_array.append((number >> n*8) & 0xff)
    return hex_array

def erle_n_converter(n):
    """
    :param n
    :return list of either 1 or 2 bytes

    If n is < 128 then encode it with 1 byte.
    If n is >= 128 then encode it with 2 byte in the following manner.
    Byte0 = ( n & 0x7F ) | 0x80
    Byte1 = ( n >> 7)
    Example: Number 0x1234 is encoded as 0xB4, 0x24
    """
    if n<128:
        out = np.array([n])
    else:
        out = np.array([np.bitwise_or(np.bitwise_and(n, 0x7F), 0x80),
                        np.right_shift(n,7)])
    return out

def inverse_erle_n_converter(n):
    """
    :param n - list of bytes,
    :return list of either 1 or 2 bytes

    If n is < 128 then encode it with 1 byte.
    If n is >= 128 then encode it with 2 byte in the following manner.
    Byte0 = ( n & 0x7F ) | 0x80
    Byte1 = ( n >> 7)
    Example: Number 0x1234 is encoded as 0xB4, 0x24
    """
    n_length = len(n)
    assert n_length <= 2 and n_length > 0
    if n_length == 2:
        out = np.bitwise_or(np.bitwise_and(n[0],0x7F),
                      np.left_shift(n[1],7))
    else:
        out = n[0]
    return out



if __name__ == '__main__':
    n = 1920
    print erle_n_converter(n)
    print inverse_erle_n_converter([0x80, 0x0F])
    print inverse_erle_n_converter([0xC9, 0x07])
    '''
    arr = int2reversed_hex_array(4448, n_bytes = 4)
    for x in arr:
        print format(x, '#04x')
    '''
    