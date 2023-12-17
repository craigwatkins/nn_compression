# -*- coding: utf-8 -*-
"""
Collection of functions to write a string of binary numbers into bytes.
This string may not be divisible by 8 so the final byte needs
to be padded out to avoid the containment of garbage data. The length of this
padding may not be known later so the length is prepended to the file in the
first three bits.

"""


def write_padded_bytes(bin_string, file_name):
    """
    Writes bytes to bin file and pads it.
    It always assumes padding is necessary and adds pad length info.
    Adds 3 bit number at beginning to indicate the length of the padding
    for the read function, then adds the padding on the final byte

    Parameters
    ----------
    bin_string : binary string
        data that needs to be saved
    file_name : string
        name of file where it will be saved
    Returns
    -------
    None.

    """
    pad_length = 8 - (len(bin_string)+3) % 8   # the amount of padding needed
    if pad_length != 8:
        pad_string = make_pad_string(pad_length)
    else:
        pad_string = ''
        pad_length = 0
    pad_length = "{0:03b}".format(pad_length)  # pad_length as 3 bit number
    string = ''.join([pad_length, bin_string, pad_string])
    with open(file_name, 'wb') as file:
        file.write(string_to_bytes(string))

import os

def read_padded_bytes(file_name):
    """
    Opens a binary file, removes padding, and returns a binary string.
    Checks if the file exists before attempting to open it.
    Handles errors during file operations.

    Parameters
    ----------
    file_name : str
        Name of the file to read

    Returns
    -------
    bin_string : str
        Bytes from file converted to string of 1's and 0's
        Returns None if an error occurs

    """
    # Check if file exists
    if not os.path.isfile(file_name):
        print(f"Error: File '{file_name}' does not exist.")
        return None

    try:
        bin_list = []
        with open(file_name, 'rb') as file:
            binaries = file.read()
        for byte in binaries:
            bin_list.append('{0:08b}'.format(byte))
        bin_string = strip_padding(bin_list)
        return bin_string
    except IOError as e:
        print(f"Error while reading the file: {e}")
        return None



def strip_padding(bin_list):
    """
    Parameters
    ----------
    bin_list : list of strings
        Each string is one byte of information

    Returns
    -------
    string
        A single binary string

    """
    padding_size = int( str(bin_list[0][0:3]),2 )
    bin_list[0] = bin_list[0][3:]
    bin_list[-1] = bin_list[-1][:len(bin_list[-1])-padding_size]
    return ''.join(bin_list)


def make_pad_string(length):
    """
    Parameters
    ----------
    length : integer
        how long the pad string needs to be

    Returns
    -------
    pad_string : binary string
        string of 0's that will pad out the last byte of a file
    """
    i=0
    pad_string =''
    while i<length:
        pad_string = pad_string+'0'
        i +=1
    return pad_string


def string_to_bytes(data):
    """
    Parameters
    ----------
    data : binary string
        string of 1's and 0's to be converted into bytes

    Returns
    -------
    bytearray object
        an array of bytes to be saved

    """
    byte_obj = bytearray()
    for i in range(0, len(data), 8):
        byte_obj.append(int(data[i:i+8], 2))
    return bytes(byte_obj)


def test():
    """
    Test code

    """
    string = "101010101010011111111111111"
    write_padded_bytes(string, 'testsavePadded.bin')
    bin_string = read_padded_bytes('testsavePadded.bin')
    if bin_string == string:
        print("success")
    else:
        print(bin_string[0:10])


def main():
    """
    Runs if this file is run on its own.

    Returns
    -------
    None.

    """
    test()


if __name__ == "__main__":

    main()