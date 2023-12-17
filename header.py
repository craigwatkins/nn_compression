# class for the header object of a compressed image
from header_property import HeaderProperty


class Header:
    def __init__(self):
        # create default header properties
        self.height = HeaderProperty('height', 32)
        self.width = HeaderProperty('width', 32)
        self.compressed_length = HeaderProperty('length', 64)
        self.uncompressed_bit_size = HeaderProperty('uncompressed_bit_size', 8)
        self.clip_min = HeaderProperty('clip_min', 8)
        self.clip_max = HeaderProperty('clip_max', 8)
        self.header_string = None
        self.header_properties = [self.height, self.width, self.compressed_length, self.uncompressed_bit_size, self.clip_min, self.clip_max]
        self.total_length = sum([h_property.get_property_bit_size() for h_property in self.header_properties])

    def build_header(self, property_values):
        # assign values to header properties and build binary header string for compression
        header_string = ''
        for i, h_property in enumerate(self.header_properties):
            h_property.set_property_value(property_values[i])
            format_string = "{0:0" + str(h_property.get_property_bit_size()) + "b}"
            header_string += format_string.format(h_property.get_property_value())
        self.header_string = header_string
        return self.header_string

    def get_header(self):
        return self.header_string

    def get_total_length(self):
        return self.total_length

    def show_header(self):
        # print the header
        for h_property in self.header_properties:
            print(f"{h_property.get_property_name()}: {h_property.get_property_value()}")

    def decompress_header_values(self, header):
        # decompress header values from a binary string and return a list of values
        current_index = 0
        header_values = {}
        for h_property in self.header_properties:
            h_property_value = header[current_index:current_index + h_property.get_property_bit_size()]
            h_property_value = int(h_property_value, 2)
            h_property.set_property_value(h_property_value)
            header_values[h_property.get_property_name()] = h_property_value
            current_index += h_property.get_property_bit_size()
        return header_values


# run if this file is called directly
if __name__ == '__main__':
    # create a header object
    header = Header()
    # build a header
    header.build_header([641, 480, 2**16-1, 8, 0, 255])
    # print the header
    print(header.get_header())
    # decompress the header
    print(header.decompress_header_values(header.get_header()))
    header.show_header()
