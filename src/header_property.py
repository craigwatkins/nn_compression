# class for a property of the header file (height, width, length, etc.)

class HeaderProperty:
    def __init__(self, name, bit_size, value=None):
        self.name = name
        self.bit_size = bit_size
        self.value = value

    def get_property(self, header):
        return int(header[0:self.bit_size], 2)

    def extract_property(self, property_bits):
        # translates the property from the header bits
        return "{0:0{1}b}".format(property_bits, self.bit_size)

    def get_property_bit_size(self):
        return self.bit_size

    def get_property_name(self):
        return self.name

    def get_property_value(self):
        return self.value

    def set_property_value(self, value):
        self.value = value
