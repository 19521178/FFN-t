import numpy as np 
import math
from itertools import islice
import cv2 as cv


def expand_element_dict(dict, key, value):
    if (key not in dict.keys()):
        dict[key] = [value]
    else:
        if value not in dict[key]:
            dict[key].append(value)
            
            
def make_chunks(data, SIZE):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]
        