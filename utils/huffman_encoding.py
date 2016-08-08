__author__ = 'lqrz'
'''
Taken from: https://rosettacode.org/wiki/Huffman_coding#Python
'''
from heapq import heappush, heappop, heapify
from collections import Counter
import numpy as np


class Huffman_encoding(object):

    def __init__(self, items):
        self.frequencies = Counter(items)

    def encode(self):
        """Huffman encode the given dict mapping symbols to weights"""
        heap = [[wt, [sym, ""]] for sym, wt in self.frequencies.items()]
        heapify(heap)
        while len(heap) > 1:
            lo = heappop(heap)
            hi = heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        sorted_mapping = sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
        converted_mapping = dict()
        for item, enc in sorted_mapping:
            converted_mapping[item] = np.array(map(int, enc))

        return converted_mapping


if __name__ == '__main__':
    txt = "this is an example for huffman encoding"
    encoder = Huffman_encoding(txt)
    huff_encoding = encoder.encode()