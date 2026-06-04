import pandas as pd 
import numpy as np
import random
import heapq
import gzip
from collections import Counter

def random_portfolio():
    # Uniformly distribute 4 independent cut points in (0, 1)
    cuts = [random.random() for _ in range(4)]
    
    # Sort the points to define sequential segment boundaries
    cuts.sort()
    
    # Calculate lengths: a, b, c, d, e
    a = cuts[0]
    b = cuts[1] - cuts[0]
    c = cuts[2] - cuts[1]
    d = cuts[3] - cuts[2]
    e = 1.0 - cuts[3]
    
    return [a, b, c, d, e]



def huffman_complexity(equation_str: str) -> int:
    """Returns the size of the equation string in bits using Huffman coding."""
    if not equation_str:
        return 0
        
    freqs = Counter(equation_str)
    if len(freqs) == 1:
        return len(equation_str) # 1 bit per identical character
        
    # Create a priority queue: [frequency, [character, bit_prefix]]
    heap = [[weight, [symbol, ""]] for symbol, weight in freqs.items()]
    heapq.heapify(heap)
    
    # Build the Huffman tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
    # Calculate total bit length from the encoded dictionary
    huff_dict = {pair[0]: len(pair[1]) for pair in heap[0][1:]}
    return sum(freqs[char] * huff_dict[char] for char in equation_str)

def gzip_complexity(equation_str: str) -> int:
    """Returns the size of the compressed equation string in bytes."""
    return len(gzip.compress(equation_str.encode('utf-8')))