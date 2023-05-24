import numpy as np
from scipy.signal import convolve2d

from queue import PriorityQueue


def iterative(i,j, skeleton, distances_matrix):
  pq = PriorityQueue()
  pq.put((0, i, j))
  priotities = [0,1,2,3,4,5,6,7]
  distances = [1, 1, 1, 1, 2**0.5, 2**0.5, 2**0.5, 2**0.5]
  while not pq.empty():
    _, i, j = pq.get()
    directions = [(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]
    for direction, distance,priority in zip(directions,distances,priotities):
      x,y = direction
      if x >= 0 and x < skeleton.shape[0] and y >= 0 and y < skeleton.shape[1] and skeleton[direction] == 1:
        distances_matrix[direction] = distance
        skeleton[direction] = 0
        pq.put((priority, direction[0],direction[1]))
        



def extract_subgraphs(skeleton):
    distances = np.zeros((skeleton.shape[0],skeleton.shape[1]),dtype = float)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if distances[i,j] == 0 and skeleton[i,j] == 1:
                iterative(i,j,skeleton,distances)
    return distances


def compute_perimeter_(skeleton):
    distances = extract_subgraphs(skeleton)
    return np.sum(distances),distances
