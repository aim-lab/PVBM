import numpy as np
from scipy.signal import convolve2d

from queue import PriorityQueue

def iterative(i_or, j_or, skeleton, all_important_index, visited, connected):
  pq = PriorityQueue()
  pq.put((0, i_or, j_or, i_or, j_or,0))
  priotities = [0,1,2,3,4,5,6,7]
  distances = [1, 1, 1, 1, 2**0.5, 2**0.5, 2**0.5, 2**0.5]
  while not pq.empty():
    _, i_or, j_or, i, j,d = pq.get()
    directions = [(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]
    for direction, distance,priority in zip(directions,distances,priotities):
      x,y = direction
      if x >= 0 and x < skeleton.shape[0] and y >= 0 and y < skeleton.shape[1] and visited[direction] == 0:
        if direction not in all_important_index:
          visited[direction] = 1
        if skeleton[x][y] == 1:
          point = direction
          if direction in all_important_index:
            connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [(direction,d + distance)]
          else : 
            pq.put((priority, i_or, j_or, x, y,d + distance ))




    
def connected_pixels(skeleton, all_important_index):
    connected = {}
    visited = np.zeros_like(skeleton)
    for i, j in all_important_index:
        if skeleton[i][j] == 1 and not visited[i][j]:
            iterative(i, j, skeleton, all_important_index,visited,connected)
            #recursive(i, j, skeleton, i, j, visited, all_important_index, connected)
    return connected

from scipy.signal import convolve2d
filter_ = np.ones((3,3))
filter_[1,1] = 10
filter_



def compute_tortuosity(skeleton):
    tmp = convolve2d(skeleton, filter_, mode="same")
    endpoints = tmp == 11
    intersection = tmp >= 13
    particular = endpoints + intersection
    origin_points = [(i, j) for i in range(particular.shape[0]) for j in range(particular.shape[1]) if particular[i, j]]
    connection_dico = connected_pixels(skeleton, origin_points)
    tor = []
    l = []
    for key, value in connection_dico.items():
        x, y = key
        for p, d in value:
            l.append(d)
            if d > 10:
                x2, y2 = p
                real_d = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
                tor.append(d / real_d)
    return np.median(tor), np.sum(l), tor, l, connection_dico


    
