import PIL
import numpy as np
from skimage.morphology import skeletonize
#import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from skimage.draw import line_aa
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening,disk)
#from tqdm.notebook import tqdm
from scipy.signal import convolve2d
filter_ = np.ones((3,3))
filter_[1,1] = 10
filter_

max_size = 20

                
def recursive(i_or,j_or,skeleton,i,j,visited,all_important_index,connected,dist,plot = True,zoom = True,size = 10,verbose = 0):
    if verbose:
        print('current :', i,j)
    tmp = np.zeros_like(skeleton)
    up = (i-1,j)
    down = (i+1,j)
    left = (i,j-1)
    right = (i,j+1)
    
    up_left = (i-1,j-1)
    up_right = (i-1,j+1)
    down_left = (i+1,j-1)
    down_right = (i+1,j+1)
    if up[0] >=0 and visited[up] == 0:
        visited[up] = 1
        if skeleton[up[0]][up[1]] ==1:    
            if  dist == max_size:
                connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [up]
                
            else : 
                recursive(i_or,j_or,skeleton,up[0],up[1],visited,all_important_index,connected,dist +1 , plot = plot)
                
    if down[0] < len(skeleton) and visited[down] == 0:
        visited[down] = 1
        if skeleton[down[0]][down[1]] ==1:
            if  dist == max_size:
                connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [down]
                
            else : 
                recursive(i_or,j_or,skeleton,down[0],down[1],visited,all_important_index,connected,dist +1, plot = plot)
                
    if left[1] >= 0 and visited[left] == 0:
        visited[left] = 1
        if skeleton[left[0]][left[1]] ==1: 
            if  dist == max_size :
                connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [left]
                
            else : 
                recursive(i_or,j_or,skeleton,left[0],left[1],visited,all_important_index,connected,dist+1, plot = plot)
        
    if right[1] <len(skeleton[0]) and visited[right] == 0:
        visited[right] = 1
        if skeleton[right[0]][right[1]] ==1:
            if right in all_important_index or dist == max_size:
                connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [right]
                
            else : 
                recursive(i_or,j_or,skeleton,right[0],right[1],visited,all_important_index,connected,dist+1, plot = plot)
                
    if up_left[0] >=0 and up_left[1] >=0 and visited[up_left] == 0:
        visited[up_left] = 1
        if skeleton[up_left[0]][up_left[1]] ==1  :
            if  dist == max_size:
                connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [up_left]
                
            else : 
                recursive(i_or,j_or,skeleton,up_left[0],up_left[1],visited,all_important_index,connected,dist +1, plot = plot)
                
    if up_right[0] >=0 and up_right[1] < len(skeleton) and visited[up_right] ==0:
        visited[up_right] = 1
        if skeleton[up_right[0]][up_right[1]] ==1:         
            if dist == max_size:
                connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [up_right]
                
            else : 
                recursive(i_or,j_or,skeleton,up_right[0],up_right[1],visited,all_important_index,connected,dist+1, plot = plot)
    
    if down_left[0] < len(skeleton) and down_left[1] >=0 and visited[down_left] ==0:
        visited[down_left] = 1
        if skeleton[down_left[0]][down_left[1]] ==1:      
            if down_left in all_important_index or dist == max_size:
                connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [down_left]
                
            else : 
                recursive(i_or,j_or,skeleton,down_left[0],down_left[1],visited,all_important_index,connected,dist+1 , plot = plot)
                
    if down_right[0] < len(skeleton) and down_right[1] >=0 and visited[down_right] ==0:
        visited[down_right] = 1
        if skeleton[down_right[0]][down_right[1]] ==1:            
            if dist == max_size:
                connected[(i_or,j_or)] = connected.get((i_or,j_or),[]) + [down_right]
                
            else : 
                recursive(i_or,j_or,skeleton,down_right[0],down_right[1],visited,all_important_index,connected,dist+1, plot = plot)
                
    
def connected_pixels(skeleton,all_important_index,verbose = 0):
    connected = {}
    for i,j in all_important_index:
            visited = np.zeros((skeleton.shape[0],skeleton.shape[1]))
            if verbose :
                print('starting point : ',i,j)
            if skeleton[i][j] ==1 and (i,j) in all_important_index :
                recursive(i,j,skeleton,i,j,visited,all_important_index,connected,0,zoom = False,plot = False,verbose = verbose)
    return connected

def compute_tortuosity(skeleton):
    tmp = convolve2d(skeleton,filter_,mode = "same")
    endpoints = tmp == 11
    intersection = tmp >= 13
    final_eroded = intersection
    origin_points = []
    for i in range(final_eroded.shape[0]):
        for j in range(final_eroded.shape[1]):
            if final_eroded[i,j] == True:
                origin_points.append((i,j))
    
    connection_dico = connected_pixels(skeleton,origin_points,verbose = 0)
    return connection_dico
    
    
