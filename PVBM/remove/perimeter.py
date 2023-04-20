import numpy as np
from scipy.signal import convolve2d

def recursive(i_or,j_or,skeleton,i,j,all_important_index,l,d = 0):
    up = (i-1,j)
    down = (i+1,j)
    left = (i,j-1)
    right = (i,j+1)
    
    up_left = (i-1,j-1)
    up_right = (i-1,j+1)
    down_left = (i+1,j-1)
    down_right = (i+1,j+1)
    if up[0] >=0 and skeleton[up[0]][up[1]] ==1:
        skeleton[up[0]][up[1]] =0
        point = up
        d = d + 1
        if up in all_important_index:
            l.append(d)        
        else : 
            recursive(i_or,j_or,skeleton,up[0],up[1],all_important_index,l,d = d)
            
    if down[0] < len(skeleton) and skeleton[down[0]][down[1]] ==1:
        skeleton[down[0]][down[1]] =0
        point = down
        d = d + 1
        if down in all_important_index:
            l.append(d)
        else : 
            recursive(i_or,j_or,skeleton,down[0],down[1],all_important_index,l,d=d)
            
    if left[1] >= 0 and skeleton[left[0]][left[1]] ==1 :
        skeleton[left[0]][left[1]] =0
        point = left
        d = d + 1
        if left in all_important_index:
            l.append(d)

        else : 
            recursive(i_or,j_or,skeleton,left[0],left[1],all_important_index,l,d = d)
        
    if right[1] <len(skeleton[0]) and skeleton[right[0]][right[1]] ==1:
        skeleton[right[0]][right[1]] =0
        point = right
        d = d + 1
        if right in all_important_index:
            l.append(d)

        else : 
            recursive(i_or,j_or,skeleton,right[0],right[1],all_important_index,l,d=d)
            
    if up_left[0] >=0 and up_left[1] >=0 and skeleton[up_left[0]][up_left[1]] ==1:
        skeleton[up_left[0]][up_left[1]] =0
        point = up_left    
        d = d + 2**0.5
        if up_left in all_important_index:
            l.append(d)

        else : 
            recursive(i_or,j_or,skeleton,up_left[0],up_left[1],all_important_index,l,d = d)
                
    if up_right[0] >=0 and up_right[1] < len(skeleton) and skeleton[up_right[0]][up_right[1]] ==1 :
        skeleton[up_right[0]][up_right[1]] =0
        point = up_right
        d = d + 2**0.5
        if up_right in all_important_index:
            l.append(d)

        else : 
            recursive(i_or,j_or,skeleton,up_right[0],up_right[1],all_important_index,l,d=d)
    
    if down_left[0] < len(skeleton) and down_left[1] >=0 and skeleton[down_left[0]][down_left[1]] ==1:
        skeleton[down_left[0]][down_left[1]] =0
        d = d + 2**0.5
        point = down_left      
        if down_left in all_important_index:
            l.append(d)
        else : 
            recursive(i_or,j_or,skeleton,down_left[0],down_left[1],all_important_index,l,d = d)

    if down_right[0] < len(skeleton) and down_right[1] >=0 and skeleton[down_right[0]][down_right[1]] ==1:
        skeleton[down_right[0]][down_right[1]] =0
        d = d + 2**0.5
        point = down_right
        if down_right in all_important_index:
            l.append(d)
        else : 
            recursive(i_or,j_or,skeleton,down_right[0],down_right[1],all_important_index,l,d = d)

        
                
    
def connected_pixels(skeleton,all_important_index,verbose = 0):
    l = []
    for i,j in all_important_index :
        if skeleton[i][j] ==1 :
            skeleton[i][j] = 0
            recursive(i,j,skeleton,i,j,all_important_index,l)
    return l

from scipy.signal import convolve2d
filter_ = np.ones((3,3))
filter_[1,1] = 10
filter_


def compute_perimeter(img):
    skeleton = img
    tmp = convolve2d(img,filter_,mode = "same")
    endpoints = tmp == 11
    intersection = tmp >= 13
    final_eroded = endpoints + intersection
    origin_points = []
    for i in range(final_eroded.shape[0]):
        for j in range(final_eroded.shape[1]):
            if final_eroded[i,j] == True:
                origin_points.append((i,j))
    l = connected_pixels(skeleton,origin_points,verbose = 0)
    return np.sum(l)
    