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
from PVBM.helpers.branching2 import compute_tortuosity

from scipy.signal import convolve2d
filter_ = np.ones((3,3))
filter_[1,1] = 10
filter_

def preprocess(skeleton):
    tmp = convolve2d(skeleton,filter_,mode = "same")
    endpoints = tmp == 11
    intersection = tmp >= 13
    final_eroded = endpoints + intersection
    origin_points = []
    for i in range(final_eroded.shape[0]):
        for j in range(final_eroded.shape[1]):
            if final_eroded[i,j] == True:
                origin_points.append((i,j))
    return origin_points

def recursive(i,j,skeleton,origin_points,visited,dist,distance_dictionnary):
    up = (i-1,j)
    down = (i+1,j)
    left = (i,j-1)
    right = (i,j+1)
    
    up_left = (i-1,j-1)
    up_right = (i-1,j+1)
    down_left = (i+1,j-1)
    down_right = (i+1,j+1)
    
    if up[0] >=0 and up and visited[up] ==0:
        visited[up] = 1
        if skeleton[up[0]][up[1]] ==1:    
            distance_dictionnary[(up[0],up[1])] = max(distance_dictionnary[up[0],up[1]],dist +1)
            recursive(up[0],up[1],skeleton,origin_points,visited,dist+1,distance_dictionnary)
            
    if down[0] < len(skeleton) and visited[down] ==0:
        visited[down] = 1
        if skeleton[down[0]][down[1]] ==1:
            distance_dictionnary[(down[0],down[1])] = max(distance_dictionnary[down[0],down[1]],dist +1)
            recursive(down[0],down[1],skeleton,origin_points,visited,dist+1,distance_dictionnary)
    
    if left[1] >= 0 and visited[left] ==0:
        visited[left] = 1
        if skeleton[left[0]][left[1]] ==1: 
            distance_dictionnary[(left[0],left[1])] = max(distance_dictionnary[left[0],left[1]],dist +1)
            recursive(left[0],left[1],skeleton,origin_points,visited,dist+1,distance_dictionnary)
            
    if right[1] <len(skeleton[0]) and visited[right] ==0:
        visited[right] = 1
        if skeleton[right[0]][right[1]] ==1: 
            distance_dictionnary[(right[0],right[1])] = max(distance_dictionnary[right[0],right[1]],dist +1)
            recursive(right[0],right[1],skeleton,origin_points,visited,dist+1,distance_dictionnary)
                
    if up_left[0] >=0 and up_left[1] >=0 and visited[up_left] ==0:
        visited[up_left] = 1
        if skeleton[up_left[0]][up_left[1]] ==1: 
            distance_dictionnary[(up_left[0],up_left[1])] = max(distance_dictionnary[up_left[0],up_left[1]],dist +1)
            recursive(up_left[0],up_left[1],skeleton,origin_points,visited,dist+1,distance_dictionnary)
                     
    if up_right[0] >=0 and up_right[1] < len(skeleton) and visited[up_right] ==0:
        visited[up_right] = 1
        if skeleton[up_right[0]][up_right[1]] ==1: 
            distance_dictionnary[(up_right[0],up_right[1])] = max(distance_dictionnary[up_right[0],up_right[1]],dist +1)
            recursive(up_right[0],up_right[1],skeleton,origin_points,visited,dist+1,distance_dictionnary)
             
    if down_left[0] < len(skeleton) and down_left[1] >=0 and visited[down_left] ==0:
        visited[down_left] = 1
        if skeleton[down_left[0]][down_left[1]] ==1: 
            distance_dictionnary[(down_left[0],down_left[1])] = max(distance_dictionnary[down_left[0],down_left[1]],dist +1)
            recursive(down_left[0],down_left[1],skeleton,origin_points,visited,dist+1,distance_dictionnary)
                
    if down_right[0] < len(skeleton) and down_right[1] >=0 and visited[down_right] ==0:
        visited[down_right] = 1
        if skeleton[down_right[0]][down_right[1]] ==1: 
            distance_dictionnary[(down_right[0],down_right[1])] = max(distance_dictionnary[down_right[0],down_right[1]],dist +1)
            recursive(down_right[0],down_right[1],skeleton,origin_points,visited,dist+1,distance_dictionnary)
                

def distance(skeleton,origin_points):
    distance_dictionnary = np.zeros((skeleton.shape[0],skeleton.shape[1])) -1
    for i,j in origin_points:
            visited = np.zeros((skeleton.shape[0],skeleton.shape[1]))
            if visited[i,j] == 0:
                visited[i,j] = 1
            if skeleton[i][j] ==1 and (i,j) in origin_points :
                recursive(i,j,skeleton,origin_points,visited,0,distance_dictionnary)
    return distance_dictionnary

def compute_distances(skeleton,origin_points):
    distance_dictionnary = distance(skeleton,origin_points)
   
    return distance_dictionnary

def isdouble(key,v1,v2,dico_angle):
    for k in dico_angle.keys():
        if set(key) == set(k):
            return True
    return False

def isdouble2(key,v1,v2,last_dico_angle):
    for k in last_dico_angle.keys():
        if set(key) == set(k):
            return True
    return False

def is_in(a,dico):
    for element in dico.keys():
        if set(element) == a:
            return True
    return False
def crop(angle):
    if angle >1 :
        return 1
    if angle <-1:
        return -1
    return angle
def compute_angles_dictionary(skeleton):
    origin_points = preprocess(skeleton)
    distances_dico = compute_distances(skeleton,origin_points) ##plotable centroid
    centroid = distances_dico.copy()
    distance_dico_tmp = distances_dico.copy()
    distances_dico = {}
    for i in range(distance_dico_tmp.shape[0]):
        for j in range(distance_dico_tmp.shape[1]):
            if distance_dico_tmp[i,j] != -1 :
                distances_dico[i,j] = distance_dico_tmp[i,j]
    connection_dico = compute_tortuosity(skeleton)
    #return connection_dico
    dico_angle = {}
    for key,values in connection_dico.items():
        for v1 in values:
            for v2 in values:
                if v2 != v1 and not isdouble(key,v1,v2,dico_angle)  and not isdouble(key,v2,v1,dico_angle) : 
                    b = np.array(key) 
                    a = np.array(v1)
                    c = np.array(v2)
                    ba = a - b
                    bc = c - b
                    if np.isnan(ba).any() or np.isnan(bc).any() or np.isinf(ba).any() or np.isinf(bc).any() or np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
                        continue

                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.arccos(crop(cosine_angle))

                    if np.mean(np.abs(b-a)) > 5 and np.mean(np.abs(b-c))>5 :
                        dico_angle[(key,v1,v2)] = np.degrees(angle)
    final_connection_dico = {}
    for key in dico_angle.keys():
        get = final_connection_dico.get(key[0],[])
        for element in key[1:]:
            if element not in get:
                get.append(element)

        final_connection_dico[key[0]] = get

    last_connection_dico = {}  
    for key,value in final_connection_dico.items():
        if len(value) > 2:
            min_idx = 0
            min_point = value[0]
            min_dist = distances_dico.get(min_point,1444)
            for i in range(len(value)):
                point = value[i]
                dist = distances_dico.get(point,1444)
                if dist < min_dist:
                    min_idx = i
                    min_point = point
                    min_dist = dist
            final_tuple = []
            for val in value:
                if val != min_point:
                    final_tuple.append(val)
                last_connection_dico[key] = final_tuple
    last_dico_angle = {}
    for key,values in last_connection_dico.items():
        for v1 in values:
            for v2 in values:
                if v2 != v1 and not isdouble2(key,v1,v2,last_dico_angle)  and not isdouble2(key,v2,v1,last_dico_angle): 
                    b = np.array(key) 
                    a = np.array(v1)
                    c = np.array(v2)
                    ba = a - b
                    bc = c - b

                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.arccos(crop(cosine_angle))

                    if np.mean(np.abs(b-a)) > 5 and np.mean(np.abs(b-c))>5 :
                        last_dico_angle[(key,v1,v2)] = np.degrees(angle)
                        
    final_angle_dico = {}
    for key,value in last_dico_angle.items():
        if not is_in(set(key),final_angle_dico):
            final_angle_dico[key] = value

    return final_angle_dico,centroid
            
    
        
    
    
