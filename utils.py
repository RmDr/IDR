import sys, os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def size(ob):
    '''returns size of object in Mb.'''
    return sys.getsizeof(ob) / float(1048576) # 1048576 = 1024 * 1024


def mkdir(path):
    try:
        os.makedirs(path)
    except WindowsError:
        pass   
	
def draw_heatmap(m):
    plt.imshow(m, aspect='auto', cmap='hot')
    plt.xticks(()); plt.yticks(()); plt.show()
    
    
def draw3d(data, elev=20, azim=45, figsize=(8, 8), point_size=5, alpha=1):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*data.T, s=point_size, alpha=1, c='green')
    ax.view_init(elev=elev, azim=azim)
    plt.show()
    
def draw2d(data, figsize=(8, 8), point_size=5, alpha=1):
    fig = plt.figure(figsize = (8,8))
    plt.scatter(*data.T, s=point_size, alpha=1, c='green')
    plt.show()