import copy
import numpy as np
import pyproj as pj
import pcl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_xy(lat, lon):
    
    # setup your projections
    crs_wgs = pj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_bng = pj.Proj(init='epsg:27700') # use a locally appropriate projected CRS

    # then cast your geographic coordinate pair to the projected system
    x, y = pj.transform(crs_wgs, crs_bng, lon, lat)
    
    return x, y

def normalize(i):
    return i/np.linalg.norm(i)

def plot(data, num, step):

    fig = plt.figure()

    ax = Axes3D(fig)
    ax.scatter(data[0:num:step,0], data[0:num:step,1], data[0:num:step,2], c=data[0:num:step,3])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

file = open("final_project_point_cloud.fuse", "r")
txt_data = file.readlines()

raw_data = np.zeros((len(txt_data), len(txt_data[0].split())))
xyz_data = np.zeros((len(txt_data), len(txt_data[0].split())))

for idx, line in enumerate(txt_data):
    raw_data[idx,:] = line.split()
    xyz_data[idx,:] = copy.deepcopy(raw_data[idx,:])
    
x, y = get_xy(xyz_data[:, 0], xyz_data[:, 1])
xyz_data[:, 0] = np.array(x)
xyz_data[:, 1] = np.array(y)

# print np.amax(xyz_data, axis=0)

plot(xyz_data, 430000, 50)