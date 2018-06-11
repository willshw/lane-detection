#!/usr/bin/env python

import copy
import numpy as np
import pyproj as pj
from numpy.polynomial import polynomial as P

from scipy import stats
from sklearn.cluster import KMeans, DBSCAN

import pcl

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D

def get_xy(lat, lon):
    '''
    convert lat, lon to x, y
    '''
    # setup your projections
    crs_wgs = pj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_bng = pj.Proj(init='epsg:3857') # use a locally appropriate projected CRS

    # then cast your geographic coordinate pair to the projected system
    x, y = pj.transform(crs_wgs, crs_bng, lon, lat)
    
    return x, y

def get_latlon(x, y):
    '''
    convert x, y to lat, lon
    '''
    # setup your projections
    crs_wgs = pj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_bng = pj.Proj(init='epsg:3857') # use a locally appropriate projected CRS

    # then cast your geographic coordinate pair to the projected system
    lat, lon = pj.transform(crs_bng, crs_wgs, x, y)
    
    return lat, lon

def normalize(i):
    return i/np.linalg.norm(i)

def plot(data, num, step):
    '''
    use matplotlib to visualize points
    '''

    fig = plt.figure()

    ax = Axes3D(fig)
    ax.scatter(data[0:num:step,0], data[0:num:step,1], data[0:num:step,2], c=data[0:num:step,3])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def find_road_plane(points):

    # cloud = pcl.PointCloud()
    # cloud.from_array(xyz_data[:,0:3].astype('float32'))

    cloud = pcl.PointCloud_PointXYZI()
    cloud.from_array(xyz_data.astype('float32'))

    # fitler statistical outlier
    # fil_stat = cloud.make_statistical_outlier_filter()
    # fil_stat = cloud.make_statistical_outlier_filter()
    # fil_stat.set_mean_k(50)
    # fil_stat.set_std_dev_mul_thresh(1.0)
    # cloud_filtered = fil_stat.filter()

    # print "Statistical Inlier Number:", cloud_filtered.size

    # find normal plane
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(0.001)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.3)
    indices, model = seg.segment()

    print "Road Plane Model:", model

    cloud_plane = cloud.extract(indices, negative=False)

    # NG : const char* not str
    # cloud_plane.to_file('table_scene_mug_stereo_textured_plane.pcd')
    pcl.save(cloud_plane, 'road_plane.pcd')
    print "Road plane point cloud file road_plane.pcd saved."

    return cloud_plane.to_array(), np.array(indices)

def read_points(filename):
    file = open(filename, "r")
    txt_data = file.readlines()

    raw_data = np.zeros((len(txt_data), len(txt_data[0].split())))
    xyz_data = np.zeros((len(txt_data), len(txt_data[0].split())))
    # rgb = np.zeros((len(txt_data), 3))

    for idx, line in enumerate(txt_data):
        raw_data[idx,:] = line.split()
        xyz_data[idx,:] = copy.deepcopy(raw_data[idx,:])

        # (r, g, b) = colorsys.hsv_to_rgb(float(xyz_data[idx,-1]) / 255, 1.0, 1.0)
        # R, G, B = int(255 * r), int(255 * g), int(255 * b)
        # rgb[idx,:] = np.array([R, G, B])

    x, y = get_xy(xyz_data[:, 0], xyz_data[:, 1])
    xyz_data[:, 0] = np.array(x)
    xyz_data[:, 1] = np.array(y)


    xyz_data[:, 2] = xyz_data[:, 2] - np.min(xyz_data[:, 2])


    print "Finished reading raw data."

    return xyz_data

def plot_points(points, skip=100):
    points = points - np.min(points, axis=0)
    # plt.ion()
    norm = clr.Normalize(vmin=0,vmax=255)

    fig = plt.figure()
    ax = Axes3D(fig)

    max_range = np.array([points[:,0].max()-points[:,0].min(), points[:,1].max()-points[:,1].min(), points[:,2].max()-points[:,2].min()]).max() / 2.0
    mid_x = (points[:,0].max() + points[:,0].min()) * 0.5
    mid_y = (points[:,1].max() + points[:,1].min()) * 0.5
    mid_z = (points[:,2].max() + points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    num = points.shape[0]
    sc = ax.scatter(points[0:num:skip,0], points[0:num:skip,1], \
                    points[0:num:skip,2], c=points[0:num:skip,3], norm=norm, s=10)

    plt.colorbar(sc)
    plt.show()

def thresholding(points, threshold=30):
    return points[np.logical_and(points[:,3]>threshold, points[:,3]<256)]

def line_clustering(points):
    # 2d clustering based on line structure in x-y plane first
    fig2=plt.figure()
    ax2 = fig2.add_subplot(111)
    xy = points[:, 0:2]
    xy = xy - np.min(xy, axis=0)
    ax2.scatter(xy[:,0], xy[:,1])

    # initialization of m and c
    A = np.vstack([xy[:,0], np.ones(len(xy))]).T
    m, c = np.linalg.lstsq(A, xy[:,1])[0]
    c1 = c-20# + np.random.normal(0, 10)
    c2 = c-10
    c3 = c# + np.random.normal(0, 10)
    c4 = c+10
    c5 = c+20
    plt.plot(xy[:,0], m*xy[:,0] + c1, 'r')
    plt.plot(xy[:,0], m*xy[:,0] + c2, 'g')
    plt.plot(xy[:,0], m*xy[:,0] + c3, 'b')
    plt.plot(xy[:,0], m*xy[:,0] + c4, 'y')
    plt.plot(xy[:,0], m*xy[:,0] + c5, 'k')


    for i in range(10):
        # classify
        y1 = np.absolute(A.dot([[m],[c1]]) - xy[:,[1]])
        y2 = np.absolute(A.dot([[m],[c2]]) - xy[:,[1]])
        y3 = np.absolute(A.dot([[m],[c3]]) - xy[:,[1]])
        y4 = np.absolute(A.dot([[m],[c4]]) - xy[:,[1]])
        y5 = np.absolute(A.dot([[m],[c5]]) - xy[:,[1]])

        k1 = np.squeeze(np.all([y1<y2,y1<y3,y1<y4,y1<y5],axis=0))
        k2 = np.squeeze(np.all([y2<y1,y2<y3,y2<y4,y2<y5],axis=0))
        k3 = np.squeeze(np.all([y3<y1,y3<y2,y3<y4,y3<y5],axis=0))
        k4 = np.squeeze(np.all([y4<y1,y4<y2,y4<y3,y4<y5],axis=0))
        k5 = np.squeeze(np.all([y5<y1,y5<y2,y5<y3,y5<y4],axis=0))
        #k2 = np.squeeze(np.logical_and(np.logical_and(y2<y1, y2<y3),y2<10))
        # k3 = np.squeeze(np.logical_and(np.logical_and(y3<y1, y3<y2),y3<10))
        # k4 = np.squeeze(np.logical_and(np.logical_and(y4<y1, y3<y2),y3<10))
        # k5 = np.squeeze(np.logical_and(np.logical_and(y3<y1, y3<y2),y3<10))
        # update
        A1 = np.vstack([xy[k1,0], np.ones(len(xy[k1]))]).T
        A2 = np.vstack([xy[k2,0], np.ones(len(xy[k2]))]).T
        A3 = np.vstack([xy[k3,0], np.ones(len(xy[k3]))]).T
        A4 = np.vstack([xy[k4,0], np.ones(len(xy[k4]))]).T
        A5 = np.vstack([xy[k5,0], np.ones(len(xy[k5]))]).T

        m1, c1 = np.linalg.lstsq(A1, xy[k1,1])[0]
        m2, c2 = np.linalg.lstsq(A2, xy[k2,1])[0]
        m3, c3 = np.linalg.lstsq(A3, xy[k3,1])[0]
        m4, c4 = np.linalg.lstsq(A4, xy[k4,1])[0]
        m5, c5 = np.linalg.lstsq(A5, xy[k5,1])[0]

        m = (m1+m2+m3+m4+m5)/5.0
        # replot
        ax2.clear()
        plt.scatter(xy[:,0], xy[:,1])
        plt.plot(xy[:,0], m*xy[:,0] + c1, 'r')
        plt.plot(xy[:,0], m*xy[:,0] + c2, 'g')
        plt.plot(xy[:,0], m*xy[:,0] + c3, 'b')
        plt.plot(xy[:,0], m*xy[:,0] + c4, 'y')
        plt.plot(xy[:,0], m*xy[:,0] + c5, 'k')

    # 3d polyline fitting based on 2d clusters
    xyz = line[:, 0:3]
    xyz = xyz - np.min(xyz, axis=0)

    max_range = np.array([xyz[:,0].max()-xyz[:,0].min(), xyz[:,1].max()-xyz[:,1].min(), xyz[:,2].max()-xyz[:,2].min()]).max() / 2.0
    mid_x = (xyz[:,0].max() + xyz[:,0].min()) * 0.5
    mid_y = (xyz[:,1].max() + xyz[:,1].min()) * 0.5
    mid_z = (xyz[:,2].max() + xyz[:,2].min()) * 0.5

    # divide into K clusters
    K1 = xyz[k1,:]
    K2 = xyz[k2,:]
    K3 = xyz[k3,:]
    K4 = xyz[k4,:]
    K5 = xyz[k5,:]
    K=[K1,K2,K3,K4,K5]

    # polyline fitting
    from numpy.polynomial import polynomial as P
    xs=np.arange(50)

    # unparallel straight lines
    fig3=plt.figure()
    ax3 = Axes3D(fig3)
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    ax3.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    for i in range(len(K)):
        X = K[i][:,0]
        Y = K[i][:,1:3]
        p1=P.polyfit(X,Y,1)
        ax3.plot(xs, p1[0,0] + p1[1,0]*xs, p1[0,1] + p1[1,1]*xs, 'r')

    # parallel straight lines
    fig4=plt.figure()
    ax4 = Axes3D(fig4)
    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)
    ax4.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    pave = np.zeros((2,2))
    for i in range(len(K)):
        X = K[i][:,0]
        Y = K[i][:,1:3]
        pave= pave + P.polyfit(X,Y,1)
    pave = pave / len(K)
    for i in range(len(K)):
        X = K[i][:,0]
        Y = K[i][:,1:3]
        p1=P.polyfit(X,Y,1)
        ax4.plot(xs, p1[0,0] + pave[1,0]*xs, p1[0,1] + pave[1,1]*xs, 'y')

    # unparallel poly lines of degree 2
    fig5=plt.figure()
    ax5 = Axes3D(fig5)
    ax5.set_xlim(mid_x - max_range, mid_x + max_range)
    ax5.set_ylim(mid_y - max_range, mid_y + max_range)
    ax5.set_zlim(mid_z - max_range, mid_z + max_range)
    ax5.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    for i in range(len(K)):
        X = K[i][:,0]
        Y = K[i][:,1:3]
        p=P.polyfit(X,Y,2)
        ax5.plot(xs, p[0,0] + p[1,0]*xs + p[2,0]*xs**2, \
                p[0,1] + p[1,1]*xs + p[2,1]*xs**2)

    # parallel poly lines of degree 2
    fig6=plt.figure()
    ax6 = Axes3D(fig6)
    ax6.set_xlim(mid_x - max_range, mid_x + max_range)
    ax6.set_ylim(mid_y - max_range, mid_y + max_range)
    ax6.set_zlim(mid_z - max_range, mid_z + max_range)
    ax6.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    pave = np.zeros((3,2))
    for i in range(len(K)):
        X = K[i][:,0]
        Y = K[i][:,1:3]
        pave= pave + P.polyfit(X,Y,2)
    pave = pave / len(K)
    for i in range(len(K)):
        X = K[i][:,0]
        Y = K[i][:,1:3]
        p=P.polyfit(X,Y,2)
        ax6.plot(xs, p[0,0] + pave[1,0]*xs + pave[2,0]*xs**2, \
                p[0,1] + pave[1,1]*xs + pave[2,1]*xs**2)

    plt.show()

if __name__ == '__main__':

    xyz_data = read_points("final_project_point_cloud.fuse")

    # save xyz data to pcd file
    road_xyz = pcl.PointCloud_PointXYZI()
    road_xyz.from_array(xyz_data.astype('float32'))
    pcl.save(road_xyz, 'road_xyz.pcd')
    print "Road plane segment point cloud file road_xyz.pcd saved."

    # find road plane using pcl call
    road_plane, road_plane_idx = find_road_plane(xyz_data)
    road_plane_flatten = road_plane[:,0:2]

    # cluster road plane, and find the road segment
    db = DBSCAN(eps=0.5, min_samples=5).fit_predict(road_plane_flatten)

    largest_cluster_label = stats.mode(db).mode[0]
    largest_cluster_points_idx = np.array(np.where(db == largest_cluster_label)).ravel()

    road_plane_seg_idx = road_plane_idx[largest_cluster_points_idx]
    road_plane_seg = copy.deepcopy(xyz_data[road_plane_seg_idx, :])

    # save road segment point cloud to numpy and pcd file

    np.save('road_plane_seg.npy', road_plane_seg)
    print "Road plane segment point count:{}".format(road_plane_seg.shape)
    print "Road plane segment point cloud file road_plane_seg.npy saved."

    road_plane_seg_cloud = pcl.PointCloud_PointXYZI()
    road_plane_seg_cloud.from_array(road_plane_seg.astype('float32'))

    pcl.save(road_plane_seg_cloud, 'road_plane_seg.pcd')
    print "Road plane segment point cloud file road_plane_seg.pcd saved."

    # plot road plane seg
    # plot_points(road_plane_seg, skip=100)

    line = thresholding(road_plane_seg)

    # plot_points(line, skip=1)

    line_clustering(line)