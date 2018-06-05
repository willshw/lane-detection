import copy
import numpy as np
import pyproj as pj

from scipy import stats
from sklearn.cluster import KMeans, DBSCAN

import pcl
# import pcl.pcl_visualization

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_xy(lat, lon):
    '''
    convert lat, lon to x, y
    '''
    # setup your projections
    crs_wgs = pj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_bng = pj.Proj(init='epsg:27700') # use a locally appropriate projected CRS

    # then cast your geographic coordinate pair to the projected system
    x, y = pj.transform(crs_wgs, crs_bng, lon, lat)
    
    return x, y

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

if __name__ == '__main__':

    xyz_data = read_points("final_project_point_cloud.fuse")
    road_plane, road_plane_idx = find_road_plane(xyz_data)

    road_plane_flatten = road_plane[:,0:2]

    db = DBSCAN(eps=0.3, min_samples=10).fit_predict(road_plane_flatten)
    
    largest_cluster_label = stats.mode(db).mode[0]
    largest_cluster_points_idx = np.array(np.where(db == largest_cluster_label)).ravel()

    road_plane_seg_idx = road_plane_idx[largest_cluster_points_idx]
    road_plane_seg = copy.deepcopy(road_plane[road_plane_seg_idx, :])

    road_plane_seg_cloud = pcl.PointCloud_PointXYZI()
    road_plane_seg_cloud.from_array(road_plane_seg.astype('float32'))

    pcl.save(road_plane_seg_cloud, 'road_plane_seg.pcd')
    print "Road plane segment point cloud file road_plane_seg.pcd saved."