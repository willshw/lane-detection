# lane-detection

Authors: [William Wang](https://github.com/willshw), [Felix Wang](https://yanweiw.github.io)

### Problem Statement

Find equations for lane markings in point cloud data.

![](images/road_xyz_1.png)

### Data Specification

final_project_point_cloud.fuse :

	Point cloud data.
	Data format:
		[latitude] [longitude] [altitude] [intensity]

Notice lane markings are reflexive and thus corresponding points will have higher intensity.

### Usage

1. To get pcl_viewer: sudo apt-get install pcl-tool
2. To open and display .pcd file: pcl_viewer filename.pcd
    - To turn on intensity, press 5

Package Used:
1. PCL
2. pyproj
3. numpy
4. scipy
5. matplotlib
6. sklearn

How-To run program:

1. Put program in the same data folder as **final_project_point_cloud.fuse**
2. python lane-detection.py

### Method

Please refer to the [slides](https://docs.google.com/presentation/d/13mMAI7IvOBTep5CeOmJUhD1hfpUZbfGIj1Kam1IZmLU/edit?usp=sharing) for details. Feel free to contact authors if you have further questions.
