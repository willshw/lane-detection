import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

xyzi = np.load("road_plane_seg.npy")
xyz_data = xyzi - np.min(xyzi, axis=0)
plt.ion()
norm=mpl.colors.Normalize(vmin=0,vmax=255)


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
num = xyz_data.shape[0]
skip=100
sc = ax.scatter(xyz_data[0:num:skip,0], xyz_data[0:num:skip,1], \
                xyz_data[0:num:skip,2], c=xyz_data[0:num:skip,3], norm=norm)
plt.colorbar(sc)
plt.show()


#filter out low intensity points
line = xyz_data[np.logical_and(xyz_data[:,3]>30, xyz_data[:,3]<250)]
num = line.shape[0]
skip=1
fig1 = plt.figure()
ax1 = Axes3D(fig1)
sc1 = ax1.scatter(line[0:num:skip,0], line[0:num:skip,1], \
                 line[0:num:skip,2], c=line[0:num:skip,3], norm=norm)
plt.colorbar(sc1)
plt.show()


# 2d clustering based on line structure in x-y plane first
fig2=plt.figure()
ax2 = fig2.add_subplot(111)
xy = line[:, 0:2]
xy = xy - np.min(xy, axis=0)
ax2.scatter(xy[:,0], xy[:,1])
plt.show()
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
fig3=plt.figure()
ax3 = Axes3D(fig3)
xyz = line[:, 0:3]
xyz = xyz - np.min(xyz, axis=0)
ax3.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
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
for i in range(len(K)):
    X = K[i][:,0]
    Y = K[i][:,1:3]
    p1=P.polyfit(X,Y,1)
    ax3.plot(xs, p1[0,0] + p1[1,0]*xs, p1[0,1] + p1[1,1]*xs, 'r')

# parallel straight lines
fig4=plt.figure()
ax4 = Axes3D(fig4)
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
