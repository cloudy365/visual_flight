
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import sys


def show3d_flight(path_flight, stride=1, llcrnrlon=115, llcrnrlat=12, urcrnrlon=125, urcrnrlat=20):
    """Show 3D interactive plot of flight path.
    
    This script provides an intuitive 3D interactive map of the flight path.
    This script is based on Basemap, so that it is a little bit lag with large number of points.
    The first time in the colorbar represents the first time stamp in the data.
    The rest time-stamps in colorbar correspond to the time of first-time + n*3600 points.
    
    Arguments:
        path_flight {[type]} -- Path of the flight xlsx data file.
    
    Keyword Arguments:
        stride {int} -- stride for plotting points (default: {1})
        llcrnrlon {float} -- lower left corner longitude (default: {115})
        llcrnrlat {float} -- lower left corner latitude (default: {12})
        urcrnrlon {float} -- upper right corner longitude (default: {125})
        urcrnrlat {float} -- upper right corner latitude (default: {20})
    """
    stride = int(stride)

    #load data
    data = pd.read_excel(path_flight) 
    X = pd.DataFrame(data, columns= ['Longitude']).values.reshape(-1)[::stride]
    Y = pd.DataFrame(data, columns= ['Latitude']).values.reshape(-1)[::stride]
    Z = pd.DataFrame(data, columns= ['GPS_Alt']).values.reshape(-1)[::stride]
    T = pd.DataFrame(data, columns= ['Time']).values.reshape(-1)[::stride]

    #total points number
    num_points = len(X)

    #initialize figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax._axis3don = True

    #plot scatters
    cm = plt.cm.get_cmap('jet')
    sc = ax.scatter(X, Y, Z,
               marker='.', c=range(num_points), s=10, cmap=cm)

    #initialize basemap
    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, resolution='h')
    ax.add_collection3d(m.drawcoastlines(linewidth=0.25, ))
    ax.add_collection3d(m.drawcountries(linewidth=0.35))

    #set meridian and parallel gridlines
    lon_step = 2.
    lat_step = 2.
    meridians = np.arange(115, 125 + lon_step, lon_step)
    parallels = np.arange(12, 20 + lat_step, lat_step)
    ax.set_yticks(parallels)
    ax.set_yticklabels(parallels)
    ax.set_xticks(meridians)
    ax.set_xticklabels(meridians)
    ax.set_zlim(0., 12000.)

    #set axis labels
    ax.set_xlabel('Longitude (degree)')
    ax.set_ylabel('Latitude (degree)')
    ax.set_zlabel('Height (m)')

    #set colorbar
    cbar = fig.colorbar(sc, ticks=range(0, num_points, 3600/stride))
    cbar.ax.set_yticklabels([i[:-4] for i in T[::3600/stride]])

    #set title
    # plt.title(path_flight.split('.')[-2])
    plt.show()


if __name__ == '__main__':
    path_flight = sys.argv[1]
    stride = int(sys.argv[2])

    show3d_flight(path_flight=path_flight, stride=stride, llcrnrlon=115, llcrnrlat=12, urcrnrlon=125, urcrnrlat=20)