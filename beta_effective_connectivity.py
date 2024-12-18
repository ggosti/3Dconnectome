import numpy as np
import pyvista as pv
import pandas as pd
import time

import matplotlib.pyplot as plt

colors = plt.cm.seismic(np.linspace(0,1,201))

def drawEdgePC(pl,nodes_pc,u,v,w):
        p0, p1 = nodes_pc[u], nodes_pc[v]
        print('p0',p0)
        print('p1',p1)
        dis = np.sqrt( np.sum((p0-p1)**2) )
        #print(dis)
        t = np.arange(0,dis,0.01) #t = np.arange(0,dis,0.1) /dis #t = np.linspace(0,1,10)
        x = p0[0] + (p1[0]-p0[0])*(t/dis)
        y = p0[1] + (p1[1]-p0[1])*(t/dis)
        #if w>0:
        z = p0[2] + (p1[2]-p0[2])*(t/dis) #0.4*(w)*(1. - (2*t - 1.)**2)
        x = np.append(x,[p1[0]])
        y = np.append(y,[p1[1]])
        z = np.append(z,[p1[2]])
        points = np.column_stack((x, y, z))
        #print('point start', points[0,:])
        #print('point end', points[-1,:])
        pointsSplice = pv.Spline(points)
        # Create a tube around the spline with a specified radius
        tube = pointsSplice.tube(radius=0.01)  # Adjust the tube radius here
        # Add the tube to the plot
        print('w',u,v,w,100+int(100*w))
        pl.add_mesh(tube, color=colors[100+int(100*w)])#, color='blue', label='Edge')
        ## Calculate the direction vector for the arrow
        vec = (points[-2,:] - points[-3,:])
        direction = 0.001*(vec)/np.sum(vec*vec)
        ## Create an arrow at the end point, pointing along the direction
        arrow = pv.Arrow(start=points[-1,:]-direction, direction=direction, tip_length=0.6, tip_radius=0.2,scale='auto')  # Adjust scale for arrow size
        # Add the arrow to the plot
        pl.add_mesh(arrow, color=colors[100+int(100*w)],show_scalar_bar=True)


# # Parcellizzazione
# https://www.sciencedirect.com/science/article/pii/S2211124720314601?via%3Dihub
df = pd.read_csv('centroids.csv',header=[0,1])
columns = ['Seed Label','Network','x','y','z']
df.columns= ['Seed Label','Network','x','y','z']
print(df.head())
X = df['x'].values
Y = df['y'].values
Z = df['z'].values
N=len(X)
df['network_number'] = pd.factorize(df['Network'])[0]
point_cloud = np.float64([X,Y,Z]).T
point_cloud = point_cloud/100. #point_cloud.max()
#point_cloud[:100,:]


net = pd.read_csv('JijAve_betaNorm.csv',header=[0],index_col=0).values
absMax = np.max(np.abs(net))

plt.figure()
plt.imshow(net,cmap='seismic',vmin=-absMax,vmax=absMax)


# PyVista Visualization
pdata = pv.PolyData(point_cloud)
pdata['orig_sphere'] = np.float64(df['network_number'])#np.random.random(N)
sphere = pv.Sphere(radius=0.02)

# Apply glyphs to the point cloud
pc = pdata.glyph(scale=False, geom=sphere, orient=False)

# Plotting
pl = pv.Plotter()
_ = pl.add_mesh(
    pc,
    cmap='tab10',
    smooth_shading=True,
    show_scalar_bar=True,
)

_ = pl.show_grid()

nodes = point_cloud.T #np.array([pos[v] for v in G]).T
nodes_pc = np.vstack([nodes,np.zeros(N)]).T



for i in range(10):
    for j in range(N):
        w = net[i,j]
        if np.abs(w) > 0.2:
            print('w',w)
            drawEdgePC(pl,nodes_pc,j,i,w)
    print('inedges_node_{:03n}'.format(i) +'.glb')
    pl.export_gltf('inedges_node_{:03n}'.format(i) +'.glb',inline_data=True)

pl.show()

#pl.export_gltf('brainDelta'+ str(1/lambdas[indx] )+'.glb',inline_data=True)  
#print('brainDelta'+ str(1/lambdas[indx] )+'.glb')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X,Y,Z,c=df['network_number'],cmap='tab10')
plt.show()
