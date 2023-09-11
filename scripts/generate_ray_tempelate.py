import argparse
import numpy as np
from sklearn.preprocessing import normalize
import os
import sys

parser =  argparse.ArgumentParser()
parser.add_argument("--h", 
                    type=int, 
                    help="Template height")
parser.add_argument("--w", type=int,  
                    help="Template width")
parser.add_argument("--o",  
                    type=str,  
                    help="Output file name without extension")
parser.add_argument("--normType",  
                    type=str,   
                    choices=['packnet', 'jdriscoll319'],  
                    help="Whether to normalize like Packnet did (z=1) or like jdriscoll319 did (norm l2 = 1) (cf github issue 115)")
parser.add_argument("--crop",  
                    type=int,  
                    help="Optional. Crop template after initialization")
args = parser.parse_args()

w = args.w
h = args.h

fx = cx = w/2
fy = cy = h/2

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]])
Kinv = np.linalg.inv(K)

p = []
for v in range(h):                  #row
    for u in range(w):              #col
        p.append(np.array([u,v,1]))

p = np.stack(p)         #p.shape = (w*h, 3)
Q = Kinv @ p.T          #Q.shape = (3, w*h)

if args.normType == 'jdriscoll319':
    Q_norm = normalize(Q, axis=0)   #Q.shape = (3, w*h) 
elif args.normType == 'packnet':
    Q_norm = Q / Q[2, :]
else:
    sys.exit('Wrong normalization type')

Q_ray_surface = np.zeros((1,3,h,w))
for v in range(w*h):
    idx = p[v,:]
    Qw = idx[0]
    Qh = idx[1]
    ray = Q_norm[:, v]
    Q_ray_surface[0, :, Qh, Qw] = ray

if args.crop:
    Q_ray_surface = Q_ray_surface[:,:,0:args.crop,:]

Q_ray_surface = Q_ray_surface.astype('float32')

fname = args.o
with open('/home/seok436/packnet-sfm-master_2/configs/' + fname + f'_{w}x{h}.npy', 'wb') as f:
    np.save(f, Q_ray_surface)