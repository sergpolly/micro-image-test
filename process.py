from imageio import imread
from pathlib import Path

import matplotlib.pyplot as plt

import skimage
import scipy

from scipy import ndimage as ndi

from scipy.ndimage import convolve
from scipy.ndimage import correlate
from skimage.morphology import reconstruction
from skimage import feature


#######################################
# mostly stollen from:
# http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
# simply amazing.
#######################################

# use modern pathlib instead of os.path
img_path = Path('./img') / 'Async2_overlapCROP.ome.tif'
# img_path = Path('./img') / 'G2M_T0_20_overlapCROP.ome.tif'
# img_path = Path('./img') / 'T25_8_overlapCROP.ome.tif'

# load tiff image using imageio, which uses Pillow underneath
im = imread(img_path)

im = skimage.img_as_float(im)

# split color channels from a 3D numpy array
r,g,b = [im[:,:,idx] for idx in (0,1,2)]


# # for some images simple thresholding works perfect :
# segmentation = np.where(g<0.3,0,1)
# label_obj,_ = ndi.label(segmentation)
# sizes = np.bincount(label_obj.flatten())
# mask_sizes = sizes > 20
# mask_sizes[0] = 0
# label_obj_clean, num_obj = ndi.label(mask_sizes[label_obj])
# for ii,jj in ndi.find_objects(label_obj_clean):
#     plt.imshow(g[ii,jj])
#     plt.show()



from skimage.morphology import watershed
from skimage.filters import sobel

elevation_map = sobel(g)
plt.imshow(elevation_map)
plt.show()

markers = np.zeros_like(g)
markers[g<0.2] = 1
markers[g>0.5] = 2
plt.imshow(markers)
plt.show()



segmentation = watershed(elevation_map,markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)
plt.imshow(segmentation)
plt.show()



label_obj,_ = ndi.label(segmentation)
sizes = np.bincount(label_obj.flatten())
mask_sizes = sizes > 20
mask_sizes[0] = 0
label_obj_clean, num_obj = ndi.label(mask_sizes[label_obj])
for ii,jj in ndi.find_objects(label_obj_clean):
    plt.imshow(g[ii,jj])
    plt.show()



##################################
#
# some more random/manual image manipulation
#
##################################

# plt.imshow(feature.canny(g,sigma=1))
# plt.show()

# gg = scipy.ndimage.gaussian_filter(g, 1)
# seed = np.copy(gg)
# seed[1:-1, 1:-1] = gg.min()
# mask = gg
# dilated = reconstruction(seed, mask, method='dilation')
# plt.imshow(gg - dilated)
# plt.show()


# def detect_edges(image,masks):
#     edges=np.zeros(image.shape)
#     for mask in masks:
#         edges=np.maximum(scipy.ndimage.convolve(image, mask),edges)
#     return edges


# Faler=[ [[-1,0,1],[-1,0,1],[-1,0,1]], 
#         [[1,1,1],[0,0,0],[-1,-1,-1]],
#     [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
#     [[0,1,0],[-1,0,1],[0,-1,0]] ]

# gg = scipy.ndimage.gaussian_filter(g,sigma=1)

# edges=detect_edges(gg, Faler)
# plt.imshow(edges)
# plt.show()



# w = [[-1,1,-1],[-1,1,-1],[-1,1,-1]]

# # w = [[1,0,0],[0,1,0],[0,0,1]]


# # ww = np.hstack((g[450,78:82],g[450,82:77:-1]))

# # w = np.vstack([ww]*5)
# # w = w - w.min()

# cw = correlate(g,w)

# plt.imshow(cw)
# plt.show()


# plt.plot(scipy.ndimage.correlate1d(g[450,:],ww)); plt.show()