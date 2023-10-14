import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter
# import imageio

#create a 3D image class
class Image3D():

# Class constructor 
    def __init__(self, image, voxel_dim):
# constructer takes 2 inputs numpy_array representing a 3D image and a tuple of 3 numerical values
# voxel_dims = tuple of voxel dimensions (tuple (dz,dy,dx))
# dz, dx, dy = voxel dimensions
# z, x, y = tuple of arrays represent the c, a and b voxel coordinates 
#image is input image array
        self.image = image
        self.voxel_dim = voxel_dim
# assigning voxel dimensions
        self.dz, self.dy, self.dx = voxel_dim
# to find the image shape. D3, D1 and D2 represent (32,128,128) dimensions of the 3D image
        self.D3, self.D2, self.D1 = image.shape

# getting the voxel coordinates from negative axis to positive with image dimesion as number of intervals
# get the coordinates for interpolation
        self.i = self.dx*np.arange( -(self.D1-1)/2, (self.D1+1)/2, 1)
        self.j = self.dy*np.arange( -(self.D2-1)/2, (self.D2+1)/2, 1)
        self.k = self.dz*np.arange( -(self.D3-1)/2, (self.D3+1)/2, 1)

#x,y,z coordinates vectors
# Create meshgrid of coordinate matrices from coordinate vectors.
        self.z, self.y, self.x = np.meshgrid(self.k, self.j, self.i, indexing = 'ij')


#create a volume resize function with one input of resize ratio which is tuple of 3 elements  
    def volume_resize(self, resize_ratio, display = True, identifier = ''):
#resize ratio tuple of 3 elements       
        r3, r2, r1 = resize_ratio
#getting the new image shape. Multiply resize ratio by image shape to create new shape
#r_D3, r_D1, r_D2 gives new image dimensions
        r_D3, r_D2, r_D1 = round(self.D3*r3), round(self.D2*r2), round(self.D1*r1)
        
#r_dz, r_dx, r_dy are new voxel dimensions
        r_dz, r_dy, r_dx = self.dz*(self.D3/r_D3), self.dy*(self.D2/r_D2), self.dx*(self.D1/r_D1)
        r_voxel_dim=(r_dz, r_dy, r_dx)

#resize-coordinates for interpolation
        r_k = r_dz*np.arange(-(r_D3-1)/2, (r_D3+1)/2, 1)
        r_j = r_dy*np.arange(-(r_D2-1)/2, (r_D2+1)/2, 1)
        r_i = r_dx*np.arange(-(r_D1-1)/2, (r_D1+1)/2, 1)
        
#resize voxel coordinates        
        r_z, r_y, r_x = np.meshgrid(r_k, r_j, r_i, indexing='ij')
        r_vox_coordinates = r_z, r_y, r_x

#using interpn function to get the interpolated image        
        interpolated_image=interpolate.interpn((self.k, self.j, self.i), self.image, (r_z,r_y,r_x), method='linear', bounds_error = False, fill_value = 0)
                
        return interpolated_image
#create a volume resize function with one input of resize ratio which is tuple of 3 elements
  
    def volume_resize_antialias(self, resize_ratio, sigma, display = True, identifier = ''):
#sigma minimum diviation
#filter the image with Gaussian filter
        filtered_image = gaussian_filter(self.image, sigma=sigma)

#resize ratio tuple of 3 elements           
        r3, r2, r1 = resize_ratio

#resize-coordinates for interpolation. multiply resize ratio by image shape to create new shape
#r_D3, r_D1, r_D2 gives new image dimensions
        r_D3, r_D2, r_D1 = round(self.D3*r3), round(self.D2*r2), round(self.D1*r1)

#r_dz, r_dx, r_dy are new voxel dimensions
        r_dz, r_dy, r_dx = self.dz*(self.D3/r_D3), self.dy*(self.D2/r_D2), self.dx*(self.D1/r_D1)
        r_voxel_dim=(r_dz, r_dy, r_dx)

#resize-coordinates for interpolation
        r_k = r_dz*np.arange(-(r_D3-1)/2, (r_D3+1)/2, 1)
        r_j = r_dy*np.arange(-(r_D2-1)/2, (r_D2+1)/2, 1)
        r_i = r_dx*np.arange(-(r_D1-1)/2, (r_D1+1)/2, 1)

#resize voxel coordinates        
        r_z, r_y, r_x = np.meshgrid(r_k, r_j, r_i, indexing='ij')
        r_vox_coordinates = r_z, r_y, r_x

#using interpn function to get the interpolated image        
        interpolated_image=interpolate.interpn((self.k, self.j, self.i), filtered_image, (r_z, r_y, r_x), method='linear', bounds_error = False, fill_value = 0)
        
        return interpolated_image       

#Load the image
image = np.load('image_train00.npy')
voxel_dim = 2.0, 0.5, 0.5
# #calling the class
c= Image3D(image, voxel_dim)

#############################################################################################
# Experiment 1
# Scenario 1: Up-sampling (to higher resolution)
# Specify a resize ratio
resize_ratio = 1, 2.5, 3.2

# Measurement of time
a=time.time()
# Calling volume_resize function
upsampled_image=c.volume_resize(resize_ratio=resize_ratio)
b=time.time()
T1 = b-a
print(f'Resampling time for \'volume_resize\' function: {round(T1, 4 )*1000} ms')

# Calling volume_resize_antialias function
a=time.time()
upsampled_antialias_image=c.volume_resize_antialias(resize_ratio=resize_ratio, sigma=1)
b=time.time()
T2=b-a
print(f'Resampling time for \'volume_resize_antialias\' function: {round(T2,4 )*1000} ms')

# specify the slices to be saved
x = [10,11,12,13,14] 
y = [0,1,2,3,4]  

# Saving the slices
for i, j in zip(x, y): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(upsampled_image[i,:,:], cmap='gray')
    ax1.set_title(f'Axial slice: {j + 1}'' volume_resize')
    ax2.imshow(upsampled_antialias_image[i,:,:], cmap='gray')
    ax2.set_title(f'Axial slice: {j + 1}'' volume_resize_antialias')
    fig.suptitle('Experiment1- Scenario 1: Up-sampling')
    fig.savefig(f'Exp_1_scenario1_Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')
    
if T2 < T1:
    print('volume_resize function takes longer time')
else:
  print('volume_resize_antialias function takes longer time') 
################################################################################################
# Experiment 1
# Scenario 2: Down-sampling (to lower resolution)
# Specify a resize ratio
resize_ratio=0.9,0.8,0.7

# Measuring time
a=time.time()
# Calling volume_resize function
downsampled_image=c.volume_resize(resize_ratio=resize_ratio)
b=time.time()
T1 = b-a
print(f'Resampling time for \'volume_resize\' function: {round(T1, 4 )*1000} ms')

# Calling volume_resize_antialias function
a=time.time()
downsampled_antialias_image=c.volume_resize_antialias(resize_ratio=resize_ratio, sigma=0.5)
b=time.time()
T2=b-a
print(f'Resampling time for \'volume_resize_antialias\' function: {round(T2,4 )*1000} ms')

# Saving slices as images
for i in range(32): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(downsampled_image[i,:,:], cmap='gray')
    ax1.set_title(f'Axial slice: {i + 1}'' volume_resize')
    ax2.imshow(downsampled_antialias_image[i,:,:], cmap='gray')
    ax2.set_title(f'Axial slice: {i + 1}'' volume_resize_antialias')
    fig.suptitle('Experiment1- Scenario 2: Down-sampling')
    fig.savefig(f'Exp_1_scenario2_Axial slice_{i+1}.png',dpi=300,bbox_inches='tight')
    if i>3:
        break
    
if T2 < T1:
    print('volume_resize function takes longer time')
else:
  print('volume_resize_antialias function takes longer time')

######################################################################################
# Experiment 1
# Scenario 3: Resampling. The images are resampled such that the dimensions are same along the 3 dimensions
#Specify a resize ratio to resample isotropically
resize_ratio = 3, 0.75, 0.75

# Measure time
a=time.time()
# Calling volume_resize function
Resampled_image=c.volume_resize(resize_ratio=resize_ratio)
b=time.time()
T1 = b-a
print(f'Resampling time for \'volume_resize\' function: {round(T1, 4 )*1000} ms')

# Calling volume_resize_antialias function
a=time.time()
Resampled_antialias_image=c.volume_resize_antialias(resize_ratio=resize_ratio, sigma=0.3)
b=time.time()
T2=b-a
print(f'Resampling time for \'volume_resize_antialias\' function: {round(T2,4 )*1000} ms')

# choosing specific value of the slice to be saved
x = [25,26,27,28,29] 
y = [0,1,2,3,4]  
# saving 5 slices 
for i, j in zip(x, y): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(Resampled_image[i,:,:], cmap='gray')
    ax1.set_title(f'Axial slice: {j + 1}'' volume_resize')
    ax2.imshow(Resampled_antialias_image[i,:,:], cmap='gray')
    ax2.set_title(f'Axial slice: {j + 1}'' volume_resize_antialias')
    fig.suptitle('Experiment1- Scenario 3: Resampling')
    fig.savefig(f'Exp_1_scenario3_Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')
if T2 < T1:
    print('volume_resize function takes longer time')
else:
  print('volume_resize_antialias function takes longer time')
print(f' Size of Resampled image: {Resampled_image.shape}')
##############################################################################################
# Experiment 2
# Down-sampling the previously up-sampled image (From Scenario 1 in Experiment 1)
C1 = Image3D(upsampled_image, voxel_dim=voxel_dim) 
resize_ratio = 1, 1/2.5, 1/3.2
# Calling volume_resize function
Resized_image = C1.volume_resize(resize_ratio=resize_ratio)

# Calling volume_resize_antialias function
C2 = Image3D(upsampled_antialias_image, voxel_dim=voxel_dim)
Resized_antialias_image = C2.volume_resize_antialias(resize_ratio=resize_ratio, sigma=1)

# saving 5 slices 
for i, j in zip(x,y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(Resized_image [i,:,:], cmap='gray')
    ax1.set_title(f'Axial slice: {j + 1}'' downsamppled_image_resize')
    ax2.imshow(Resized_antialias_image [i,:,:], cmap='gray')
    ax2.set_title(f'Axial slice: {j + 1}'' downsamppled_image_resize_antialias')
    plt.suptitle('Experiment2- Down-sampling the previously up-sampled image')
    fig.savefig(f'Exp_2_Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')
    plt.show()

#mean and standard deviation of voxel intensities
# mean and standard deviation of the voxel-level intensity differences between the original image and the down-sampled images

Original_image = np.load('image_train00.npy')

mean_difference1 = np.mean(Original_image - Resized_image)
print(f'Mean difference between original and resized images:{mean_difference1}')

mean_difference2 = np.mean(Original_image - Resized_antialias_image)
print(f'Mean difference between original and resize_antialias images:{mean_difference2}')

std_difference1 = np.std(Original_image -Resized_image)
print(f'Standard deviation difference between original and resized images:{std_difference1}')

std_difference2 = np.std(Original_image-Resized_antialias_image)
print(f'Standard deviation difference between original and resized images:{std_difference2}')
