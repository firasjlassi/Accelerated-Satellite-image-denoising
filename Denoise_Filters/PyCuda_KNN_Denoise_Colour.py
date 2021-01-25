# coding=utf-8

import time
import cv2
import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule


# KNN Denoise filter for colour image V1.1
# Based on KNN filter CUDA program provided by Nvidia - CUDA SDK samples
# Alain PAILLOU - August 2019


In_File = "/Your_Directory/Your_Colour_Image.jpg"
Out_File_OpenCV = "/Your_Directory/Your_Colour_Image_OpenCV.jpg"
Out_File_PyCuda = "/Your_Directory/Your_Colour_Image_PyCuda.jpg"


mod = SourceModule("""
__global__ void KNN_Colour(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = NLM_BLOCK_RADIUS + 2;
    const float limxmax = imageW - NLM_BLOCK_RADIUS - 2;
    const float limymin = NLM_BLOCK_RADIUS + 2;
    const float limymax = imageH - NLM_BLOCK_RADIUS - 2;
   
    
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};
        float3 clr00 = {0, 0, 0};
        float3 clrIJ = {0, 0, 0};
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4];
        clr00.y = img_g[index4];
        clr00.z = img_b[index4];
    
        for(float i = -NLM_BLOCK_RADIUS; i <= NLM_BLOCK_RADIUS; i++)
            for(float j = -NLM_BLOCK_RADIUS; j <= NLM_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index2];
                clrIJ.y = img_g[index2];
                clrIJ.z = img_b[index2];
                float distanceIJ = ((clrIJ.x - clr00.x) * (clrIJ.x - clr00.x)
                + (clrIJ.y - clr00.y) * (clrIJ.y - clr00.y)
                + (clrIJ.z - clr00.z) * (clrIJ.z - clr00.z)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr.x = clr.x + (clr00.x / 256.0 - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y / 256.0 - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z / 256.0 - clr.z) * lerpQ;
        
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
""")

KNN_Colour_GPU = mod.get_function("KNN_Colour")


image_brut_CV = cv2.imread(In_File,cv2.IMREAD_COLOR)
height,width,layers = image_brut_CV.shape
nb_pixels = height * width

# Set blocks et Grid sizes
nb_ThreadsX = 8
nb_ThreadsY = 8
nb_blocksX = (width // nb_ThreadsX) + 1
nb_blocksY = (height // nb_ThreadsY) + 1

# Set KNN parameters
KNN_Noise = 0.32
Noise = 1.0/(KNN_Noise*KNN_Noise)
lerpC = 0.2



# Algorithm CPU using OpenCV fastNlMeansDenoisingColored routine
tps1 = time.time()
param=21.0
image_brut_CPU=cv2.fastNlMeansDenoisingColored(image_brut_CV, None, param, param, 3, 5) # application filtre denoise software colour
image_brut_CPU=cv2.cvtColor(image_brut_CPU, cv2.COLOR_BGR2RGB)
tps_CPU = time.time() - tps1              
print("OpenCV treatment OK")
print ("CPU treatment time : ",tps_CPU)
print("")
cv2.imwrite(Out_File_OpenCV, cv2.cvtColor(image_brut_CPU, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format


# Algorithm GPU using PyCuda
print("Test GPU ",nb_blocksX*nb_blocksY," Blocks ",nb_ThreadsX*nb_ThreadsY," Threads/Block")
tps1 = time.time()

b,g,r = cv2.split(image_brut_CV)
b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
img_b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
drv.memcpy_htod(b_gpu, b)
drv.memcpy_htod(img_b_gpu, b)
res_b = np.empty_like(b)
            
g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
drv.memcpy_htod(g_gpu, g)
img_g_gpu = drv.mem_alloc(g.size * g.dtype.itemsize)
drv.memcpy_htod(img_g_gpu, g)
res_g = np.empty_like(g)
            
r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
drv.memcpy_htod(r_gpu, r)
img_r_gpu = drv.mem_alloc(r.size * r.dtype.itemsize)
drv.memcpy_htod(img_r_gpu, r)
res_r = np.empty_like(r)

KNN_Colour_GPU(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
         np.float32(lerpC), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

drv.memcpy_dtoh(res_r, r_gpu)
drv.memcpy_dtoh(res_g, g_gpu)
drv.memcpy_dtoh(res_b, b_gpu)
r_gpu.free()
g_gpu.free()
b_gpu.free()
img_r_gpu.free()
img_g_gpu.free()
img_b_gpu.free()

image_brut_GPU=cv2.merge((res_r,res_g,res_b))

tps_GPU = time.time() - tps1

print("PyCuda treatment OK")
print ("GPU treatment time : ",tps_GPU)
print("")

cv2.imwrite(Out_File_PyCuda, cv2.cvtColor(image_brut_GPU, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format

Ecart1 = tps_CPU/tps_GPU
print ("Acceleration factor CPU/GPU : ",Ecart1)
