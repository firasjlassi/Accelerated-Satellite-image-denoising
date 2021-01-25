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


# Fast NLM Denoise filter for monochrome image V1.1
# Based on Fast NLM filter CUDA program provided by Nvidia - CUDA SDK samples
# Alain PAILLOU - August 2019


In_File = "/Your_Directory/Your_Colour_Image.jpg"
Out_File_OpenCV = "/Your_Directory/Your_Colour_Image_OpenCV.jpg"
Out_File_PyCuda = "/Your_Directory/Your_Colour_Image_PyCuda.jpg"


mod = SourceModule("""
__global__ void NLM2_Mono(unsigned char *dest_r, unsigned char *img_r,
int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = NLM_BLOCK_RADIUS + 2;
    const float limxmax = imageW - NLM_BLOCK_RADIUS - 2;
    const float limymin = NLM_BLOCK_RADIUS + 2;
    const float limymax = imageH - NLM_BLOCK_RADIUS - 2;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ = img_r[index3];
                
                clr += clrIJ * weightIJ;
 
                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float clr00 = 0.0;
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4] / 256.0;
        
        clr = clr + (clr00 - clr) * lerpQ;
       
        dest_r[index5] = (int)(clr * 256.0);
    }
}
""")

NLM2_Mono_GPU = mod.get_function("NLM2_Mono")


image_brut_CV = cv2.imread(In_File,cv2.IMREAD_GRAYSCALE)
height,width = image_brut_CV.shape
nb_pixels = height * width


# Set blocks et Grid sizes
nb_ThreadsX = 8
nb_ThreadsY = 8
nb_blocksX = (width // nb_ThreadsX) + 1
nb_blocksY = (height // nb_ThreadsY) + 1

# Set NLM parameters
NLM_Noise = 1.45
Noise = 1.0/(NLM_Noise*NLM_Noise)
lerpC = 0.2



# Algorithm CPU using OpenCV fastNlMeansDenoising routine
tps1 = time.time()
param=21.0
image_brut_CPU=cv2.fastNlMeansDenoising(image_brut_CV, None, param, 3, 5) # application filtre denoise software colour
tps_CPU = time.time() - tps1              
print("OpenCV treatment OK")
print ("CPU treatment time : ",tps_CPU)
print("")
cv2.imwrite(Out_File_OpenCV, image_brut_CPU, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format


# Algorithm GPU using PyCuda
print("Test GPU ",nb_blocksX*nb_blocksY," Blocks ",nb_ThreadsX*nb_ThreadsY," Threads/Block")
tps1 = time.time()
            
r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
drv.memcpy_htod(r_gpu, image_brut_CV)
img_r_gpu = drv.mem_alloc(image_brut_CV.size * image_brut_CV.dtype.itemsize)
drv.memcpy_htod(img_r_gpu, image_brut_CV)
res_r = np.empty_like(image_brut_CV)
            

NLM2_Mono_GPU(r_gpu, img_r_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
         np.float32(lerpC), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
drv.memcpy_dtoh(res_r, r_gpu)

r_gpu.free()
img_r_gpu.free()

image_brut_GPU=res_r

tps_GPU = time.time() - tps1

print("PyCuda treatment OK")
print ("GPU treatment time : ",tps_GPU)
print("")

cv2.imwrite(Out_File_PyCuda, image_brut_GPU, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format


Ecart1 = tps_CPU/tps_GPU
print ("Acceleration factor CPU/GPU : ",Ecart1)



