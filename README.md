# Accelerated-Satellite-image-denoising

<img alt="GitHub followers" src="https://img.shields.io/github/followers/kalifiabillal?color=yellow&label=kalifiabillal&style=for-the-badge">   <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/kalifiabillal/Android-Arduino-Automotive?style=for-the-badge">   <img alt="Visual Studio App Center (Minimum) OS Version" src="https://img.shields.io/visual-studio-app-center/releases/osver/kalifiabillal/Android-Arduino-Automotive/a87b9e745655355612fff4418953e0c3f7074250?style=for-the-badge">   <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/Kalifiabillal/Android-Arduino-Automotive?color=green&style=for-the-badge">   <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/kalifiabillal/Android-Arduino-Automotive?style=for-the-badge">

## Accelerated Satellite image denoising using Nvidia GPU's ?

Using an Nvidia GPU GTX 1650 (3910M) and an I5 9300H, Satellite images are corrupted by noise during image
acquisition and transmission. The removal of noise from the image by attenuating the high-frequency image components removes important details as well. In order to 
retain the useful information, improve the visual appearance, and accurately classify an image, an effective denoising technique is required. 

## What is Cuda ?

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia.CUDA is a parallel 
computing platform and application programming interface model created by Nvidia. It allows software developers and software engineers to use a CUDA-enabled 
graphics processing unit for general purpose processing – an approach termed GPGPU.

## KNN Filter (Wiki)

Non-local means is an algorithm in image processing for image denoising. Unlike "local mean" filters, which take the mean value of a group of pixels surrounding a 
target pixel to smooth the image, non-local means filtering takes a mean of all pixels in the image, weighted by how similar these pixels are to the target pixel. 
This results in much greater post-filtering clarity, and less loss of detail in the image compared with local mean algorithms.

If compared with other well-known denoising techniques, non-local means adds "method noise" (i.e. error in the denoising process) which looks more like white noise, 
which is desirable because it is typically less disturbing in the denoised product.Recently non-local means has been extended to other image processing 
applications such as deinterlacing, view interpolation, and depth maps regularization.
