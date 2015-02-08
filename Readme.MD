This project uses OpenCV (C++) to implement Weiner Filtering simulation
It uses the Gaussian function or more specifically  its Fourier Transforms (FT) as described in:

 Weisstein, Eric W. "Fourier Transform--Gaussian." From MathWorld--A Wolfram Web Resource.
 http://mathworld.wolfram.com/FourierTransformGaussian.html 
 
 The image is first translated where the first and third quadrants are swapped. The second and fourth quadrants are swapped likewise. The FT of the image is compute. It is then multiplied with the FT of the Gaussian.  The inverse transform is then polluted with noise. Weiner filtering is used to deconvolve the image. 
 
 It was developed on Windows but it uses just OpenCV GUI and hence should work with any OS with OpenCV.
 