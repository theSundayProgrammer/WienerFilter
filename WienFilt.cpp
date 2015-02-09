// HistLena.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// Example1.cpp : Defines the entry point for the console application.
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#define _USE_MATH_DEFINES
#include <math.h>


using namespace cv;
using namespace std;
typedef complex<float> complexVal;

Mat_<complex<float>> GaussianConvolve(Mat_<complex<float>> image, unsigned int rows, unsigned int cols);
Mat_<complex<float>> GaussianDeconvolve(Mat_<complex<float>> image, unsigned int rows, unsigned int cols);

template<typename T>
Mat_<T> transLocate(Mat_<T>& image)
{
	auto cols = image.cols / 2;
	auto rows = image.rows / 2;
	for (auto i = 0; i < rows; ++i)
		for (auto j = 0; j < cols; ++j)
		{
			//first swap
			T temp = image[i][j];
			image[i][j] = image[i + rows][j + cols];
			image[i + rows][j + cols] = temp;
			//second swap
			temp = image[i][j + cols];
			image[i][j + cols] = image[i + rows][j];
			image[i + rows][j] = temp;
		}
	return image;
}
Mat GetImage(const char* imgFileName)
{
	//const char* lenaPng = R"FileName(C:\Projects\opencv-master\opencv-master\samples\winrt\OcvImageProcessing\OcvImageProcessing\Assets\Lena.png)FileName";
	Mat src = imread(imgFileName);
	if (src.empty())
		throw "Error";
	Mat gs;
	cv::cvtColor(src, gs, CV_BGR2GRAY);
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(gs.rows);
	int n = getOptimalDFTSize(gs.cols); // on the border add zero values
	copyMakeBorder(gs, padded, 0, m - gs.rows, 0, n - gs.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat finalImage;
	padded.convertTo(finalImage, CV_64F);
	{
		Mat shoImg2;
		normalize(finalImage, shoImg2, 0, 1, CV_MINMAX);
		imshow("original", shoImg2);
	}
	return finalImage;
}

Mat CreateImage(const char* imgFileName)
{
	Mat_<double> inputImage = GetImage(imgFileName);
	inputImage = transLocate(inputImage);
	cv::Mat fourierTransform;
	cv::dft(inputImage, fourierTransform, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

	auto const rows = inputImage.rows;
	auto const cols = inputImage.cols;
	auto image = GaussianConvolve(fourierTransform, rows, cols);
	Mat transformed;


	cv::dft(image, transformed, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	Mat_<float> retVal(transformed);
	Mat noise(inputImage.size(),CV_32F);
	randn(noise, 0.0, 1.0);
	Mat img = transLocate(retVal);
	return img+noise;
}
// Test the `showHistogram()` function above
int main(int argc, char* argv[])
{

	if (argc < 2)
	{
		printf("Usage: %s <filename>\n", argv[0]);
		return 0;
	}
	auto inputImage = CreateImage(argv[1]);
	auto const rows = inputImage.rows;
	auto const cols = inputImage.cols;
	{
		Mat shoImg2;
		normalize(inputImage, shoImg2, 0, 1, CV_MINMAX);
		imshow("inter3", shoImg2);
	}
	inputImage = transLocate(Mat_<float>(inputImage));
	cv::Mat fourierTransform;
	cv::dft(inputImage, fourierTransform, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
	Mat_<complex<float>> fted(fourierTransform);
	auto outputImage = GaussianDeconvolve(fted, rows, cols);

	Mat img;
	cv::dft(outputImage, img, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	img = transLocate(Mat_<float>(img));
	Mat shoImg2;
	normalize(img, shoImg2, 0, 1, CV_MINMAX);
	imshow("Deconvolved", shoImg2);

	waitKey(0);
	return 0;
}

double const a = 1000.0;
double const rootPi = sqrt(M_PI / a);
double const expfactor = M_PI*M_PI / a;

Mat_<complex<float>> GaussianConvolve(Mat_<complex<float>> image, unsigned int const rows, unsigned int const cols)
{

	for (auto r = 0u; r < rows / 2; ++r)
		for (auto c = 0u; c < cols / 2; ++c)
		{
			auto x = 0.1*r;
			auto y = 0.1*c;
			double v = rootPi * exp(-expfactor*(x*x + y*y));
			image[r][c] *= v;
		}
	for (auto r = rows / 2; r < rows; ++r)
		for (auto c = 0u; c < cols / 2; ++c)
		{
			auto x = 0.1*(r - rows / 2 + 1);
			auto y = 0.1*c;
			double v = rootPi * exp(-expfactor*(x*x + y*y));
			image[r][c] *= v;
		}
	for (auto r = 0u; r < rows / 2; ++r)
		for (auto c = cols / 2; c < cols; ++c)
		{
			auto x = 0.1*r;
			auto y = 0.1*(c - cols / 2 + 1);
			double v = rootPi * exp(-expfactor*(x*x + y*y));
			image[r][c] *= v;
		}
	for (auto r = rows / 2; r < rows; ++r)
		for (auto c = cols / 2; c < cols; ++c)
		{
			auto x = 0.1*(r - rows / 2 + 1);
			auto y = 0.1*(c - cols / 2 + 1);
			double v = rootPi * exp(-expfactor*(x*x + y*y));
			image[r][c] *= v;
		}
	return image;
}

Mat_<complex<float>> GaussianDeconvolve(Mat_<complex<float>> image, unsigned int const rows, unsigned int const cols)
{
	double const b = 1.0;

	for (auto r = 0u; r < rows / 2; ++r)
		for (auto c = 0u; c < cols / 2; ++c)
		{
			auto x = 0.1*r;
			auto y = 0.1*c;
			double v = rootPi * exp(-expfactor*(x*x + y*y));
			v = v / (v*v + b);
			image[r][c] *= v;
		}
	for (auto r = rows / 2; r < rows; ++r)
		for (auto c = 0u; c < cols / 2; ++c)
		{
			auto x = 0.1*(r - rows / 2 + 1);
			auto y = 0.1*c;
			double v = rootPi * exp(-expfactor*(x*x + y*y));
			v = v / (v*v + b);
			image[r][c] *= v;
		}
	for (auto r = 0u; r < rows / 2; ++r)
		for (auto c = cols / 2; c < cols; ++c)
		{
			auto x = 0.1*r;
			auto y = 0.1*(c - cols / 2 + 1);
			double v = rootPi * exp(-expfactor*(x*x + y*y));
			v = v / (v*v + b);
			image[r][c] *= v;
		}
	for (auto r = rows / 2; r < rows; ++r)
		for (auto c = cols / 2; c < cols; ++c)
		{
			auto x = 0.1*(r - rows / 2 + 1);
			auto y = 0.1*(c - cols / 2 + 1);
			double v = rootPi * exp(-expfactor*(x*x + y*y));
			v = v / (v*v + b);
			image[r][c] *= v;
		}
	return image;
}
