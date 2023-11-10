#ifndef UTILITY_H
#define UTILITY_H

#include "../image/image.h"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <sstream>
#include <math.h>

class utility
{
	public:
		utility();
		virtual ~utility();
		static std::string intToString(int number);
		static int checkValue(int value);
		static void addGrey(image &src, image &tgt, int value, int x, int y, int width, int height);
		static void binarize(image &src, image &tgt, int threshold, int x, int y, int width, int height);
		static void scale(image &src, image &tgt, float ratio);
		static void cv_gray(cv::Mat &src, cv::Mat &tgt);
		static void cv_avgblur(cv::Mat &src, cv::Mat &tgt, int WindowSize);
		static void applyDFT(cv::Mat &src, cv::Mat &tgt, int x, int y, int width, int height);
		static void applyLowPassFilter(Mat& src, Mat& tgt, float cutoff, Rect roi);
		static void applyHighPassFilter(Mat& src, Mat& tgt, float cutoff, Rect roi);
		static void unsharpMasking(Mat& src, Mat& tgt, float cutoff, float T, Rect roi);
		static void greyAugmentedImages(Mat &src, Rect roi, int A, int B);
		static void applyBandStopFilter(Mat& src, Mat& tgt, float lowCutoff, float highCutoff, Rect roi);
		// static void filterHSVComponentAndDisplay(Mat& src, char component, string filterType, int F, int T = 0);
		static void processROI(Mat& I, Mat& I2, Rect roi, istringstream& iss);
};

#endif

