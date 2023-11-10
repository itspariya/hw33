#include "utility.h"

#define MAXRGB 255
#define MINRGB 0

using namespace cv;

std::string utility::intToString(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

int utility::checkValue(int value)
{
	if (value > MAXRGB)
		return MAXRGB;
	if (value < MINRGB)
		return MINRGB;
	return value;
}
/*-----------------------------------------------------------------------**/
void utility::addGrey(image &src, image &tgt, int value, int x, int y, int width, int height)
{
    tgt = src;

    // Apply the operation only within the ROI
    for (int i = y; i < y + height && i < src.getNumberOfRows(); i++) {
        for (int j = x; j < x + width && j < src.getNumberOfColumns(); j++) {
            int newValue = checkValue(src.getPixel(i, j) + value);
            tgt.setPixel(i, j, newValue); // Set the new value only within the ROI
        }
    }
}

/*-----------------------------------------------------------------------**/
void utility::binarize(image &src, image &tgt, int threshold, int x, int y, int width, int height)
{
    tgt = src;

    // Apply the operation only within the ROI
    for (int i = y; i < y + height && i < src.getNumberOfRows(); i++) {
        for (int j = x; j < x + width && j < src.getNumberOfColumns(); j++) {
            int pixelValue = src.getPixel(i, j) < threshold ? MINRGB : MAXRGB;
            tgt.setPixel(i, j, pixelValue); // Set the new value only within the ROI
        }
    }
}
/*-----------------------------------------------------------------------**/
void utility::scale(image &src, image &tgt, float ratio)
{
	int rows = (int)((float)src.getNumberOfRows() * ratio);
	int cols  = (int)((float)src.getNumberOfColumns() * ratio);
	tgt.resize(rows, cols);
	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<cols; j++)
		{	
			int i2 = (int)floor((float)i/ratio);
			int j2 = (int)floor((float)j/ratio);
			if (ratio == 2) {
				tgt.setPixel(i,j,checkValue(src.getPixel(i2,j2)));
			}

			if (ratio == 0.5) {
				int value = src.getPixel(i2,j2) + src.getPixel(i2,j2+1) + src.getPixel(i2+1,j2) + src.getPixel(i2+1,j2+1);
				tgt.setPixel(i,j,checkValue(value/4));
			}
		}
	}
}

/*-----------------------------------------------------------------------**/
void utility::cv_gray(Mat &src, Mat &tgt)
{
	cvtColor(src, tgt, COLOR_BGR2GRAY);
}

/*-----------------------------------------------------------------------**/
void utility::cv_avgblur(Mat &src, Mat &tgt, int WindowSize)
{
	blur(src,tgt,Size(WindowSize,WindowSize));
}

/*-----------------------------------------------------------------------**/
void utility::applyDFT(cv::Mat &src, cv::Mat &tgt, int x, int y, int width, int height) {
    // Extract the ROI from the source image
    cv::Mat roiSrc = src(cv::Rect(x, y, width, height));

    // Convert to floating-point format
    roiSrc.convertTo(roiSrc, CV_32F);

    // DFT
    cv::Mat dftResult;
    cv::dft(roiSrc, dftResult, cv::DFT_COMPLEX_OUTPUT);

    // Compute the magnitude (amplitude)
    std::vector<cv::Mat> planes;
    cv::split(dftResult, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];

    // Switch to logarithmic scale to display
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

    // Display the DFT amplitude
    cv::imshow("DFT Amplitude", magI);
    cv::waitKey();

    // Inverse DFT to get back to the spatial domain
    cv::Mat inverseTransform;
    cv::idft(dftResult, inverseTransform, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);

    // Place the processed ROI back into the target image
    inverseTransform.copyTo(tgt(cv::Rect(x, y, width, height)));
}