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

/*-----------------------------------------------------------------------**/
void utility::applyLowPassFilter(Mat& srcROI, Mat& tgtROI, float cutoff) {
    Mat planes[] = {Mat_<float>(srcROI), Mat::zeros(srcROI.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         
    dft(complexI, complexI);            

    // Create a circular mask, low pass
    Mat mask = Mat::zeros(srcROI.size(), CV_32F);
    Point center = Point(mask.rows / 2, mask.cols / 2);
    double radius;

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            radius = sqrt(pow(i - center.x, 2) + pow(j - center.y, 2));
            if (radius < cutoff) {
                mask.at<float>(i, j) = 1.0;
            }
        }
    }

    // Apply mask
    Mat planesDFT[2];
    split(complexI, planesDFT);
    multiply(planesDFT[0], mask, planesDFT[0]);
    multiply(planesDFT[1], mask, planesDFT[1]);
    merge(planesDFT, 2, complexI);

    // Transform back to spatial domain
    idft(complexI, complexI);
    split(complexI, planes);
    normalize(planes[0], tgtROI, 0, 255, NORM_MINMAX, CV_8U);

}
/*-----------------------------------------------------------------------**/
void utility::applyHighPassFilter(Mat& srcROI, Mat& tgtROI, float cutoff) {
    Mat planes[] = {Mat_<float>(srcROI), Mat::zeros(srcROI.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         
    dft(complexI, complexI);            

    // Create a circular mask, high pass
    Mat mask = Mat::zeros(srcROI.size(), CV_32F);
    Point center = Point(mask.rows / 2, mask.cols / 2);
    double radius;

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            radius = sqrt(pow(i - center.x, 2) + pow(j - center.y, 2));
            if (radius > cutoff) {
                mask.at<float>(i, j) = 1.0;
            }
        }
    }

    // Apply mask
    Mat planesDFT[2];
    split(complexI, planesDFT);
    multiply(planesDFT[0], mask, planesDFT[0]);
    multiply(planesDFT[1], mask, planesDFT[1]);
    merge(planesDFT, 2, complexI);

    // Transform back to spatial domain
    idft(complexI, complexI);
    split(complexI, planes);
    normalize(planes[0], tgtROI, 0, 255, NORM_MINMAX, CV_8U);
}
/*-----------------------------------------------------------------------**/

void utility::unsharpMasking(Mat& srcROI, Mat& tgtROI, float cutoff, float T) {
    Mat planes[] = {Mat_<float>(srcROI), Mat::zeros(srcROI.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         
    dft(complexI, complexI);            

    // Create a circular mask, high pass
    Mat mask = Mat::zeros(srcROI.size(), CV_32F);
    Point center = Point(mask.rows / 2, mask.cols / 2);
    double radius;

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            radius = sqrt(pow(i - center.x, 2) + pow(j - center.y, 2));
            if (radius > cutoff) {
                mask.at<float>(i, j) = T; // Scale the high-frequency components by T
            }
        }
    }

    // Apply mask
    Mat planesDFT[2];
    split(complexI, planesDFT);
    multiply(planesDFT[0], mask, planesDFT[0]);
    multiply(planesDFT[1], mask, planesDFT[1]);
    merge(planesDFT, 2, complexI);

    // Transform back to spatial domain
    idft(complexI, complexI);
    split(complexI, planes);
    Mat highFreqComponent;
    normalize(planes[0], highFreqComponent, 0, 255, NORM_MINMAX, CV_8U);

    // Add the scaled high-frequency components back to the original image
    tgtROI = srcROI + highFreqComponent;
}
/*-----------------------------------------------------------------------**/
void utility::applyBandStopFilter(Mat& srcROI, Mat& tgtROI, float lowCutoff, float highCutoff) {
    // Convert to floating-point format and perform DFT
    Mat planes[] = {Mat_<float>(srcROI), Mat::zeros(srcROI.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);

    // Create a band-stop mask
    Mat mask = Mat::ones(srcROI.size(), CV_32F);
    Point center = Point(mask.rows / 2, mask.cols / 2);
    double radius;

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            radius = sqrt(pow(i - center.x, 2) + pow(j - center.y, 2));
            if (radius >= lowCutoff && radius <= highCutoff) {
                mask.at<float>(i, j) = 0;
            }
        }
    }

    // Apply mask
    Mat planesDFT[2];
    split(complexI, planesDFT);
    multiply(planesDFT[0], mask, planesDFT[0]);
    multiply(planesDFT[1], mask, planesDFT[1]);
    merge(planesDFT, 2, complexI);

    // Transform back to spatial domain
    idft(complexI, complexI);
    split(complexI, planes);
    normalize(planes[0], tgtROI, 0, 255, NORM_MINMAX, CV_8U);
}
/*-----------------------------------------------------------------------**/

void utility::greyAugmentedImages(Mat &src, Rect roi, int F, int T)
{
    // Extract ROI from the input image
    cv::Mat ROI = src( roi);

    // a. Rotate original ROI
    imwrite("Original_Roi.jpg", ROI);
    for(int angle = 1; angle <= 3; angle++) {
        Mat rotated;
        cv::rotate(ROI, rotated, angle-1); // Note the rotation code is used directly        
        imwrite("Rotated_ROI_" + std::to_string(angle*90) + ".jpg", rotated);
    }

    // b. Low-pass filter and then rotate
    Mat lowPassROI;
    utility::applyLowPassFilter(ROI, lowPassROI, F); // Assuming applyLowPassFilter is implemented
    for(int angle = 1; angle <= 3; angle++) {
        Mat rotated;
        cv::rotate(lowPassROI, rotated, angle-1);
        imwrite("LowPass_Rotated_ROI_" + std::to_string(angle*90) + ".jpg", rotated);
    }

    // c. Unsharp masking and then rotate
    Mat unsharpROI;
    utility::unsharpMasking(ROI, unsharpROI, F, T); // Assuming applyUnsharpMasking is implemented
    for(int angle = 1; angle <= 3; angle++) {
        Mat rotated;
        cv::rotate(unsharpROI, rotated, angle-1);
        imwrite("Unsharp_Rotated_ROI_" + std::to_string(angle*90) + ".jpg", rotated);
    }
}
/*-----------------------------------------------------------------------**/

void utility::filterHSVComponentAndDisplay(Mat& src, char component, string filterType, int F, int T = 0) {
        // Convert to HSV
        Mat hsv;
        cvtColor(src, hsv, COLOR_BGR2HSV);

        // Split into H, S, V channels
        std::vector<Mat> hsvChannels(3);
        split(hsv, hsvChannels);

        // Select the channel to filter
        Mat& targetChannel = (component == 'H') ? hsvChannels[0] : 
                             (component == 'S') ? hsvChannels[1] : hsvChannels[2];

        // Apply the desired filter
        if (filterType == "low-pass") {
            utility::applyLowPassFilter(targetChannel, targetChannel, F);
        } else if (filterType == "high-pass") {
            utility::applyHighPassFilter(targetChannel, targetChannel, F);
        } else if (filterType == "band-stop") {
            utility::applyBandStopFilter(targetChannel, targetChannel, F, T); // T is used as highF for band-stop
        }

        // Merge back and convert to RGB
        merge(hsvChannels, hsv);
        Mat filteredRGB;
        cvtColor(hsv, filteredRGB, COLOR_HSV2BGR);

        // Display or save the result
        imshow("Filtered Image", filteredRGB);
        waitKey(0);
} 
/*-----------------------------------------------------------------------**/

void utility::processROI(Mat& I, Mat& I2, Rect roi, istringstream& iss) {
    string functionName;
    iss >> functionName;

    cv::Mat roiMat = I(roi); 
    cv::Mat roiMat2 = I2(roi);

    if (functionName == "gray") {
        cv::Mat grayImage;
        cvtColor(I(roi), grayImage, COLOR_BGR2GRAY);
        if (I2.channels() == 3) {
            cv::Mat colorImage;
            cvtColor(grayImage, colorImage, COLOR_GRAY2BGR);
            colorImage.copyTo(I2(roi));
        } else {
            grayImage.copyTo(I2(roi));
        }
    } else if (functionName == "blur_avg") {
        int blurSize;
        iss >> blurSize;
        Mat blurred;
        blur(I(roi), blurred, Size(blurSize, blurSize));
        if (I2.channels() == 3 && blurred.channels() == 1) {
            cvtColor(blurred, blurred, COLOR_GRAY2BGR);
        }
        blurred.copyTo(I2(roi));
    } else if (functionName == "dft") {
        utility::applyDFT(roiMat, roiMat2, roi.x, roi.y, roi.width, roi.height);
    } else if (functionName == "low_pass") {
        float cutoff;
        iss >> cutoff;
        // Apply low-pass filter in the frequency domain
        utility::applyLowPassFilter(roiMat, roiMat2, cutoff);
    } else if (functionName == "high_pass") {
        float cutoff;
        iss >> cutoff;
        // Apply high-pass filter in the frequency domain
        utility::applyHighPassFilter(roiMat, roiMat2, cutoff);
    } else if (functionName == "unsharp") {
        float cutoff, T;
        iss >> cutoff >> T;
        // Apply unsharp masking in the frequency domain
        utility::unsharpMasking(roiMat, roiMat2, cutoff, T);
    } else if (functionName == "band_stop") {
        float lowCutoff, highCutoff;
        iss >> lowCutoff >> highCutoff;
        // Apply band-stop filter in the frequency domain
        utility::applyBandStopFilter(roiMat, roiMat2, lowCutoff, highCutoff);
    } else {
        printf("No OpenCV function for ROI: %s\n", functionName.c_str());
    }

        if (I2.channels() == 3 && roiMat2.channels() == 1) {
            cvtColor(roiMat2, roiMat2, COLOR_GRAY2BGR);
        }
        roiMat2.copyTo(I2(roi));

}
