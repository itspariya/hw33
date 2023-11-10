#include "../iptools/core.h"
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
Mat I;

#define MAXLEN 256

// ROI structure
struct ROI {
    int x, y, width, height;
    string function;
    vector<int> params;
    bool useOpenCV;

    ROI(int x, int y, int w, int h, string func, vector<int> p, bool useCV = false)
        : x(x), y(y), width(w), height(h), function(std::move(func)), params(std::move(p)), useOpenCV(useCV) {}
};

int main(int argc, char** argv) {
    image src, tgt;
    FILE* fp;
    char str[MAXLEN];
    char infile[MAXLEN];
    char outfile[MAXLEN];
    vector<ROI> rois;

    if ((fp = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Can't open file: %s\n", argv[1]);
        exit(1);
    }

    while (fgets(str, MAXLEN, fp) != NULL) {
        string line(str);
        istringstream iss(line);

        if (line[0] == '#') continue; // Skip comments

        iss >> infile >> outfile;

        int numberOfROIs;
        iss >> numberOfROIs;
        rois.clear(); // Clear the ROI vector for each new line

        for (int i = 0; i < numberOfROIs; ++i) {
            int x, y, width, height;
            string function;
            iss >> x >> y >> width >> height >> function;

            vector<int> params;
            int param;

            // Check if "opencv" is in the parameters and set useOpenCV flag accordingly
            bool useOpenCV = (function == "opencv");

            // Now, after processing parameters, you can set the useOpenCV flag in the ROI struct
            rois.emplace_back(x, y, width, height, function, params, useOpenCV);
        }

        // Process the image
        src.read(infile);
        tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
        Mat I = imread(infile);
        if (I.empty()) {
            cout << "Could not open or find the image: " << infile << endl;
            continue; // Skip this ROI and move to the next one
        }

        tgt.resize(I.rows, I.cols);


        for (const auto& roi : rois) {
            if (roi.useOpenCV) {
                cv::Mat I2;
                // Implement specific OpenCV functions based on roi.function
                if (roi.function == "gray") {
                    // Convert to grayscale
                    cvtColor(I, I2, COLOR_BGR2GRAY);
                    cvtColor(I2, I2, COLOR_GRAY2BGR); // Convert back to BGR if needed
                } else if (roi.function == "blur_avg") {
                    // Apply average blur
                    Size ksize(roi.params[0], roi.params[1]);
                    blur(I, I2, ksize);
                }
                I2.copyTo(I);
            } else {
                // Standard processing
                if (roi.function == "add") {
                    utility::addGrey(src, tgt, roi.params[0], roi.x, roi.y, roi.width, roi.height);
                } else if (roi.function == "binarize") {
                    utility::binarize(src, tgt, roi.params[0], roi.x, roi.y, roi.width, roi.height);
                }
            }
        }

        // Filter to Whole Image
        if (numberOfROIs == 0) {
            string line;
            getline(iss, line); // Use istringstream to get the rest of the line
            istringstream issLine(line);
            string function;
            issLine >> function;

            if (function == "add") {
                int param;
                issLine >> param;
                utility::addGrey(src, tgt, param, 0, 0, src.getNumberOfColumns(), src.getNumberOfRows());
            } else if (function == "binarize") {
                int param;
                issLine >> param;
                utility::binarize(src, tgt, param, 0, 0, src.getNumberOfColumns(), src.getNumberOfRows());
            } else if (function == "scale") {
                double param;
                issLine >> param;
                utility::scale(src, tgt, param);
            } else if (function == "opencv") {
                cv::Mat I = cv::imread(infile);
			    cv::Mat I2;
                // Here, apply OpenCV functions to the entire image
                // For example, apply the "gray" and "blur_avg" functions from your provided code
                string pch;
                while (issLine >> pch) {
                    if (pch == "gray") {
                        cv::Mat I2;
                        // Convert to grayscale
                        cvtColor(I, I2, COLOR_BGR2GRAY);
                        cvtColor(I2, I2, COLOR_GRAY2BGR); // Convert back to BGR if needed
                        I2.copyTo(I);
                    } else if (pch == "blur_avg") {
                        int blur_param;
                        issLine >> blur_param;
                        cv::Mat I2;
                        // Apply average blur
                        Size ksize(blur_param, blur_param);
                        blur(I, I2, ksize);
                        I2.copyTo(I);
                    } else {
                        printf("No function: %s\n", pch.c_str());
                        continue;
                    }
                }
            }
        }

        tgt.save(outfile);
        imwrite(outfile, I);
    }

    fclose(fp);
    return 0;
}
