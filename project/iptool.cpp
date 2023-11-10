#include "../iptools/core.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <vector>

using namespace std;
using namespace cv;

#define MAXLEN 256

int main (int argc, char** argv) {
    image src, tgt;
    FILE *fp;
    char str[MAXLEN];
    string infile, outfile;
    int roiCount = 0;

    if ((fp = fopen(argv[1],"r")) == NULL) {
        fprintf(stderr, "Can't open file: %s\n", argv[1]);
        exit(1);
    }

    while(fgets(str,MAXLEN,fp) != NULL) {
        if (str[0] == '#') continue;

        istringstream iss(str);
        iss >> infile >> outfile;

        string processingType;
        iss >> processingType;

        char infileCStr[MAXLEN];
        strcpy(infileCStr, infile.c_str());

        if (processingType == "opencv") {
            cv::Mat I = cv::imread(infile);
            cv::Mat I2 = I.clone();

            if (I.empty()) {
                cout << "Could not open or find the image: " << infile << endl;
                return -1;
            }

            iss >> roiCount;

            if (roiCount == 0) {
                // Apply function to the entire image
                string functionName;
                iss >> functionName;
                Rect roi(0, 0, I.cols, I.rows);
                utility::processROI(I, I2, roi, iss);
            } else {
                for (int i = 0; i < roiCount; i++) {
                    int x, y, width, height;
                    iss >> x >> y >> width >> height;
                    Rect roi(x, y, width, height);
                    utility::processROI(I, I2, roi, iss);
                }
            }

            imwrite(outfile, I2);
        } else {
            src.read(infileCStr); 
            tgt = src; // clone source to target for processing

            iss >> roiCount;

            for (int i = 0; i < roiCount; i++) {
                int x, y, w, h;
                iss >> x >> y >> w >> h;

                string functionName;
                iss >> functionName;
                if (functionName == "add") {
                    int value;
                    iss >> value;
                    utility::addGrey(src, tgt, value, x, y, w, h);
                }
                else if (functionName == "binarize") {
                    int threshold;
                    iss >> threshold;
                    utility::binarize(src, tgt, threshold, x, y, w, h);
                }
                else if (functionName == "scale") {
                    float scale;
                    iss >> scale;
                    utility::scale(src, tgt, scale);
                }
                else {
                    printf("No custom function for ROI: %s\n", functionName.c_str());
                    continue;
                }
            }

            tgt.save(outfile.c_str());
        }
    }

    fclose(fp);
    return 0;
}
