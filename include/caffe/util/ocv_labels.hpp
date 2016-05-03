#ifndef OCV_LABELS_HPP
#define	OCV_LABELS_HPP

#include <string>

#include <opencv2/opencv.hpp>

namespace caffe {

using namespace cv;
using std::string;

void setLabel(
    Mat& im, 
    const string& label, 
    const Point& org, 
    const double scale=1.0
);

}

#endif	/* OCV_LABELS_HPP */

