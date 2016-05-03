#include <string>

#include <opencv2/opencv.hpp>

#include "caffe/util/ocv_labels.hpp"

using namespace cv;
using std::string;

namespace caffe {

void setLabel(
    Mat& im, 
    const string& label, 
    const Point& org, 
    const double scale
) {
  static const int fontface = FONT_HERSHEY_SIMPLEX;
  static const int thickness = 1;
  int baseline = 0;

  Size text = getTextSize(label, fontface, scale, thickness, &baseline);
  rectangle(
      im, 
      org + Point(0, baseline), 
      org + Point(text.width, -text.height),
      CV_RGB(0, 0, 0), 
      CV_FILLED
  );
  putText(
      im, 
      label, 
      org, 
      fontface, 
      scale, 
      CV_RGB(255, 255, 255), 
      thickness,
      20
  );
}

}