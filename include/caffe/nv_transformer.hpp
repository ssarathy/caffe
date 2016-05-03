#ifndef CAFFE_DATA_TRANSFORMER_NV_HPP
#define	CAFFE_DATA_TRANSFORMER_NV_HPP

#include <vector>
#include <boost/array.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "layer.hpp"

namespace caffe {

using boost::array;

template<typename Dtype>
class CoverageGenerator;

template<typename Dtype>
struct BboxLabel_;

struct AugmentSelection;

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class NVTransformationLayer : public Layer<Dtype> {
public:
  typedef Size_<Dtype> Size2v;
  typedef Point_<Dtype> Point2v;
  typedef Rect_<Dtype> Rectv;
  typedef Vec<Dtype, 3> Vec3v;
  typedef Mat_<Vec<Dtype, 1> > Mat1v;
  typedef Mat_<Vec3v> Mat3v;
  typedef BboxLabel_<Dtype> BboxLabel;
  
  explicit NVTransformationLayer(const LayerParameter& param);
  
  virtual ~NVTransformationLayer() {};
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "NVTransformation"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
  
protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top
  );
  

  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top, 
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom
  ) {}
  
  void transform(
      const Mat3v& inputImage,
      const vector<BboxLabel>& inputBboxes,
      Mat3v& outputImage,
      Dtype* outputLabel
  );
  
  
  /**
   * @return a Dtype from [0..1].
   */
  Dtype randDouble();
  
  void visualize_bboxlist(
      const Mat3v& img, 
      const Mat3v& img_aug, 
      const vector<BboxLabel>& bboxlist, 
      const vector<BboxLabel>& bboxlist_aug, 
      const Dtype* transformed_label, 
      const AugmentSelection&
  ) const;

  bool augmentation_flip(
      const Mat3v& img,
      Mat3v& img_aug,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>&
  );
  float augmentation_rotate(
      const Mat3v& img_src,
      Mat3v& img_aug,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>&
  );
  float augmentation_scale(
      const Mat3v& img,
      Mat3v& img_temp,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>&
  );
  void transform_scale(
      const Mat3v& img,
      Mat3v& img_temp,
      const vector<BboxLabel>& bboxList,
      vector<BboxLabel>& bboxList_aug,
      const Size& size
  );
  Point augmentation_crop(
      const Mat3v& img_temp,
      Mat3v& img_aug,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>&
  );
  
  void transform_crop(
      const Mat3v& img_temp,
      Mat3v& img_aug,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>& bboxlist_aug,
      Rect inner,
      Size2i outer_area,
      Point2i outer_offset
  ) const;
  
  float augmentation_hueRotation(
      const Mat3v& img,
      Mat3v& result
  );
  
  float augmentation_desaturation(
      const Mat3v& img,
      Mat3v& result
  );
  
  Mat1v getTransformationMatrix(Rect region, Dtype rotation) const;
  Rect getBoundingRect(Rect region, Dtype rotation) const;
  void matToBlob(const Mat3v& source, Dtype* destination) const;
  void matsToBlob(const vector<Mat3v>& source, Blob<Dtype>& destination) const;
  vector<Mat3v> blobToMats(const Blob<Dtype>& image) const;
  vector<vector<BboxLabel> > blobToLabels(const Blob<Dtype>& labels) const;
  Mat3v dataToMat(
      const Dtype* _data, 
      Size dimensions
  ) const;
  void retrieveMeanImage(Size dimensions = Size());
  void retrieveMeanChannels();
  
  void meanSubtract(Mat3v& source) const;
  void pixelMeanSubtraction(Mat3v& source) const;
  void channelMeanSubtraction(Mat3v& source) const;
  
  NVAugmentationParameter a_param_;
  NVGroundTruthParameter g_param_;
  TransformationParameter t_param_;
  
  shared_ptr<CoverageGenerator<Dtype> > coverage_;
  
  Phase phase_;
  
  Mat3v data_mean_;
  array<Dtype, 3> mean_values_;
  shared_ptr<Caffe::RNG> rng_;
};

}  // namespace caffe

#endif	/* CAFFE_DATA_TRANSFORMER_NV_HPP */
