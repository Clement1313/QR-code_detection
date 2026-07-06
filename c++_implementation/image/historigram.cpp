#include "historigram.hh"

namespace qr_code {
  histogram_1d histogram(image::gray8_image& gray_image){
    histogram_1d histogram{};
    for (int offset = 0; offset < gray_image.sx * gray_image.sy; offset++)
    {
      histogram.histogram[*(gray_image.get_buffer() + offset)]++;
    }
    return histogram;
  }

  void _histogram_accumule(histogram_1d& histogram) {
    
  }

  histogram_1d histogram_accumule(image::gray8_image& gray_image){
    histogram_1d histogram{};
    for (int offset = 0; offset < gray_image.sx * gray_image.sy; offset++)
    {
      histogram.histogram[*(gray_image.get_buffer() + offset)]++;
    }
    f
    return histogram;  }

  histogram_1d histogram_egalise(image::gray8_image& gray_image) {
    histogram_1d histogram = histogram(gray_image);

  }

}