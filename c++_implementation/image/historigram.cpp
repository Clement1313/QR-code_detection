#include "historigram.hh"
#include <math.h>

namespace qr_code {
  histogram_1d histogram(image::gray8_image& gray_image){
    histogram_1d histogram{};
    for (int offset = 0; offset < gray_image.sx * gray_image.sy; offset++)
    {
      histogram.histogram[*(gray_image.get_buffer() + offset)]++;
    }
    return histogram;
  }

  histogram_1d histogram_accumule(image::gray8_image& gray_image){
    histogram_1d histogram{};
    for (int offset = 0; offset < gray_image.sx * gray_image.sy; offset++)
    {
      histogram.histogram[*(gray_image.get_buffer() + offset)]++;
    }
    for (int offset = 1;offset < IMAGE_NB_LEVELS; offset++) {
      histogram.histogram[offset] = histogram.histogram[offset-1] + histogram.histogram[offset];
    }
    return histogram;
  }

  void  egalisation_histogram(image::gray8_image& gray_image) {
    histogram_1d histogram = histogram_accumule(gray_image);
    unsigned int born_sup = IMAGE_MAX_LEVEL;
    float nombre_pixel = static_cast<float>(gray_image.sx * gray_image.sy);
    for (int offset = 0; offset < gray_image.sx * gray_image.sy; offset++) {
      float hc = histogram.histogram[gray_image.get_buffer()[offset]];
      gray_image.get_buffer()[offset] = static_cast<unsigned int>(round(born_sup* (hc / nombre_pixel)));
    }
  }

}