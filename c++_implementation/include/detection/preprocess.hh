#ifndef QR_CODE_DETECTION_PREPROCESS_HH
#define QR_CODE_DETECTION_PREPROCESS_HH

#include "../util/image.hh"
#include <memory>

namespace qr_code {
struct Preprocess {
  Preprocess(image::rgb24_image image,image::rgb24_image equalized,image::gray8_image gray,image::gray8_image bin,image::gray8_image denoises):
  image_originel(image), image_equalized(equalized), image_grayscale(gray), image_binary(bin),image_denoise(denoises)
  {
  };
  image::rgb24_image image_originel;
  image::rgb24_image image_equalized;
  image::gray8_image image_grayscale;
  image::gray8_image image_binary;
  image::gray8_image image_denoise;
};

std::shared_ptr<Preprocess> preprocess(const image::rgb24_image &image);
}
#endif // QR_CODE_DETECTION_PREPROCESS_HH
