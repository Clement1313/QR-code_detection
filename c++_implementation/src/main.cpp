#include "detection/preprocess.hh"
#include "util/image_io.hh"
#include "evaluation/evaluation.hh"


#include <iostream>
#define PATH "C:\\Users\\marce\\Desktop\\QR-code_detection"
#define PATH_TEST "../data/IMG_3888.JPG"
#define PATH_EQUALIZED "equalized.tga"
#define PATH_GRAYSCALE "grayscale.tga"
#define PATH_BINARY "binary.tga"
#define PATH_DENOISED "denoisese.tga"
#define DATASET_FOLDER "../data/Dataset/detection/monitor"


void test_preprocess(const char * filename) {
    image::rgb24_image* image = image::load_image(filename);
    if (!image) {
      std::cerr << "Error loading image\n";
      return;
    }
    std::shared_ptr<qr_code::Preprocess> preprocess_image = qr_code::preprocess(*image);
    image::rgb24_image grayscale = image::gray_to_rgb(preprocess_image->image_grayscale);
    image::rgb24_image binary  = image::gray_to_rgb(preprocess_image->image_binary);
    image::rgb24_image denoised = image::gray_to_rgb(preprocess_image->image_denoise);
    image::save_image(preprocess_image->image_equalized,PATH_EQUALIZED);
    image::save_image(grayscale,PATH_GRAYSCALE);
    image::save_image(binary,PATH_BINARY);
    image::save_image(denoised,PATH_DENOISED);
}

int main() {
  test_preprocess(PATH_TEST);
  // qr_code::evaluate_folder(DATASET_FOLDER, 0.5, true); // execute evaluation pipeline
  return 0;
}
