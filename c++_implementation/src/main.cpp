#include <iostream>

#include "detection/preprocess.hh"
#include "util/image_io.hh"
#define PATH "C:\\Users\\marce\\Desktop\\QR-code_detection"
#define PATH_TEST "../data/IMG_3888.JPG"
#define PATH_EQUALIZED "equalized.tga"
#define PATH_GRAYSCALE "grayscale.tga"
#define PATH_BINARY "binary.tga"
#define PATH_DENOISED "denoisese.tga"
#define PATH_MEDIAN "median.tga"

void test_preprocess(const char* filename)
{
    image::rgb24_image* image = image::load_image(filename);
    if (!image)
    {
        std::cerr << "Error loading image\n";
        return;
    }
    std::shared_ptr<qr_code::Preprocess> preprocess_image =
        qr_code::preprocess(*image);
    image::rgb24_image grayscale =
        image::gray_to_rgb(preprocess_image->image_grayscale);
    image::rgb24_image binary =
        image::gray_to_rgb(preprocess_image->image_binary);
    image::rgb24_image denoised =
        image::gray_to_rgb(preprocess_image->image_denoise);
    image::save_image(preprocess_image->image_equalized, PATH_EQUALIZED);
    image::save_image(grayscale, PATH_GRAYSCALE);
    image::save_image(binary, PATH_BINARY);
    image::save_image(denoised, PATH_DENOISED);
}

// void test_median(const char* filename) // enable median funcion in preprocess.hh
// {
//     image::rgb24_image* image = image::load_image(filename);
//     if (!image)
//     {
//         std::cerr << "Error loading image\n";
//         return;
//     }

//     image::gray8_image grayscale = image::rgb_to_gray(*image);
//     image::gray8_image median_result{ grayscale.sx, grayscale.sy };

//     qr_code::median(grayscale, median_result, 1);

//     image::rgb24_image grayscale_rgb = image::gray_to_rgb(grayscale);
//     image::rgb24_image median_rgb = image::gray_to_rgb(median_result);

//     image::save_image(grayscale_rgb, PATH_GRAYSCALE);
//     image::save_image(median_rgb, PATH_MEDIAN);
// }

int main()
{
    test_preprocess(PATH_TEST);
    // test_median(PATH_TEST);
    return 0;
}
