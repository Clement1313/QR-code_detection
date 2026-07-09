#ifndef QR_CODE_DETECTION_HISTORIGRAM_HH
#define QR_CODE_DETECTION_HISTORIGRAM_HH
#include "image.hh"

namespace qr_code {
typedef struct { unsigned int histogram[IMAGE_NB_LEVELS]; } histogram_1d;

    /*
      Signature : histogram: (tifo::gray8_image) -> histogram_1d
      Description: réalise un histogramme d'une image grise
    */
    histogram_1d histogram(const image::gray8_image& gray_image);


    /*
      Signature: histogram_accumule: (tifo::gray8_imrage) -> histogram_1d
      Description: retourne l'histogram accumulée
    */
    histogram_1d histogram_accumule(const image::gray8_image& gray_image);

    /*
      Signature: egalisation_histogram: (tifo::gray8_imrage) -> void
      Description: egalise l'illumination de l'image
    */
    void  egalisation_histogram(image::gray8_image& gray_image);

}
#endif // QR_CODE_DETECTION_HISTORIGRAM_HH
