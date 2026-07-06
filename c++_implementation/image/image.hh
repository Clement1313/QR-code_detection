#ifndef IMAGE_HH
#define	IMAGE_HH

#include <cstdint>

#define IMAGE_NB_LEVELS 256
#define IMAGE_MAX_LEVEL 255
#define TL_IMAGE_ALIGNMENT 64

namespace image {

    typedef uint8_t* GRAY8 alignas(TL_IMAGE_ALIGNMENT);
    typedef uint8_t* RGB8 alignas(TL_IMAGE_ALIGNMENT) ;

class gray8_image {

  public:
            gray8_image(int sx, int sy);
            virtual ~gray8_image();

             const GRAY8& get_buffer() const;

            GRAY8& get_buffer();

  public:
            int sx;
            int sy;
            std::size_t length;
            GRAY8 pixels;
    };

class rgb24_image {

        public:

            rgb24_image(int sx, int sy);
            virtual ~rgb24_image();
            const RGB8& get_buffer() const;
            RGB8& get_buffer();

    public:
            int sx;
            int sy;
            std::size_t length;
            RGB8 pixels;
};

   gray8_image rgb_to_gray(rgb24_image& image);

   rgb24_image gray_to_rgb(gray8_image& image);


}
#endif	/* IMAGE_HH */
