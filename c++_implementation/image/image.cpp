#include "image.hh"
#include <cstdlib>
#include <algorithm>
namespace image {

gray8_image::gray8_image(int _sx, int _sy) {
  sx = _sx;
  sy = _sy;

  length = sx*sy;
  pixels = static_cast<GRAY8>(std::aligned_alloc(TL_IMAGE_ALIGNMENT, length));
}

gray8_image::gray8_image(const gray8_image& other)
  : sx(other.sx), sy(other.sy), length(other.length) {
  pixels = new uint8_t[length];
  std::copy(other.pixels, other.pixels + length, pixels);
}

gray8_image::gray8_image(gray8_image&& other) noexcept
  : sx(other.sx), sy(other.sy), length(other.length), pixels(other.pixels) {
  other.pixels = nullptr;
  other.sx = 0;
  other.sy = 0;
  other.length = 0;
}

gray8_image& gray8_image::operator=(const gray8_image& other) {
  if (this != &other) {
    delete[] pixels;
    sx = other.sx;
    sy = other.sy;
    length = other.length;
    pixels = new uint8_t[length];
    std::copy(other.pixels, other.pixels + length, pixels);
  }
  return *this;
}

gray8_image& gray8_image::operator=(gray8_image&& other) noexcept {
  if (this != &other) {
    delete[] pixels;
    sx = other.sx;
    sy = other.sy;
    length = other.length;
    pixels = other.pixels;
    other.pixels = nullptr;
    other.sx = 0;
    other.sy = 0;
    other.length = 0;
  }
  return *this;
}


gray8_image::~gray8_image() {
  std::free(pixels);
}

const GRAY8& gray8_image::get_buffer() const {
  return pixels;
}

GRAY8& gray8_image::get_buffer() {
  return pixels;
}

rgb24_image::rgb24_image(int _sx, int _sy) {
  sx = _sx;
  sy = _sy;

  length = sx*sy*3;
  pixels = static_cast<RGB8>(std::aligned_alloc(TL_IMAGE_ALIGNMENT, length));
}

rgb24_image::rgb24_image(const rgb24_image& other)
  : sx(other.sx), sy(other.sy), length(other.length) {
  pixels = new uint8_t[length];
  std::copy(other.pixels, other.pixels + length, pixels);
}

rgb24_image::rgb24_image(rgb24_image&& other) noexcept
  : sx(other.sx), sy(other.sy), length(other.length), pixels(other.pixels) {
  other.pixels = nullptr;
  other.sx = 0;
  other.sy = 0;
  other.length = 0;
}

rgb24_image& rgb24_image::operator=(const rgb24_image& other) {
  if (this != &other) {
    delete[] pixels;
    sx = other.sx;
    sy = other.sy;
    length = other.length;
    pixels = new uint8_t[length];
    std::copy(other.pixels, other.pixels + length, pixels);
  }
  return *this;
}

rgb24_image& rgb24_image::operator=(rgb24_image&& other) noexcept {
  if (this != &other) {
    delete[] pixels;
    sx = other.sx;
    sy = other.sy;
    length = other.length;
    pixels = other.pixels;
    other.pixels = nullptr;
    other.sx = 0;
    other.sy = 0;
    other.length = 0;
  }
  return *this;
}

rgb24_image::~rgb24_image() {
  std::free(pixels);
}

const RGB8& rgb24_image::get_buffer() const {
  return pixels;
}

RGB8& rgb24_image::get_buffer() {
  return pixels;
}

gray8_image rgb_to_gray(rgb24_image& image){
  gray8_image new_image{image.sx,image.sy};
  for (int offset = 0; offset < (image.sx * image.sy); offset++)
  {
    uint8_t val_red = image.get_buffer()[offset * 3] * 0.299;
    uint8_t val_green = image.get_buffer()[offset* 3+ 1 ] * 0.587;
    uint8_t val_blue = image.get_buffer()[offset* 3 + 2] * 0.114;
    new_image.get_buffer()[offset] =  val_red + val_green  + val_blue;
  }
  return new_image;
}

rgb24_image gray_to_rgb(gray8_image& image)
{
  rgb24_image new_image{image.sx,image.sy};

  int length = image.sx * image.sy;
  int offset_rgb = 0;
  for (int offset = 0; offset < length;offset++)
  {
    new_image.get_buffer()[offset_rgb] = image.get_buffer()[offset];
    new_image.get_buffer()[offset_rgb + 1] = image.get_buffer()[offset];
    new_image.get_buffer()[offset_rgb + 2] = image.get_buffer()[offset];
    offset_rgb += 3;
  }
  return new_image;
}

}
