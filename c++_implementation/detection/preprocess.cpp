#include "preprocess.hh"
#include "../image/historigram.hh"
#include <vector>
#include <algorithm>
#include <cmath>
namespace qr_code {

void color_intensity(const image::rgb24_image & image,image::gray8_image& buffer, char color)
{
  int index = 0;
  switch (color) {
    case 'R': {
      index = 0;
      break;
    }
    case 'G': {
      index = 1;
      break;
    }
    case 'B': {
      index = 2;
      break;
    }
    default:;
    index = 0;
    break;
    }

  for (int offset = 0; offset < image.sx * image.sy; offset++) {
    int id = 3 * offset;
    uint8_t val = image.get_buffer()[id + index];
    buffer.get_buffer()[offset] = val;
  }
  egalisation_histogram(buffer);

}



void equalized_image(const image::rgb24_image & image,image::rgb24_image & buffer) {
   image::gray8_image red_intensity {image.sx,image.sy};
   image::gray8_image green_intensity {image.sx,image.sy};
   image::gray8_image blue_intensity {image.sx,image.sy};

  color_intensity(image,red_intensity,'R');
  color_intensity(image,green_intensity,'G');
  color_intensity(image,blue_intensity,'B');

  for (int offset = 0; offset < buffer.sx * buffer.sy; offset++) {
    int offset_rgb = offset * 3;
    buffer.get_buffer()[offset_rgb] = red_intensity.get_buffer()[offset];
    buffer.get_buffer()[offset_rgb + 1] = green_intensity.get_buffer()[offset];
    buffer.get_buffer()[offset_rgb + 2] = blue_intensity.get_buffer()[offset];
  }

}




int mean(image::gray8_image &image,int x,int y, uint8_t rayon,uint8_t blocksize) {
  int result = 0;
  for (int dy = y - rayon; dy <= y + rayon; dy++) {
   for (int dx = x - rayon; dx <= x + rayon; dx++) {
     int indice_y = dy;
     if (indice_y < 0) {
       indice_y *= -1;
     }
     if (indice_y >= image.sy) {
       indice_y = 2 * image.sy - dy - 2;
     }
     int indice_x = dx;
     if (indice_x < 0 ) {
       indice_x *= -1;
     }
     if (indice_x >= image.sx) {
       indice_x = 2 * image.sx - dx  - 2;
     }
     result += image.get_buffer()[indice_y * image.sx + indice_x];
   }
  }

  return round(static_cast<float>(result)/static_cast<float>(blocksize * blocksize));
}

void threshold_local(std::vector<uint8_t>& buffer,image::gray8_image & image,uint8_t blocksize, uint8_t offset) {
  if (!(blocksize & 1)) {
    blocksize += 1;
  }
  uint8_t rayon = (blocksize - 1) /2;

  for (int y = 0; y < image.sy; y++) {
    for (int x = 0; x < image.sx; x++) {
      buffer[y * image.sx + x] = std::max(0,mean(image,x,y,rayon, blocksize) - offset);
    }
  }
}

void bin(image::gray8_image& image,image::gray8_image& buffer) {
  std::vector<uint8_t> threshold(image.sx * image.sy,0);
  threshold_local(threshold,image,35,10);
  for (int offset = 0; offset < image.sx * image.sy; offset++) {
    uint8_t val = image.get_buffer()[offset];
    if (val > threshold[offset]) {
      buffer.get_buffer()[offset] = 255;
    }
    else {
      buffer.get_buffer()[offset] = 0;
    }
  }

}

void erosion(const image::gray8_image& buffer, image::gray8_image& eroded,int radius)
{
  for (int y = 0; y < buffer.sy; ++y) {
    for (int x = 0; x < buffer.sx; ++x) {
     uint8_t  min_value = 255;

      for (int dy = -radius; dy <= radius; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= buffer.sy) {
          continue;
        }

        for (int dx = -radius; dx <= radius; ++dx) {
          int xx = x + dx;
          if (xx < 0 || xx >= buffer.sx) {
            continue;
          }

          uint8_t  value = buffer.get_buffer()[yy * buffer.sx + xx];
          min_value = std::min(min_value, value);
        }
      }
      eroded.get_buffer()[y * buffer.sx + x] = min_value;
    }
  }
}

void dilatation(const image::gray8_image& input, image::gray8_image& dilation, int radius)
{
  for (int y = 0; y < input.sy; ++y) {
    uint8_t* lineptr = dilation.get_buffer() + y * input.sx;

    for (int x = 0; x < input.sx; ++x) {
      uint8_t max_value = 0;

      for (int dy = -radius; dy <= radius; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= input.sy) {
          continue;
        }

        for (int dx = -radius; dx <= radius; ++dx) {
          int xx = x + dx;
          if (xx < 0 || xx >= input.sx) {
            continue;
          }

          uint8_t value = input.get_buffer()[yy * input.sx + xx];
          max_value = std::max(max_value, value);
        }
      }

      lineptr[x] = max_value;
    }
  }
}

void opening(const image::gray8_image& input, image::gray8_image& result,int  radius) {
  image::gray8_image save {input.sx,input.sy};
  erosion(input,save,radius);
  dilatation(save,result,radius);
}

void closing( const image::gray8_image& input, image::gray8_image& result,int  radius) {
  image::gray8_image save {input.sx,input.sy};
  dilatation(input,save,radius);
  erosion(save,result,radius);
}

void denoise(const image::gray8_image& input, image::gray8_image& result,int  radius) {
  image::gray8_image save {input.sx,input.sy};
  closing(input,save,radius);
  opening(save,result,radius);
}

std::shared_ptr<Preprocess> preprocess(const image::rgb24_image &image) {
  image::rgb24_image equalized {image.sx,image.sy};
  equalized_image(image,equalized);
  image::gray8_image gray  = image::rgb_to_gray(equalized);
  image::gray8_image bin_im {image.sx,image.sy};
  bin(gray,bin_im);
  image::gray8_image denoises {image.sx,image.sy};
  denoise(bin_im,denoises,1);
  return std::make_shared<Preprocess>(image,equalized,gray,bin_im,denoises);
}


}

