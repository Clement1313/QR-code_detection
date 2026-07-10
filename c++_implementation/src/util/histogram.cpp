#include "util/histogram.hh"
#include <algorithm>
#include <array>
#include <math.h>
#include <thread>
#include <vector>
#include <iostream>

namespace {
  std::size_t thread_count_for(std::size_t work_size) {
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const std::size_t max_threads = hardware_threads == 0 ? 1U : hardware_threads;
    return std::min(max_threads, work_size == 0 ? std::size_t{1} : work_size);
  }

  void accumulate_histogram_range(
      const uint8_t* buffer,
      std::size_t begin,
      std::size_t end,
      std::array<unsigned int, IMAGE_NB_LEVELS>& local_histogram) {
    for (std::size_t offset = begin; offset < end; ++offset) {
      ++local_histogram[buffer[offset]];
    }
  }
}

namespace qr_code {
  histogram_1d histogram(const image::gray8_image& gray_image){
    histogram_1d histogram{};
    const std::size_t pixel_count = static_cast<std::size_t>(gray_image.sx) * static_cast<std::size_t>(gray_image.sy);
    const std::size_t workers = thread_count_for(pixel_count);
    const uint8_t* buffer = gray_image.get_buffer();

    if (workers <= 1) {
      for (std::size_t offset = 0; offset < pixel_count; ++offset) {
        ++histogram.histogram[buffer[offset]];
      }
      return histogram;
    }

    std::vector<std::array<unsigned int, IMAGE_NB_LEVELS>> local_histograms(workers);
    std::vector<std::thread> threads;
    threads.reserve(workers);

    const std::size_t chunk_size = (pixel_count + workers - 1) / workers;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      const std::size_t begin = worker * chunk_size;
      const std::size_t end = std::min(begin + chunk_size, pixel_count);
      if (begin >= end) {
        continue;
      }
      threads.emplace_back(
          accumulate_histogram_range,
          buffer,
          begin,
          end,
          std::ref(local_histograms[worker]));
    }

    for (std::thread& thread : threads) {
      thread.join();
    }

    for (const auto& local_histogram : local_histograms) {
      for (std::size_t level = 0; level < IMAGE_NB_LEVELS; ++level) {
        histogram.histogram[level] += local_histogram[level];
      }
    }

    return histogram;
  }

  histogram_1d histogram_accumule(const image::gray8_image& gray_image){
    histogram_1d histogram{};
    const std::size_t pixel_count = static_cast<std::size_t>(gray_image.sx) * static_cast<std::size_t>(gray_image.sy);
    const std::size_t workers = thread_count_for(pixel_count);
    const uint8_t* buffer = gray_image.get_buffer();

    if (workers <= 1) {
      for (std::size_t offset = 0; offset < pixel_count; ++offset) {
        ++histogram.histogram[buffer[offset]];
      }
    } else {
      std::vector<std::array<unsigned int, IMAGE_NB_LEVELS>> local_histograms(workers);
      std::vector<std::thread> threads;
      threads.reserve(workers);

      const std::size_t chunk_size = (pixel_count + workers - 1) / workers;
      for (std::size_t worker = 0; worker < workers; ++worker) {
        const std::size_t begin = worker * chunk_size;
        const std::size_t end = std::min(begin + chunk_size, pixel_count);
        if (begin >= end) {
          continue;
        }
        threads.emplace_back(
            accumulate_histogram_range,
            buffer,
            begin,
            end,
            std::ref(local_histograms[worker]));
      }

      for (std::thread& thread : threads) {
        thread.join();
      }

      for (const auto& local_histogram : local_histograms) {
        for (std::size_t level = 0; level < IMAGE_NB_LEVELS; ++level) {
          histogram.histogram[level] += local_histogram[level];
        }
      }
    }

    for (int offset = 1;offset < IMAGE_NB_LEVELS; offset++) {
      histogram.histogram[offset] = histogram.histogram[offset-1] + histogram.histogram[offset];
    }
    return histogram;
  }

  void  egalisation_histogram(image::gray8_image& gray_image) {
    histogram_1d histogram = histogram_accumule(gray_image);
    unsigned int born_sup = IMAGE_MAX_LEVEL;
    const std::size_t pixel_count = static_cast<std::size_t>(gray_image.sx) * static_cast<std::size_t>(gray_image.sy);
    const float nombre_pixel = static_cast<float>(pixel_count);
    uint8_t* buffer = gray_image.get_buffer();
    const std::size_t workers = thread_count_for(pixel_count);

    auto equalize_range = [&histogram, born_sup, nombre_pixel, buffer](std::size_t begin, std::size_t end) {
      for (std::size_t offset = begin; offset < end; ++offset) {
        const float hc = histogram.histogram[buffer[offset]];
        buffer[offset] = static_cast<unsigned int>(round(born_sup * (hc / nombre_pixel)));
      }
    };

    if (workers <= 1) {
      equalize_range(0, pixel_count);
      return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers);
    const std::size_t chunk_size = (pixel_count + workers - 1) / workers;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      const std::size_t begin = worker * chunk_size;
      const std::size_t end = std::min(begin + chunk_size, pixel_count);
      if (begin >= end) {
        continue;
      }
      threads.emplace_back(equalize_range, begin, end);
    }

    for (std::thread& thread : threads) {
      thread.join();
    }
  }

}