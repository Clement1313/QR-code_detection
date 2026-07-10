#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "util/image_io.hh"

namespace qr_code
{
    struct Point
    {
        double row;
        double col;
    };

    struct LabelImage
    {
        int sx;
        int sy;
        std::vector<int32_t> data; // size sx * sy

        int32_t& at(int x, int y)
        {
            return data[y * sx + x];
        }
        int32_t at(int x, int y) const
        {
            return data[y * sx + x];
        }
    };

    struct Region
    {
        int label;
        std::array<int, 4> bbox;
        long area;
        std::vector<std::pair<int, int>>
            coords;
        double eccentricity;
    };

    struct Element
    {
        std::array<int, 4> bbox;
        std::vector<Point> corners;
        long area;
    };

    struct Triplet
    {
        Element a;
        Element b;
        Element c;
    };

    LabelImage labels(const image::gray8_image& denoise);

    std::vector<Region> regionprops(const LabelImage& lab);

    bool square_filter(const image::gray8_image& denoise,
                        const image::gray8_image& binary,
                        const Region& region,
                        std::vector<Point>& corners_out);

    std::vector<Triplet> get_triplets(const std::vector<Element>& elements,
                                      int image_sx, int image_sy,
                                      double angle_tolerance = 35.0);

    std::vector<Point> get_qr_corners(const Triplet& triplet);

    std::vector<Triplet>
    filter_contained_triplets(const std::vector<Triplet>& triplets,
                              double size_tolerance = 0.9,
                              double overlap_threshold = 0.8);

} // namespace qr_code