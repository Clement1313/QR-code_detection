#pragma once

#include <string>
#include <vector>

#include "../util/image.hh"
#include "../detection/detection.hh"
#include "../detection/preprocess.hh"

namespace qr_code
{

    void save_preprocess_debug(
        const std::string& folder,
        const Preprocess& pre);


    
    void save_labels_debug(
        const std::string& filename,
        const LabelImage& lab);


    void save_regions_debug(
        const std::string& filename,
        const LabelImage& lab,
        const std::vector<Region>& regions);


    void save_elements_debug(
        const std::string& filename,
        const image::rgb24_image& image,
        const std::vector<Element>& elements);


    void save_triplets_debug(
        const std::string& filename,
        const image::rgb24_image& image,
        const std::vector<Triplet>& triplets);


    void save_qr_debug(
        const std::string& filename,
        image::rgb24_image image,
        const std::vector<std::vector<Point>>& corners);

}