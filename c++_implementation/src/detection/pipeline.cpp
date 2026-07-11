#include "detection/pipeline.hh"

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "detection/preprocess.hh"
#include "util/image_io.hh"

namespace qr_code
{
    std::vector<std::vector<Point>> process_directory(const std::string& image_path)
    {
        image::rgb24_image* im = image::load_image(image_path.c_str());
        if (!im)
            throw std::runtime_error("Impossible de charger " + image_path);

        std::shared_ptr<Preprocess> pre = preprocess(*im);

        LabelImage lab = labels(pre->image_denoise);
        std::vector<Region> regions = regionprops(lab);

        std::vector<Element> elements;
        for (const Region& region : regions)
        {
            std::vector<Point> corners;
            bool ok = square_filter(pre->image_denoise, pre->image_binary,
                                    region, corners);
            if (ok)
                elements.push_back({ region.bbox, corners, region.area });
        }

        std::vector<Triplet> triplets = get_triplets(elements, im->sx, im->sy);
        triplets = filter_contained_triplets(triplets);

        std::cout << triplets.size() << " QR code detectes\n";

        image::rgb24_image result{ im->sx, im->sy };
        const uint8_t* src = im->get_buffer();
        uint8_t* dst = result.get_buffer();
        std::copy(src, src + static_cast<size_t>(im->sx) * im->sy * 3, dst);

        std::vector<std::vector<Point>> qr_codes;
        for (const Triplet& triplet : triplets)
        {
            std::vector<Point> corners = get_qr_corners(triplet);
            qr_codes.push_back(corners);
            draw_qr(result, corners);
        }

        std::filesystem::path output_dir("results");
        std::filesystem::create_directories(output_dir);

        std::filesystem::path input_path(image_path);
        std::filesystem::path output_path =
            output_dir / (input_path.stem().string() + "_res.png");

        image::save_image(result, output_path.string().c_str());

        delete im;
        return qr_codes;
    }

} // namespace qr_code