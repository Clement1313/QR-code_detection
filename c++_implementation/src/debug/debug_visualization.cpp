#include "debug/debug_visualization.hh"

#include <cmath>
#include <filesystem>
#include <random>

#include "util/image_io.hh"

namespace qr_code
{

    namespace
    {
        void color_from_id(int id, uint8_t& r, uint8_t& g, uint8_t& b)
        {
            std::mt19937 gen(id * 12345);

            r = static_cast<uint8_t>(50 + gen() % 205);
            g = static_cast<uint8_t>(50 + gen() % 205);
            b = static_cast<uint8_t>(50 + gen() % 205);
        }

        /**
         * Définit un pixel RGB
         */
        void set_pixel(image::rgb24_image& image, int x, int y, uint8_t r,
                       uint8_t g, uint8_t b)
        {
            if (x < 0 || y < 0 || x >= image.sx || y >= image.sy)
                return;

            int offset = 3 * (y * image.sx + x);

            image.get_buffer()[offset] = r;
            image.get_buffer()[offset + 1] = g;
            image.get_buffer()[offset + 2] = b;
        }

        void draw_rectangle(image::rgb24_image& image,
                            const std::array<int, 4>& bbox, uint8_t r,
                            uint8_t g, uint8_t b)
        {
            int min_y = bbox[0];
            int min_x = bbox[1];
            int max_y = bbox[2];
            int max_x = bbox[3];

            for (int x = min_x; x < max_x; x++)
            {
                set_pixel(image, x, min_y, r, g, b);
                set_pixel(image, x, max_y, r, g, b);
            }

            for (int y = min_y; y < max_y; y++)
            {
                set_pixel(image, min_x, y, r, g, b);
                set_pixel(image, max_x, y, r, g, b);
            }
        }

        void draw_point(image::rgb24_image& image, const Point& p, uint8_t r,
                        uint8_t g, uint8_t b)
        {
            int x = static_cast<int>(p.col);
            int y = static_cast<int>(p.row);

            for (int dy = -2; dy <= 2; dy++)
            {
                for (int dx = -2; dx <= 2; dx++)
                {
                    set_pixel(image, x + dx, y + dy, r, g, b);
                }
            }
        }

    } // namespace

    void save_labels_debug(const std::string& filename, const LabelImage& lab)
    {
        image::rgb24_image output(lab.sx, lab.sy);

        for (int y = 0; y < lab.sy; y++)
        {
            for (int x = 0; x < lab.sx; x++)
            {
                int label = lab.at(x, y);

                if (label == 0)
                {
                    set_pixel(output, x, y, 0, 0, 0);
                }
                else
                {
                    uint8_t r, g, b;

                    color_from_id(label, r, g, b);

                    set_pixel(output, x, y, r, g, b);
                }
            }
        }

        image::save_image(output, filename.c_str());
    }

    void save_regions_debug(const std::string& filename, const LabelImage& lab,
                            const std::vector<Region>& regions)
    {
        image::rgb24_image output(lab.sx, lab.sy);

        for (int y = 0; y < lab.sy; y++)
        {
            for (int x = 0; x < lab.sx; x++)
            {
                set_pixel(output, x, y, 0, 0, 0);
            }
        }

        for (const Region& region : regions)
        {
            uint8_t r, g, b;

            color_from_id(region.label, r, g, b);

            for (const auto& coord : region.coords)
            {
                int y = coord.first;
                int x = coord.second;

                set_pixel(output, x, y, r, g, b);
            }

            draw_rectangle(output, region.bbox, 255, 255, 255);
        }

        image::save_image(output, filename.c_str());
    }

    void save_elements_debug(const std::string& filename,
                             const image::rgb24_image& image,
                             const std::vector<Element>& elements)
    {
        image::rgb24_image output(image);

        for (size_t i = 0; i < elements.size(); i++)
        {
            uint8_t r, g, b;

            color_from_id(static_cast<int>(i + 1), r, g, b);

            draw_rectangle(output, elements[i].bbox, r, g, b);

            for (const Point& p : elements[i].corners)
            {
                draw_point(output, p, 255, 255, 255);
            }
        }

        image::save_image(output, filename.c_str());
    }

    void save_triplets_debug(const std::string& filename,
                             const image::rgb24_image& image,
                             const std::vector<Triplet>& triplets)
    {
        image::rgb24_image output(image);

        for (size_t i = 0; i < triplets.size(); i++)
        {
            uint8_t r, g, b;

            color_from_id(static_cast<int>(i + 50), r, g, b);

            draw_rectangle(output, triplets[i].a.bbox, r, g, b);

            draw_rectangle(output, triplets[i].b.bbox, r, g, b);

            draw_rectangle(output, triplets[i].c.bbox, r, g, b);

            for (const Point& p : triplets[i].a.corners)
                draw_point(output, p, r, g, b);

            for (const Point& p : triplets[i].b.corners)
                draw_point(output, p, r, g, b);

            for (const Point& p : triplets[i].c.corners)
                draw_point(output, p, r, g, b);
        }

        image::save_image(output, filename.c_str());
    }

    void save_qr_debug(const std::string& filename, image::rgb24_image image,
                       const std::vector<std::vector<Point>>& corners)
    {
        for (const auto& qr : corners)
        {
            draw_qr(image, qr);
        }

        image::save_image(image, filename.c_str());
    }

    void save_preprocess_debug(const std::string& folder, const Preprocess& pre)
    {
        std::filesystem::create_directories(folder);

        std::string path;

        path = folder + "/01_original.png";
        image::save_image(const_cast<image::rgb24_image&>(pre.image_originel),
                          path.c_str());

        path = folder + "/02_equalized.png";
        image::save_image(const_cast<image::rgb24_image&>(pre.image_equalized),
                          path.c_str());

        image::rgb24_image gray_rgb = image::gray_to_rgb(
            const_cast<image::gray8_image&>(pre.image_grayscale));

        path = folder + "/03_gray.png";
        image::save_image(gray_rgb, path.c_str());

        image::rgb24_image binary_rgb = image::gray_to_rgb(
            const_cast<image::gray8_image&>(pre.image_binary));

        path = folder + "/04_binary.png";
        image::save_image(binary_rgb, path.c_str());

        image::rgb24_image denoise_rgb = image::gray_to_rgb(
            const_cast<image::gray8_image&>(pre.image_denoise));

        path = folder + "/05_denoise.png";
        image::save_image(denoise_rgb, path.c_str());
    }

} // namespace qr_code