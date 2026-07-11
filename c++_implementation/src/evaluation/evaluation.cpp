#include "evaluation/evaluation.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "debug/debug_visualization.hh"
#include "detection/preprocess.hh"
#include "util/image_io.hh"

#define DEBUG false

namespace
{
    std::string strip(const std::string& s)
    {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    const std::vector<std::string> kImageExtensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    };

    bool is_image_file(const std::filesystem::path& p)
    {
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return std::find(kImageExtensions.begin(), kImageExtensions.end(), ext)
            != kImageExtensions.end();
    }
} // namespace

namespace qr_code
{
    std::vector<std::vector<Point>>
    parse_ground_truth(const std::filesystem::path& txt_path)
    {
        std::vector<std::vector<Point>> qr_codes;
        std::vector<Point> current_pts;
        bool in_sets = false;

        std::ifstream file(txt_path);
        if (!file)
            throw std::runtime_error("Impossible d'ouvrir "
                                     + txt_path.string());

        std::string raw_line;
        while (std::getline(file, raw_line))
        {
            std::string line = strip(raw_line);
            if (line.empty() || line[0] == '#')
                continue;
            if (line == "SETS")
            {
                in_sets = true;
                continue;
            }

            std::istringstream iss(line);
            std::vector<double> values;
            double v;
            while (iss >> v)
                values.push_back(v);

            if (in_sets)
            {
                if (values.size() != 8)
                    throw std::runtime_error("Ligne invalide : " + line);
                std::vector<Point> pts;
                for (int i = 0; i < 8; i += 2)
                    pts.push_back({ values[i], values[i + 1] });
                qr_codes.push_back(std::move(pts));
            }
            else
            {
                if (values.size() != 2)
                    throw std::runtime_error("Ligne invalide : " + line);
                current_pts.push_back({ values[0], values[1] });
            }
        }

        if (!current_pts.empty())
        {
            if (current_pts.size() != 4)
                throw std::runtime_error("Le fichier contient "
                                         + std::to_string(current_pts.size())
                                         + " points au lieu de 4.");
            qr_codes.push_back(current_pts);
        }

        return qr_codes;
    }

    BBox corners_to_bbox(const std::vector<Point>& corners)
    {
        double min_row = corners[0].row, max_row = corners[0].row;
        double min_col = corners[0].col, max_col = corners[0].col;
        for (const Point& p : corners)
        {
            min_row = std::min(min_row, p.row);
            max_row = std::max(max_row, p.row);
            min_col = std::min(min_col, p.col);
            max_col = std::max(max_col, p.col);
        }
        return { min_row, min_col, max_row, max_col };
    }

    std::vector<BBox> detect_qr_old(const std::string& image_path)
    {
        image::rgb24_image* image = image::load_image(image_path.c_str());
        if (!image)
            throw std::runtime_error("Impossible de charger " + image_path);

        std::shared_ptr<Preprocess> pre = preprocess(*image);

        const image::gray8_image& binary = pre->image_binary;
        const image::gray8_image& denoise = pre->image_denoise;

        LabelImage lab = labels(denoise);
        std::vector<Region> regions = regionprops(lab);

        std::vector<Element> elements;
        for (const Region& region : regions)
        {
            std::vector<Point> corners;
            bool ok = square_filter(denoise, binary, region, corners);
            if (ok)
                elements.push_back({ region.bbox, corners, region.area });
        }

        std::vector<Triplet> triplets =
            get_triplets(elements, image->sx, image->sy);

        std::vector<BBox> detections;
        for (const Triplet& triplet : triplets)
        {
            std::vector<Point> qr_corners = get_qr_corners(triplet);
            if (qr_corners.empty())
                continue;

            draw_qr(*image, qr_corners);

            detections.push_back(corners_to_bbox(qr_corners));
        }

        std::filesystem::path input(image_path);
        std::filesystem::create_directories("results");

        std::filesystem::path output =
            std::filesystem::path("results") / input.filename();

        output.replace_extension(".tga");

        save_image(*image, output.string().c_str());

        delete image;
        return detections;
    }

    std::vector<BBox> detect_qr_debug(const std::string& image_path)
    {
        image::rgb24_image* image = image::load_image(image_path.c_str());

        if (!image)
            throw std::runtime_error("Impossible de charger " + image_path);

        std::shared_ptr<Preprocess> pre = preprocess(*image);
        std::string debug_folder =
            "debug/" + std::filesystem::path(image_path).stem().string();
        qr_code::save_preprocess_debug(debug_folder, *pre);
        const image::gray8_image& binary = pre->image_binary;
        const image::gray8_image& denoise = pre->image_denoise;

        LabelImage lab = labels(denoise);
        save_labels_debug(debug_folder + "/06_labels.png", lab);

        std::vector<Region> regions = regionprops(lab);
        save_regions_debug(debug_folder + "/07_regions.png", lab, regions);

        std::vector<Element> elements;
        for (const Region& region : regions)
        {
            std::vector<Point> corners;
            bool ok = square_filter(denoise, binary, region, corners);
            if (ok)
            {
                elements.push_back({ region.bbox, corners, region.area });
            }
        }
        save_elements_debug(debug_folder + "/08_elements.png", *image,
                            elements);

        std::vector<Triplet> triplets =
            get_triplets(elements, image->sx, image->sy);
        save_triplets_debug(debug_folder + "/09_triplets.png", *image,
                            triplets);

        std::vector<BBox> detections;
        std::vector<std::vector<Point>> debug_corners;
        for (const Triplet& triplet : triplets)
        {
            std::vector<Point> qr_corners = get_qr_corners(triplet);
            if (qr_corners.empty())
                continue;
            detections.push_back(corners_to_bbox(qr_corners));
            debug_corners.push_back(qr_corners);
        }

        save_qr_debug(debug_folder + "/10_qr.png", *image, debug_corners);
        delete image;
        return detections;
    }

    std::vector<BBox> detect_qr(const std::string& image_path)
    {
        if (DEBUG)
        {
            return detect_qr_debug(image_path);
        }
        else
        {
            return detect_qr_old(image_path);
        }
    }

    double compute_iou(const BBox& box1, const BBox& box2)
    {
        auto [aminr, aminc, amaxr, amaxc] = box1;
        auto [bminr, bminc, bmaxr, bmaxc] = box2;

        double xA = std::max(aminc, bminc);
        double yA = std::max(aminr, bminr);
        double xB = std::min(amaxc, bmaxc);
        double yB = std::min(amaxr, bmaxr);

        double inter = std::max(0.0, xB - xA) * std::max(0.0, yB - yA);
        double areaA = (amaxc - aminc) * (amaxr - aminr);
        double areaB = (bmaxc - bminc) * (bmaxr - bminr);
        double uni = areaA + areaB - inter;

        if (uni == 0.0)
            return 0.0;
        return inter / uni;
    }

    std::tuple<int, int, int> evaluate(const std::vector<BBox>& detections,
                                       const std::vector<BBox>& ground_truth,
                                       double iou_threshold)
    {
        std::vector<bool> matched_gt(ground_truth.size(), false);
        int tp = 0;
        int fp = 0;

        for (const BBox& det : detections)
        {
            double best_iou = 0.0;
            int best_gt = -1;
            for (size_t i = 0; i < ground_truth.size(); i++)
            {
                if (matched_gt[i])
                    continue;
                double iou = compute_iou(det, ground_truth[i]);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_gt = static_cast<int>(i);
                }
            }
            if (best_iou >= iou_threshold)
            {
                tp++;
                matched_gt[best_gt] = true;
            }
            else
            {
                fp++;
            }
        }

        int matched_count = static_cast<int>(
            std::count(matched_gt.begin(), matched_gt.end(), true));
        int fn = static_cast<int>(ground_truth.size()) - matched_count;

        return { tp, fp, fn };
    }

    double precision(int tp, int fp)
    {
        return (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
    }

    double recall(int tp, int fn)
    {
        return (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
    }

    double f1_score(int tp, int fp, int fn)
    {
        double p = precision(tp, fp);
        double r = recall(tp, fn);
        return (p + r) > 0.0 ? 2 * p * r / (p + r) : 0.0;
    }

    double mean_iou(const std::vector<BBox>& detections,
                    const std::vector<BBox>& ground_truth)
    {
        std::vector<bool> used(ground_truth.size(), false);
        std::vector<double> scores;

        for (const BBox& det : detections)
        {
            double best = 0.0;
            int best_gt = -1;
            for (size_t i = 0; i < ground_truth.size(); i++)
            {
                if (used[i])
                    continue;
                double iou = compute_iou(det, ground_truth[i]);
                if (iou > best)
                {
                    best = iou;
                    best_gt = static_cast<int>(i);
                }
            }
            if (best_gt != -1)
            {
                used[best_gt] = true;
                scores.push_back(best);
            }
        }

        if (scores.empty())
            return 0.0;
        return std::accumulate(scores.begin(), scores.end(), 0.0)
            / static_cast<double>(scores.size());
    }

    std::optional<std::filesystem::path>
    find_gt_file(const std::filesystem::path& image_path)
    {
        std::vector<std::filesystem::path> candidates = {
            std::filesystem::path(image_path).replace_extension(".txt"),
            image_path.parent_path() / (image_path.stem().string() + ".txt")
        };
        for (const std::filesystem::path& c : candidates)
        {
            if (std::filesystem::exists(c))
                return c;
        }
        return std::nullopt;
    }

    EvalResult evaluate_folder(const std::string& folder_path,
                               double iou_threshold, bool verbose)
    {
        std::filesystem::path folder(folder_path);

        int total_tp = 0, total_fp = 0, total_fn = 0;
        std::vector<double> all_ious;
        int processed = 0;
        int skipped = 0;

        std::vector<std::filesystem::path> entries;
        for (const auto& entry : std::filesystem::directory_iterator(folder))
            entries.push_back(entry.path());
        std::sort(entries.begin(), entries.end());

        for (const std::filesystem::path& image_file : entries)
        {
            if (!is_image_file(image_file))
                continue;

            std::optional<std::filesystem::path> gt_file =
                find_gt_file(image_file);
            if (!gt_file)
            {
                if (verbose)
                    std::cout << "  [SKIP] Pas de verite terrain pour "
                              << image_file.filename().string() << "\n";
                skipped++;
                continue;
            }

            std::vector<std::vector<Point>> gt_qr_list =
                parse_ground_truth(*gt_file);
            std::vector<BBox> gt_bboxes;
            for (const std::vector<Point>& qr : gt_qr_list)
                gt_bboxes.push_back(corners_to_bbox(qr));

            std::vector<BBox> detections;
            try
            {
                detections = detect_qr(image_file.string());
            }
            catch (const std::exception& e)
            {
                std::cout << "  [ERREUR] " << image_file.filename().string()
                          << " : " << e.what() << "\n";
                skipped++;
                continue;
            }

            auto [tp, fp, fn] = evaluate(detections, gt_bboxes, iou_threshold);
            double miou = mean_iou(detections, gt_bboxes);

            total_tp += tp;
            total_fp += fp;
            total_fn += fn;
            all_ious.push_back(miou);
            processed++;

            if (verbose)
            {
                double p = precision(tp, fp);
                double r = recall(tp, fn);
                std::cout << "  " << image_file.filename().string()
                          << " | det=" << detections.size()
                          << " gt=" << gt_bboxes.size() << " | TP=" << tp
                          << " FP=" << fp << " FN=" << fn << " | P=" << p
                          << " R=" << r << " IoU=" << miou << "\n";
            }
        }

        double global_p = precision(total_tp, total_fp);
        double global_r = recall(total_tp, total_fn);
        double global_f1 = f1_score(total_tp, total_fp, total_fn);
        double global_iou = all_ious.empty()
            ? 0.0
            : std::accumulate(all_ious.begin(), all_ious.end(), 0.0)
                / static_cast<double>(all_ious.size());

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Dossier : " << folder.string() << "\n";
        std::cout << "Images traitees : " << processed
                  << "  |  Ignorees : " << skipped << "\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "Precision globale  : " << global_p << "\n";
        std::cout << "Rappel global      : " << global_r << "\n";
        std::cout << "F1 global          : " << global_f1 << "\n";
        std::cout << "Mean IoU global    : " << global_iou << "\n";
        std::cout << std::string(60, '=') << "\n";

        return { global_p, global_r, global_f1, global_iou };
    }

} // namespace qr_code