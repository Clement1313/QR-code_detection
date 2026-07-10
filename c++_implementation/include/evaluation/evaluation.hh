#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "detection/detection.hh"

namespace qr_code
{
    using BBox = std::tuple<double, double, double, double>;

    std::vector<std::vector<Point>> parse_ground_truth(
        const std::filesystem::path& txt_path);

    BBox corners_to_bbox(const std::vector<Point>& corners);

    std::vector<BBox> detect_qr(const std::string& image_path);

    double compute_iou(const BBox& box1, const BBox& box2);

    std::tuple<int, int, int> evaluate(const std::vector<BBox>& detections,
                                        const std::vector<BBox>& ground_truth,
                                        double iou_threshold = 0.5);

    double precision(int tp, int fp);
    double recall(int tp, int fn);
    double f1_score(int tp, int fp, int fn);

    double mean_iou(const std::vector<BBox>& detections,
                     const std::vector<BBox>& ground_truth);

    std::optional<std::filesystem::path> find_gt_file(
        const std::filesystem::path& image_path);

    struct EvalResult
    {
        double precision;
        double recall;
        double f1;
        double mean_iou;
    };

    EvalResult evaluate_folder(const std::string& folder_path,
                                double iou_threshold = 0.5,
                                bool verbose = true);

} // namespace qr_code

