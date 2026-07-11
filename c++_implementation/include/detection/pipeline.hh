#pragma once

#include <string>
#include <vector>

#include "detection/detection.hh"

namespace qr_code
{
    std::vector<std::vector<Point>> process_directory(
        const std::string& image_path);

} // namespace qr_code