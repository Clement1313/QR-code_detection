#include "detection/detection.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace qr_code
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;

        LabelImage label_image(const image::gray8_image& image,
                               int connectivity)
        {
            int w = image.sx, h = image.sy;
            LabelImage result{
                w, h, std::vector<int32_t>(static_cast<size_t>(w) * h, 0)
            };
            const uint8_t* buf = image.get_buffer();

            static const int n4y[] = { -1, 1, 0, 0 };
            static const int n4x[] = { 0, 0, -1, 1 };
            static const int n8y[] = { -1, 1, 0, 0, -1, -1, 1, 1 };
            static const int n8x[] = { 0, 0, -1, 1, -1, 1, -1, 1 };

            const int* ny = (connectivity == 8) ? n8y : n4y;
            const int* nx = (connectivity == 8) ? n8x : n4x;
            int n_neighbors = (connectivity == 8) ? 8 : 4;

            int32_t current_label = 0;
            std::vector<std::pair<int, int>> stack;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    if (buf[y * w + x] == 0 || result.at(x, y) != 0)
                        continue;

                    current_label++;
                    stack.clear();
                    stack.push_back({ x, y });
                    result.at(x, y) = current_label;

                    while (!stack.empty())
                    {
                        auto [cx, cy] = stack.back();
                        stack.pop_back();

                        for (int k = 0; k < n_neighbors; k++)
                        {
                            int nnx = cx + nx[k], nny = cy + ny[k];
                            if (nnx < 0 || nnx >= w || nny < 0 || nny >= h)
                                continue;
                            if (buf[nny * w + nnx] == 0)
                                continue;
                            if (result.at(nnx, nny) != 0)
                                continue;
                            result.at(nnx, nny) = current_label;
                            stack.push_back({ nnx, nny });
                        }
                    }
                }
            }
            return result;
        }

        image::gray8_image crop_image(const image::gray8_image& src,
                                      const std::array<int, 4>& bbox)
        {
            int minr = bbox[0], minc = bbox[1], maxr = bbox[2], maxc = bbox[3];
            int h = std::max(0, maxr - minr);
            int w = std::max(0, maxc - minc);
            image::gray8_image out{ std::max(1, w), std::max(1, h) };

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    out.get_buffer()[y * out.sx + x] =
                        src.get_buffer()[(minr + y) * src.sx + (minc + x)];

            return out;
        }

        image::gray8_image get_inner_square(const image::gray8_image& square)
        {
            LabelImage lab =
                label_image(square, 8); // connexité par défaut skimage
            std::vector<Region> regions = regionprops(lab);

            image::gray8_image mask{ square.sx, square.sy };
            std::fill(mask.get_buffer(),
                      mask.get_buffer() + square.sx * square.sy, 0);

            if (regions.empty())
                return mask;

            const Region* best = &regions[0];
            for (const Region& r : regions)
                if (r.area > best->area)
                    best = &r;

            for (const auto& [row, col] : best->coords)
                mask.get_buffer()[row * square.sx + col] = 255;

            return mask;
        }

        bool is_foreground(const image::gray8_image& mask, int y, int x)
        {
            if (x < 0 || x >= mask.sx || y < 0 || y >= mask.sy)
                return false;
            return mask.get_buffer()[y * mask.sx + x] != 0;
        }

        std::vector<Point> trace_boundary(const image::gray8_image& mask)
        {
            static const int off_dy[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
            static const int off_dx[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

            auto find_offset_index = [](int dy, int dx) {
                for (int i = 0; i < 8; i++)
                    if (off_dy[i] == dy && off_dx[i] == dx)
                        return i;
                return 0;
            };

            std::vector<Point> boundary;
            int start_y = -1, start_x = -1;
            for (int y = 0; y < mask.sy && start_y < 0; y++)
                for (int x = 0; x < mask.sx; x++)
                    if (is_foreground(mask, y, x))
                    {
                        start_y = y;
                        start_x = x;
                        break;
                    }

            if (start_y < 0)
                return boundary;

            int cur_y = start_y, cur_x = start_x;
            int back_y = cur_y, back_x = cur_x - 1;

            bool has_neighbor = false;
            for (int i = 0; i < 8; i++)
                if (is_foreground(mask, cur_y + off_dy[i], cur_x + off_dx[i]))
                {
                    has_neighbor = true;
                    break;
                }

            boundary.push_back(
                { static_cast<double>(cur_y), static_cast<double>(cur_x) });

            if (!has_neighbor)
                return boundary; // pixel isolé

            int first_y = cur_y, first_x = cur_x;
            size_t max_iter = static_cast<size_t>(mask.sx) * mask.sy * 8 + 8;

            for (size_t iter = 0; iter < max_iter; iter++)
            {
                int idx = find_offset_index(back_y - cur_y, back_x - cur_x);

                int found_y = -1, found_x = -1;
                int last_bg_y = back_y, last_bg_x = back_x;

                for (int s = 1; s <= 8; s++)
                {
                    int k = (idx + s) % 8;
                    int ny = cur_y + off_dy[k], nx = cur_x + off_dx[k];
                    if (is_foreground(mask, ny, nx))
                    {
                        found_y = ny;
                        found_x = nx;
                        break;
                    }
                    last_bg_y = ny;
                    last_bg_x = nx;
                }

                if (found_y < 0)
                    break;

                cur_y = found_y;
                cur_x = found_x;
                back_y = last_bg_y;
                back_x = last_bg_x;

                if (cur_y == first_y && cur_x == first_x)
                    break;

                boundary.push_back(
                    { static_cast<double>(cur_y), static_cast<double>(cur_x) });
            }

            return boundary;
        }

        double perpendicular_distance(const Point& p, const Point& a,
                                      const Point& b)
        {
            double dy = b.row - a.row, dx = b.col - a.col;
            double norm = std::hypot(dy, dx);
            if (norm == 0.0)
                return std::hypot(p.row - a.row, p.col - a.col);
            double num = std::abs(dy * (p.col - a.col) - dx * (p.row - a.row));
            return num / norm;
        }

        std::vector<Point> douglas_peucker(const std::vector<Point>& points,
                                           double tolerance)
        {
            if (points.size() < 3)
                return points;

            double max_dist = 0.0;
            size_t index = 0;
            for (size_t i = 1; i + 1 < points.size(); i++)
            {
                double d = perpendicular_distance(points[i], points.front(),
                                                  points.back());
                if (d > max_dist)
                {
                    max_dist = d;
                    index = i;
                }
            }

            if (max_dist > tolerance)
            {
                std::vector<Point> left(points.begin(),
                                        points.begin() + index + 1);
                std::vector<Point> right(points.begin() + index, points.end());
                std::vector<Point> res_left = douglas_peucker(left, tolerance);
                std::vector<Point> res_right =
                    douglas_peucker(right, tolerance);
                res_left.pop_back();
                res_left.insert(res_left.end(), res_right.begin(),
                                res_right.end());
                return res_left;
            }
            return { points.front(), points.back() };
        }

        std::vector<Point> get_corner(const image::gray8_image& mask)
        {
            std::vector<Point> contour = trace_boundary(mask);
            if (contour.empty())
                return {};

            std::vector<Point> poly = douglas_peucker(contour, 6.0);
            if (poly.size() >= 2)
            {
                const Point& first = poly.front();
                const Point& last = poly.back();
                if (std::abs(first.row - last.row) < 1e-6
                    && std::abs(first.col - last.col) < 1e-6)
                    poly.pop_back();
            }
            return poly;
        }

        std::vector<Point> convex_hull(std::vector<Point> pts)
        {
            if (pts.size() < 3)
                return pts;

            std::sort(pts.begin(), pts.end(),
                      [](const Point& a, const Point& b) {
                          if (a.row != b.row)
                              return a.row < b.row;
                          return a.col < b.col;
                      });
            pts.erase(std::unique(pts.begin(), pts.end(),
                                  [](const Point& a, const Point& b) {
                                      return std::abs(a.row - b.row) < 1e-9
                                          && std::abs(a.col - b.col) < 1e-9;
                                  }),
                      pts.end());
            if (pts.size() < 3)
                return pts;

            auto cross = [](const Point& o, const Point& a, const Point& b) {
                return (a.col - o.col) * (b.row - o.row)
                    - (a.row - o.row) * (b.col - o.col);
            };

            std::vector<Point> hull(2 * pts.size());
            int k = 0;
            for (size_t i = 0; i < pts.size(); i++)
            {
                while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0)
                    k--;
                hull[k++] = pts[i];
            }
            int lower = k + 1;
            for (int i = static_cast<int>(pts.size()) - 2; i >= 0; i--)
            {
                while (k >= lower
                       && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0)
                    k--;
                hull[k++] = pts[i];
            }
            hull.resize(k - 1);
            return hull;
        }

        std::vector<Point> order_points(const std::vector<Point>& coords)
        {
            if (coords.empty())
                return coords;

            double mean_row = 0, mean_col = 0;
            for (const Point& p : coords)
            {
                mean_row += p.row;
                mean_col += p.col;
            }
            mean_row /= coords.size();
            mean_col /= coords.size();

            std::vector<Point> sorted_coords = coords;
            std::sort(sorted_coords.begin(), sorted_coords.end(),
                      [&](const Point& a, const Point& b) {
                          double angle_a =
                              std::atan2(a.row - mean_row, a.col - mean_col);
                          double angle_b =
                              std::atan2(b.row - mean_row, b.col - mean_col);
                          return angle_a < angle_b;
                      });
            return sorted_coords;
        }

        std::vector<Point> get_four_coord(std::vector<Point> coords)
        {
            if (coords.size() < 4)
                return coords;

            if (coords.size() > 4)
            {
                std::vector<Point> hull = convex_hull(coords);
                if (hull.size() >= 3)
                    coords = hull;
            }

            if (coords.size() == 4)
                return order_points(coords);

            size_t n = coords.size();
            if (n < 4)
                return {};

            std::vector<Point> best;
            double best_score = std::numeric_limits<double>::infinity();

            std::vector<bool> mask(n, false);
            std::fill(mask.end() - 4, mask.end(), true);
            do
            {
                std::vector<Point> subset;
                for (size_t i = 0; i < n; i++)
                    if (mask[i])
                        subset.push_back(coords[i]);

                std::vector<Point> ordered = order_points(subset);
                double sides[4];
                for (int i = 0; i < 4; i++)
                    sides[i] =
                        std::hypot(ordered[i].row - ordered[(i + 1) % 4].row,
                                   ordered[i].col - ordered[(i + 1) % 4].col);

                double diag1 = std::hypot(ordered[0].row - ordered[2].row,
                                          ordered[0].col - ordered[2].col);
                double diag2 = std::hypot(ordered[1].row - ordered[3].row,
                                          ordered[1].col - ordered[3].col);

                double mean_side =
                    (sides[0] + sides[1] + sides[2] + sides[3]) / 4.0;
                double var = 0;
                for (double s : sides)
                    var += (s - mean_side) * (s - mean_side);
                double score = std::sqrt(var / 4.0) + std::abs(diag1 - diag2);

                if (score < best_score)
                {
                    best_score = score;
                    best = ordered;
                }
            } while (std::next_permutation(mask.begin(), mask.end()));

            return best;
        }


        bool check_side_ratio(const std::vector<Point>& coords,
                              double min_ratio = 0.35)
        {
            double sides[4];
            for (int i = 0; i < 4; i++)
                sides[i] = std::hypot(coords[i].row - coords[(i + 1) % 4].row,
                                      coords[i].col - coords[(i + 1) % 4].col);
            double mn = *std::min_element(sides, sides + 4);
            double mx = *std::max_element(sides, sides + 4);
            if (mx == 0.0)
                return false;
            return (mn / mx) >= min_ratio;
        }

        bool check_form(const std::vector<Point>& coords)
        {
            return coords.size() == 4 && is_convex_quad(coords)
                && check_side_ratio(coords);
        }

        bool check_region_shape(const Region& region,
                                double max_eccentricity = 0.6)
        {
            return region.eccentricity <= max_eccentricity;
        }



        bool check_area(long a1, long a2, double ratio = 1.6)
        {
            if (a1 <= 0 || a2 <= 0)
                return false;
            double mx = static_cast<double>(std::max(a1, a2));
            double mn = static_cast<double>(std::min(a1, a2));
            return (mx / mn) < ratio;
        }

        bool similar_area(const Element& m1, const Element& m2,
                          const Element& m3)
        {
            return check_area(m1.area, m2.area) && check_area(m1.area, m3.area)
                && check_area(m2.area, m3.area);
        }

        std::optional<double> marker_size_from_corners(const Element& element)
        {
            if (element.corners.size() != 4)
                return std::nullopt;
            double sum = 0;
            for (int i = 0; i < 4; i++)
                sum += std::hypot(
                    element.corners[i].row - element.corners[(i + 1) % 4].row,
                    element.corners[i].col - element.corners[(i + 1) % 4].col);
            return sum / 4.0;
        }

        double estimate_max_distance(const std::vector<Element>& elements,
                                     int image_sx, int image_sy,
                                     double factor = 8.0)
        {
            double min_dim = std::min(image_sx, image_sy);
            if (elements.empty())
                return min_dim * 0.5;

            std::vector<double> sizes;
            for (const Element& e : elements)
            {
                double h = e.bbox[2] - e.bbox[0];
                double w = e.bbox[3] - e.bbox[1];
                sizes.push_back((h + w) / 2.0);
            }
            std::sort(sizes.begin(), sizes.end());
            double median_size = (sizes.size() % 2 == 1)
                ? sizes[sizes.size() / 2]
                : (sizes[sizes.size() / 2 - 1] + sizes[sizes.size() / 2]) / 2.0;

            return std::clamp(median_size * factor, 50.0, min_dim * 0.8);
        }

        double get_triplet_score(const std::array<Point, 3>& centers,
                                 const std::array<const Element*, 3>& elements)
        {
            const Point& A = centers[0];
            const Point& B = centers[1];
            const Point& C = centers[2];

            double angles[3] = { angle(B, A, C), angle(A, B, C),
                                 angle(A, C, B) };
            int idx = 0;
            double best_dev = std::abs(angles[0] - 90.0);
            for (int i = 1; i < 3; i++)
            {
                double dev = std::abs(angles[i] - 90.0);
                if (dev < best_dev)
                {
                    best_dev = dev;
                    idx = i;
                }
            }
            double right_angle_dev = std::abs(angles[idx] - 90.0) / 90.0;

            std::array<double, 3> d = {
                std::hypot(A.row - B.row, A.col - B.col),
                std::hypot(A.row - C.row, A.col - C.col),
                std::hypot(B.row - C.row, B.col - C.col)
            };
            std::sort(d.begin(), d.end());
            double side1 = d[0], side2 = d[1], diag = d[2];

            double side_dev = (std::max(side1, side2) > 0)
                ? std::abs(side1 - side2) / std::max(side1, side2)
                : 0.0;
            double expected_diag = std::sqrt(2.0) * (side1 + side2) / 2.0;
            double diag_dev =
                (diag > 0) ? std::abs(diag - expected_diag) / diag : 0.0;

            std::array<std::optional<double>, 3> sizes = {
                marker_size_from_corners(*elements[0]),
                marker_size_from_corners(*elements[1]),
                marker_size_from_corners(*elements[2])
            };
            double size_dev = 0.0;
            if (sizes[0] && sizes[1] && sizes[2])
            {
                double mn = std::min({ *sizes[0], *sizes[1], *sizes[2] });
                double mx = std::max({ *sizes[0], *sizes[1], *sizes[2] });
                if (mx > 0)
                    size_dev = (mx - mn) / mx;
            }

            return right_angle_dev + side_dev + diag_dev + size_dev;
        }



        bool inside(const Point& p, const Point& a, const Point& b)
        {
            return (b.col - a.col) * (p.row - a.row)
                - (b.row - a.row) * (p.col - a.col)
                >= 0;
        }

        Point intersect_edges(const Point& p1, const Point& p2, const Point& a,
                              const Point& b)
        {
            double A1 = b.col - a.col;
            double B1 = a.row - b.row;
            double C1 = A1 * a.row + B1 * a.col;
            double A2 = p2.col - p1.col;
            double B2 = p1.row - p2.row;
            double C2 = A2 * p1.row + B2 * p1.col;
            double det = A1 * B2 - A2 * B1;
            if (std::abs(det) < 1e-10)
                return p2;
            double row = (B2 * C1 - B1 * C2) / det;
            double col = (A1 * C2 - A2 * C1) / det;
            return { row, col };
        }

        std::vector<Point> clip_polygon(const std::vector<Point>& subject,
                                        const std::vector<Point>& clip_poly)
        {
            std::vector<Point> output = subject;
            size_t n_clip = clip_poly.size();

            for (size_t i = 0; i < n_clip && !output.empty(); i++)
            {
                const Point& a = clip_poly[i];
                const Point& b = clip_poly[(i + 1) % n_clip];

                std::vector<Point> input_list = output;
                output.clear();
                size_t n_in = input_list.size();

                for (size_t j = 0; j < n_in; j++)
                {
                    const Point& curr = input_list[j];
                    const Point& prev = input_list[(j == 0) ? n_in - 1 : j - 1];

                    bool curr_in = inside(curr, a, b);
                    bool prev_in = inside(prev, a, b);

                    if (curr_in)
                    {
                        if (!prev_in)
                            output.push_back(intersect_edges(prev, curr, a, b));
                        output.push_back(curr);
                    }
                    else if (prev_in)
                    {
                        output.push_back(intersect_edges(prev, curr, a, b));
                    }
                }
            }
            return output;
        }

        bool should_remove_smaller(const std::vector<Point>& outer_corners,
                                   const std::vector<Point>& inner_corners,
                                   double size_tolerance,
                                   double overlap_threshold)
        {
            double outer_area = polygon_area(outer_corners);
            double inner_area = polygon_area(inner_corners);

            if (inner_area == 0.0 || inner_area >= outer_area * size_tolerance)
                return false;

            std::vector<Point> intersection =
                clip_polygon(inner_corners, outer_corners);
            if (intersection.size() < 3)
                return false;

            double intersection_area = polygon_area(intersection);
            double overlap_ratio = intersection_area / inner_area;
            return overlap_ratio >= overlap_threshold;
        }

        void draw_line(image::rgb24_image& image, int r0, int c0, int r1,
                       int c1)
        {
            int dr = std::abs(r1 - r0), dc = std::abs(c1 - c0);
            int sr = (r0 < r1) ? 1 : -1;
            int sc = (c0 < c1) ? 1 : -1;
            int err = dr - dc;
            int r = r0, c = c0;

            while (true)
            {
                if (r >= 0 && r < image.sy && c >= 0 && c < image.sx)
                {
                    int off = (r * image.sx + c) * 3;
                    image.get_buffer()[off] = 255;
                    image.get_buffer()[off + 1] = 0;
                    image.get_buffer()[off + 2] = 0;
                }
                if (r == r1 && c == c1)
                    break;
                int e2 = 2 * err;
                if (e2 > -dc)
                {
                    err -= dc;
                    r += sr;
                }
                if (e2 < dr)
                {
                    err += dr;
                    c += sc;
                }
            }
        }

        bool check_finder_pattern_ratio(const image::gray8_image& binary,
                                        const std::array<int, 4>& bbox,
                                        double tolerance = 0.5,
                                        double expand_factor = 1.8)
        {
            int minr = bbox[0], minc = bbox[1], maxr = bbox[2], maxc = bbox[3];
            int cy = (minr + maxr) / 2;
            int cx = (minc + maxc) / 2;

            int half_h = static_cast<int>((maxr - minr) * expand_factor / 2.0);
            int half_w = static_cast<int>((maxc - minc) * expand_factor / 2.0);

            int scan_minr = std::max(0, cy - half_h);
            int scan_maxr = std::min(binary.sy, cy + half_h);
            int scan_minc = std::max(0, cx - half_w);
            int scan_maxc = std::min(binary.sx, cx + half_w);

            auto scan_line = [&](bool horizontal) {
                std::vector<int> runs;
                int prev = -1;
                int run_len = 0;
                int start = horizontal ? scan_minc : scan_minr;
                int end = horizontal ? scan_maxc : scan_maxr;

                for (int p = start; p < end; p++)
                {
                    int val = horizontal
                        ? binary.get_buffer()[cy * binary.sx + p]
                        : binary.get_buffer()[p * binary.sx + cx];
                    int bit = (val > 127) ? 1 : 0;

                    if (bit == prev)
                    {
                        run_len++;
                    }
                    else
                    {
                        if (prev != -1)
                            runs.push_back(run_len);
                        prev = bit;
                        run_len = 1;
                    }
                }
                if (prev != -1)
                    runs.push_back(run_len);

                if (runs.size() < 5)
                    return false;

                for (size_t i = 0; i + 5 <= runs.size(); i++)
                {
                    double unit =
                        (runs[i] + runs[i + 1] + runs[i + 3] + runs[i + 4])
                        / 4.0;
                    if (unit <= 0.0)
                        continue;

                    double expected[5] = { unit, unit, 3.0 * unit, unit, unit };
                    bool ok = true;
                    for (int k = 0; k < 5; k++)
                    {
                        double diff =
                            std::abs(runs[i + k] - expected[k]) / expected[k];
                        if (diff > tolerance)
                        {
                            ok = false;
                            break;
                        }
                    }
                    if (ok)
                        return true;
                }
                return false;
            };

            return scan_line(true) && scan_line(false);
        }

    } // namespace



    bool is_convex_quad(const std::vector<Point>& coords)
    {
        int sum_signs = 0;
        for (int i = 0; i < 4; i++)
        {
            const Point& p0 = coords[(i + 3) % 4];
            const Point& p1 = coords[i];
            const Point& p2 = coords[(i + 1) % 4];
            double v1_row = p1.row - p0.row, v1_col = p1.col - p0.col;
            double v2_row = p2.row - p1.row, v2_col = p2.col - p1.col;
            double cross = v1_row * v2_col - v1_col * v2_row;
            sum_signs += (cross > 0) - (cross < 0);
        }
        return std::abs(sum_signs) == 4;
    }

double angle(const Point& a, const Point& b, const Point& c)
    {
        double ba_row = a.row - b.row, ba_col = a.col - b.col;
        double bc_row = c.row - b.row, bc_col = c.col - b.col;
        double na = std::hypot(ba_row, ba_col);
        double nc = std::hypot(bc_row, bc_col);
        if (na == 0.0 || nc == 0.0)
            return 0.0;
        ba_row /= na;
        ba_col /= na;
        bc_row /= nc;
        bc_col /= nc;
        double dot =
            std::clamp(ba_row * bc_row + ba_col * bc_col, -1.0, 1.0);
        return std::acos(dot) * 180.0 / kPi;
    }

    double polygon_area(const std::vector<Point>& corners)
    {
        double sum1 = 0, sum2 = 0;
        size_t n = corners.size();
        for (size_t i = 0; i < n; i++)
        {
            size_t prev = (i == 0) ? n - 1 : i - 1;
            sum1 += corners[i].col * corners[prev].row;
            sum2 += corners[i].row * corners[prev].col;
        }
        return 0.5 * std::abs(sum1 - sum2);
    }

    LabelImage labels(const image::gray8_image& denoise)
    {
        return label_image(denoise, 4);
    }

    std::vector<Region> regionprops(const LabelImage& lab)
    {
        int max_label = 0;
        for (int32_t v : lab.data)
            max_label = std::max(max_label, static_cast<int>(v));
        if (max_label == 0)
            return {};

        struct Accum
        {
            int minr = std::numeric_limits<int>::max();
            int minc = std::numeric_limits<int>::max();
            int maxr = std::numeric_limits<int>::min();
            int maxc = std::numeric_limits<int>::min();
            long area = 0;
            double sum_r = 0, sum_c = 0;
            std::vector<std::pair<int, int>> coords;
        };

        std::vector<Accum> accums(max_label + 1);

        for (int y = 0; y < lab.sy; y++)
        {
            for (int x = 0; x < lab.sx; x++)
            {
                int32_t l = lab.at(x, y);
                if (l == 0)
                    continue;
                Accum& a = accums[l];
                a.minr = std::min(a.minr, y);
                a.maxr = std::max(a.maxr, y);
                a.minc = std::min(a.minc, x);
                a.maxc = std::max(a.maxc, x);
                a.area++;
                a.sum_r += y;
                a.sum_c += x;
                a.coords.push_back({ y, x });
            }
        }

        std::vector<Region> regions;
        for (int l = 1; l <= max_label; l++)
        {
            Accum& a = accums[l];
            if (a.area == 0)
                continue;

            double mean_r = a.sum_r / a.area;
            double mean_c = a.sum_c / a.area;

            double mu20 = 0, mu02 = 0, mu11 = 0;
            for (const auto& [row, col] : a.coords)
            {
                double dr = row - mean_r;
                double dc = col - mean_c;
                mu20 += dr * dr;
                mu02 += dc * dc;
                mu11 += dr * dc;
            }
            mu20 /= a.area;
            mu02 /= a.area;
            mu11 /= a.area;

            double common = std::sqrt(
                std::max(0.0, (mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11));
            double major =
                std::sqrt(std::max(0.0, 8.0 * (mu20 + mu02 + common)));
            double minor =
                std::sqrt(std::max(0.0, 8.0 * (mu20 + mu02 - common)));

            double eccentricity = 0.0;
            if (major > 1e-9)
            {
                double ratio = minor / major;
                eccentricity = std::sqrt(std::max(0.0, 1.0 - ratio * ratio));
            }

            Region region;
            region.label = l;
            region.bbox = { a.minr, a.minc, a.maxr + 1, a.maxc + 1 };
            region.area = a.area;
            region.coords = std::move(a.coords);
            region.eccentricity = eccentricity;
            regions.push_back(std::move(region));
        }

        return regions;
    }

    bool square_filter(const image::gray8_image& denoise,
                       const image::gray8_image& binary, const Region& region,
                       std::vector<Point>& corners_out)
    {
        corners_out.clear();

        if (region.area < 50)
            return false;

        long image_area = static_cast<long>(denoise.sx) * denoise.sy;
        if (region.area > image_area / 4)
            return false;

        int region_h = region.bbox[2] - region.bbox[0];
        int region_w = region.bbox[3] - region.bbox[1];
        if (region_h > denoise.sy / 2 || region_w > denoise.sx / 2)
            return false;

        image::gray8_image square1 = crop_image(denoise, region.bbox);
        image::gray8_image square_im = get_inner_square(square1);
        std::vector<Point> coords = get_corner(square_im);

        if (coords.size() < 4)
        {
            corners_out = coords;
            return false;
        }

        coords = get_four_coord(coords);
        if (coords.size() != 4)
            return false;

        coords = order_points(coords);

        if (!check_region_shape(region))
            return false;

        if (!check_form(coords))
            return false;
        if (!check_finder_pattern_ratio(binary, region.bbox))
            return false;
        bool ok = check_form(coords);
        corners_out = coords;
        return true;
    }

    Point get_center(const std::array<int, 4>& bbox)
    {
        return { (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0 };
    }

    std::vector<Triplet> get_triplets(const std::vector<Element>& elements,
                                      int image_sx, int image_sy,
                                      double angle_tolerance)
    {
        struct Candidate
        {
            double score;
            std::array<size_t, 3> idx;
        };

        std::vector<Candidate> candidates;
        size_t n = elements.size();

        (void)estimate_max_distance(elements, image_sx, image_sy);

        for (size_t i = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++)
                for (size_t k = j + 1; k < n; k++)
                {
                    const Element& m1 = elements[i];
                    const Element& m2 = elements[j];
                    const Element& m3 = elements[k];

                    if (!similar_area(m1, m2, m3))
                        continue;

                    std::array<Point, 3> centers = { get_center(m1.bbox),
                                                     get_center(m2.bbox),
                                                     get_center(m3.bbox) };

                    double angles[3] = {
                        angle(centers[1], centers[0], centers[2]),
                        angle(centers[0], centers[1], centers[2]),
                        angle(centers[0], centers[2], centers[1])
                    };
                    int idx = 0;
                    double best_dev = std::abs(angles[0] - 90.0);
                    for (int a = 1; a < 3; a++)
                    {
                        double dev = std::abs(angles[a] - 90.0);
                        if (dev < best_dev)
                        {
                            best_dev = dev;
                            idx = a;
                        }
                    }
                    if (std::abs(angles[idx] - 90.0) > angle_tolerance)
                        continue;

                    std::array<const Element*, 3> elems = { &m1, &m2, &m3 };
                    double score = get_triplet_score(centers, elems);
                    candidates.push_back({ score, { i, j, k } });
                }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.score < b.score;
                  });

        std::vector<bool> used(n, false);
        std::vector<Triplet> triplets;
        for (const Candidate& c : candidates)
        {
            if (used[c.idx[0]] || used[c.idx[1]] || used[c.idx[2]])
                continue;
            used[c.idx[0]] = used[c.idx[1]] = used[c.idx[2]] = true;
            triplets.push_back(
                { elements[c.idx[0]], elements[c.idx[1]], elements[c.idx[2]] });
        }

        return triplets;
    }

    std::vector<Point> get_qr_corners(const Triplet& triplet)
    {
        const Element& A0 = triplet.a;
        const Element& B0 = triplet.b;
        const Element& C0 = triplet.c;

        Point cA = get_center(A0.bbox);
        Point cB = get_center(B0.bbox);
        Point cC = get_center(C0.bbox);

        double angles[3] = { angle(cB, cA, cC), angle(cA, cB, cC),
                             angle(cA, cC, cB) };
        int idx = 0;
        double best_dev = std::abs(angles[0] - 90.0);
        for (int i = 1; i < 3; i++)
        {
            double dev = std::abs(angles[i] - 90.0);
            if (dev < best_dev)
            {
                best_dev = dev;
                idx = i;
            }
        }

        const Element *TL, *TR, *BL;
        if (idx == 0)
        {
            TL = &A0;
            TR = &B0;
            BL = &C0;
        }
        else if (idx == 1)
        {
            TL = &B0;
            TR = &A0;
            BL = &C0;
        }
        else
        {
            TL = &C0;
            TR = &A0;
            BL = &B0;
        }

        Point A = get_center(TL->bbox);
        Point B = get_center(TR->bbox);
        Point C = get_center(BL->bbox);

        double u_row = B.row - A.row, u_col = B.col - A.col;
        double u_norm = std::hypot(u_row, u_col);
        if (u_norm > 0)
        {
            u_row /= u_norm;
            u_col /= u_norm;
        }

        double v_row = C.row - A.row, v_col = C.col - A.col;
        double v_norm = std::hypot(v_row, v_col);
        if (v_norm > 0)
        {
            v_row /= v_norm;
            v_col /= v_norm;
        }

        auto finder_size = [](const Element& e) {
            int minr = e.bbox[0], minc = e.bbox[1], maxr = e.bbox[2],
                maxc = e.bbox[3];
            return ((maxr - minr) + (maxc - minc)) / 2.0;
        };

        double size =
            (finder_size(*TL) + finder_size(*TR) + finder_size(*BL)) / 3.0;
        double offset = 1.2 * size / 2.0;

        Point TL_corner{ A.row - offset * u_row - offset * v_row,
                         A.col - offset * u_col - offset * v_col };
        Point TR_corner{ B.row + offset * u_row - offset * v_row,
                         B.col + offset * u_col - offset * v_col };
        Point BL_corner{ C.row - offset * u_row + offset * v_row,
                         C.col - offset * u_col + offset * v_col };
        Point BR_corner{ TR_corner.row + BL_corner.row - TL_corner.row,
                         TR_corner.col + BL_corner.col - TL_corner.col };

        return { TL_corner, TR_corner, BR_corner, BL_corner };
    }

    std::vector<Triplet>
    filter_contained_triplets(const std::vector<Triplet>& triplets,
                              double size_tolerance, double overlap_threshold)
    {
        std::vector<std::vector<Point>> corners_list;
        corners_list.reserve(triplets.size());
        for (const Triplet& t : triplets)
            corners_list.push_back(get_qr_corners(t));

        std::vector<bool> keep(triplets.size(), true);
        for (size_t i = 0; i < triplets.size(); i++)
        {
            if (!keep[i])
                continue;
            for (size_t j = 0; j < triplets.size(); j++)
            {
                if (i == j || !keep[j])
                    continue;
                if (should_remove_smaller(corners_list[i], corners_list[j],
                                          size_tolerance, overlap_threshold))
                    keep[j] = false;
            }
        }

        std::vector<Triplet> result;
        for (size_t i = 0; i < triplets.size(); i++)
            if (keep[i])
                result.push_back(triplets[i]);
        return result;
    }

    void draw_qr(image::rgb24_image& image, const std::vector<Point>& corners)
    {
        if (corners.size() != 4)
            return;
        for (int i = 0; i < 4; i++)
        {
            const Point& p1 = corners[i];
            const Point& p2 = corners[(i + 1) % 4];
            draw_line(image, static_cast<int>(p1.row), static_cast<int>(p1.col),
                      static_cast<int>(p2.row), static_cast<int>(p2.col));
        }
    }

} // namespace qr_code
