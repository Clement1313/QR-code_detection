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
        
        bool check_area(long a1, long a2, double ratio = 1.6)
        {
            if (a1 <= 0 || a2 <= 0)
                return false;
            double mx = static_cast<double>(std::max(a1, a2));
            double mn = static_cast<double>(std::min(a1, a2));
            return (mx / mn) < ratio;
        }

        bool similar_area(const Element& m1, const Element& m2, const Element& m3)
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

    } // namespace

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

} // namespace qr_code
