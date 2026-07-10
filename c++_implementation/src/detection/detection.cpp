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
