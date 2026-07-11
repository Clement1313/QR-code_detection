#include "detection/detection.hh"
#include <gtest/gtest.h>

using namespace qr_code;

namespace {
Element make_element(std::array<int, 4> bbox, long area = -1) {
  int minr = bbox[0], minc = bbox[1], maxr = bbox[2], maxc = bbox[3];
  std::vector<Point> corners = {
      {static_cast<double>(minr), static_cast<double>(minc)},
      {static_cast<double>(minr), static_cast<double>(maxc)},
      {static_cast<double>(maxr), static_cast<double>(maxc)},
      {static_cast<double>(maxr), static_cast<double>(minc)},
  };
  long computed_area =
      (area >= 0) ? area : static_cast<long>(maxr - minr) * (maxc - minc);
  return {bbox, corners, computed_area};
}

} // namespace

TEST(GetCenter, ComputesMidpoint) {
  std::array<int, 4> bbox{10, 20, 30, 40};
  Point c = get_center(bbox);
  EXPECT_DOUBLE_EQ(c.row, 20.0);
  EXPECT_DOUBLE_EQ(c.col, 30.0);
}

TEST(Angle, RightAngleIsNinety) {
  Point a{0, 1}, b{0, 0}, c{1, 0};
  EXPECT_NEAR(angle(a, b, c), 90.0, 0.1);
}

TEST(IsConvexQuad, DetectsSquareAsConvex) {
  std::vector<Point> square = {{0, 0}, {0, 10}, {10, 10}, {10, 0}};
  EXPECT_TRUE(is_convex_quad(square));
}

TEST(PolygonArea, ComputesSquareArea) {
  std::vector<Point> square = {{0, 0}, {0, 10}, {10, 10}, {10, 0}};
  EXPECT_NEAR(polygon_area(square), 100.0, 1e-6);
}

TEST(GetTriplets, ThreeValidMarkersFormOneTriplet)
{
    // Configuration en L, angle droit au marqueur "A" (haut-gauche)
    std::vector<Element> elements = {
        make_element({ 0, 0, 20, 20 }),      // TL
        make_element({ 0, 180, 20, 200 }),   // TR
        make_element({ 180, 0, 200, 20 }),   // BL
    };

    auto triplets = get_triplets(elements, 400, 400);
    EXPECT_EQ(triplets.size(), 1u);
}

TEST(GetTriplets, TooFewElementsReturnsEmpty)
{
    std::vector<Element> elements = {
        make_element({ 0, 0, 20, 20 }),
        make_element({ 0, 180, 20, 200 }),
    };
    auto triplets = get_triplets(elements, 400, 400);
    EXPECT_TRUE(triplets.empty());
}

TEST(GetTriplets, DissimilarAreasRejected)
{
    std::vector<Element> elements = {
        make_element({ 0, 0, 20, 20 }, /*area=*/400),
        make_element({ 0, 180, 20, 200 }, /*area=*/400),
        // aire très différente (ratio > 1.6) -> similar_area doit rejeter
        make_element({ 180, 0, 260, 80 }, /*area=*/6400),
    };
    auto triplets = get_triplets(elements, 400, 400);
    EXPECT_TRUE(triplets.empty());
}

TEST(GetTriplets, NoSharedMarkerBetweenTwoQrCodes)
{
    // deux QR codes séparés, chacun avec 3 marqueurs cohérents entre eux
    std::vector<Element> elements = {
        make_element({ 0, 0, 20, 20 }),
        make_element({ 0, 180, 20, 200 }),
        make_element({ 180, 0, 200, 20 }),

        make_element({ 500, 500, 520, 520 }),
        make_element({ 500, 680, 520, 700 }),
        make_element({ 680, 500, 700, 520 }),
    };
    auto triplets = get_triplets(elements, 1000, 1000);
    EXPECT_EQ(triplets.size(), 2u);
}

TEST(GetTriplets, MarkerNotReusedAcrossTriplets)
{
    // un marqueur "central" pourrait potentiellement matcher deux triplets ;
    // on vérifie qu'aucun élément n'apparaît deux fois dans le résultat
    std::vector<Element> elements = {
        make_element({ 0, 0, 20, 20 }),
        make_element({ 0, 180, 20, 200 }),
        make_element({ 180, 0, 200, 20 }),
        make_element({ 180, 180, 200, 200 }), // 4e marqueur "coin bas-droit"
    };
    auto triplets = get_triplets(elements, 400, 400);

    std::vector<std::array<int,4>> seen;
    for (const auto& t : triplets)
    {
        for (const Element* e : { &t.a, &t.b, &t.c })
        {
            for (const auto& s : seen)
                EXPECT_NE(s, e->bbox) << "un marqueur est réutilisé dans plusieurs triplets";
            seen.push_back(e->bbox);
        }
    }
}

class FilterContainedTripletsTest : public ::testing::Test
{
protected:
    Triplet big = { make_element({ 0, 0, 20, 20 }),
                    make_element({ 0, 180, 20, 200 }),
                    make_element({ 180, 0, 200, 20 }) };

    Triplet small = { make_element({ 80, 80, 90, 90 }),
                      make_element({ 80, 120, 90, 130 }),
                      make_element({ 120, 80, 130, 90 }) };
};

TEST_F(FilterContainedTripletsTest, EmptyInputReturnsEmpty)
{
    EXPECT_TRUE(filter_contained_triplets({}).empty());
}

TEST_F(FilterContainedTripletsTest, SingleTripletKept)
{
    auto result = filter_contained_triplets({ big });
    EXPECT_EQ(result.size(), 1u);
}

TEST_F(FilterContainedTripletsTest, SmallerInsideBiggerIsRemoved)
{
    auto result = filter_contained_triplets({ big, small });
    EXPECT_EQ(result.size(), 1u);

    double kept_area = polygon_area(get_qr_corners(result[0]));
    double big_area = polygon_area(get_qr_corners(big));
    EXPECT_NEAR(kept_area, big_area, 1e-6);
}

TEST_F(FilterContainedTripletsTest, OrderIndependence)
{
    auto result_big_first = filter_contained_triplets({ big, small });
    auto result_small_first = filter_contained_triplets({ small, big });

    EXPECT_EQ(result_big_first.size(), 1u);
    EXPECT_EQ(result_small_first.size(), 1u);

    double area1 = polygon_area(get_qr_corners(result_big_first[0]));
    double area2 = polygon_area(get_qr_corners(result_small_first[0]));
    EXPECT_NEAR(area1, area2, 1e-6);
}

TEST_F(FilterContainedTripletsTest, TwoSeparateQrCodesBothKept)
{
    Triplet qr2 = { make_element({ 500, 500, 520, 520 }),
                    make_element({ 500, 680, 520, 700 }),
                    make_element({ 680, 500, 700, 520 }) };

    auto result = filter_contained_triplets({ big, qr2 });
    EXPECT_EQ(result.size(), 2u);
}

TEST_F(FilterContainedTripletsTest, ThreeNestedTripletsKeepsOnlyOuter)
{
    Triplet biggest = { make_element({ 0, 0, 20, 20 }),
                         make_element({ 0, 380, 20, 400 }),
                         make_element({ 380, 0, 400, 20 }) };
    Triplet medium = { make_element({ 100, 100, 115, 115 }),
                        make_element({ 100, 250, 115, 265 }),
                        make_element({ 250, 100, 265, 115 }) };
    Triplet tiny = { make_element({ 150, 150, 158, 158 }),
                      make_element({ 150, 200, 158, 208 }),
                      make_element({ 200, 150, 208, 158 }) };

    auto result = filter_contained_triplets({ biggest, medium, tiny });
    EXPECT_EQ(result.size(), 1u);

    double kept_area = polygon_area(get_qr_corners(result[0]));
    double biggest_area = polygon_area(get_qr_corners(biggest));
    EXPECT_NEAR(kept_area, biggest_area, 1e-6);
}