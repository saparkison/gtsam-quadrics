/* ----------------------------------------------------------------------------

 * QuadricSLAM Copyright 2020, ARC Centre of Excellence for Robotic Vision, Queensland University of Technology (QUT)
 * Brisbane, QLD 4000
 * All Rights Reserved
 * Authors: Lachlan Nicholson, et al. (see THANKS for the full author list)
 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Utilities.h
 * @date Apr 14, 2020
 * @author Lachlan Nicholson
 * @brief a namespace providing a number of useful functions
 */

#pragma once

#include <quadricslam/base/Noise.h>

#include <gtsam/base/Matrix.h>

#include <vector>

namespace gtsam {
namespace utils {

/** 
 * Returns the real roots of the polynomial
 * If disc > 0: 2 solutions
 * If disc == 0: 1 real solution
 * If disc < 0: 2 imaginary solutions
 */
Vector2 solvePolynomial(const double& a, const double& b, const double& c);

Vector2 getConicPointsAtX(const Matrix3& pointConic, const double& x);

Vector2 getConicPointsAtY(const Matrix3& pointConic, const double& y);

/** Interpolate poses */ 
Pose3 interpolate(const Pose3& p1, const Pose3& p2, const double& percent);

/** Converts Pose3 to Matrix and provides optional jacobians */
Matrix44 matrix(const Pose3& pose, OptionalJacobian<16,6> H = boost::none);

/** 
 * Performs the kronecker product 
 * See: https://en.wikipedia.org/wiki/Kronecker_product
 */
Matrix kron(const Matrix m1, const Matrix m2);

/** 
 * Builds the orthogonal transpose vectorization matrix of an m by n matrix 
 * See: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/special.html#VecTranspose
 */
Matrix TVEC(const int m, const int n);

} // namespace utils
} // namespace gtsam
