// Author: Tucker Haydon

#pragma once

#include <osqp.h>
#include <Eigen/Dense>

namespace p4 {
  // Example:
  // Given a 3D problem (XYZ), the y-snap of the third node can be
  // constrained to -1 as follows:
  //   NodeEqualityBound(1,2,4,-1)
  //
  struct NodeEqualityBound {
    // The dimension being constrained. If a path has four dimensions
    // (x,y,z,yaw), then 0 = x, 1 = y, etc.
    size_t dimension_idx;
    // The index of the node being constrained.
    size_t node_idx;
    // The index of the derivative being constraints. 0 = position, 1 =
    // velocity, etc.
    size_t derivative_idx;
    // The value that the constraint takes.
    double value;
  
    NodeEqualityBound(
      const size_t dimension_idx_,
      const size_t node_idx_,
      const size_t derivative_idx_,
      const double value_)
      : dimension_idx(dimension_idx_),
        node_idx(node_idx_),
        derivative_idx(derivative_idx_),
        value(value_) {}
  }; 

  // Example:
  // Given a 3D problem (XYZ), the y-snap of the third node can be
  // constrained to +-1 as follows:
  //   NodeInequalityBound(1,2,4,-1,+1)
  //
  struct NodeInequalityBound {
    static constexpr c_float INFTY = OSQP_INFTY;

    // The dimension being constrained. If a path has four dimensions
    // (x,y,z,yaw), then 0 = x, 1 = y, etc.
    size_t dimension_idx;
    // The index of the node being constrained.
    size_t node_idx;
    // The index of the derivative being constraints. 0 = position, 1 =
    // velocity, etc.
    size_t derivative_idx;
    // The value that the constraint takes.
    double lower, upper;
  
    NodeInequalityBound(
      const size_t dimension_idx_,
      const size_t node_idx_,
      const size_t derivative_idx_,
      const double lower_,
      const double upper_)
      : dimension_idx(dimension_idx_),
        node_idx(node_idx_),
        derivative_idx(derivative_idx_),
        lower(lower_),
        upper(upper_) {}
  }; 

  // Constraint: a' x < b
  // Example:
  // Given a 3D problem (XYZ), the x-vel of the 3rd segment can be constrained
  // above zero as follows:
  //   SegmentInequalityBound(2,1,Eigen::Vector3d(-1,0,0),0)
  //
  struct SegmentInequalityBound {
    static constexpr c_float INFTY = OSQP_INFTY;

    // The index of the node being constrained.
    size_t segment_idx;
    // The index of the derivative being constraints. 0 = position, 1 =
    // velocity, etc.
    size_t derivative_idx;
    // A mapping that from N-to-scalar
    Eigen::MatrixXd mapping;
    // The value that the constraint takes.
    double value;
  
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SegmentInequalityBound(
      const size_t segment_idx_,
      const size_t derivative_idx_,
      Eigen::MatrixXd mapping_,
      const double value_)
      :
        segment_idx(segment_idx_),
        derivative_idx(derivative_idx_),
        mapping(mapping_),
        value(value_) {}
  }; 
}
