// Author: Tucker Haydon

#pragma once

namespace mediation_layer {
  // A path constraint encapsulates relevant data that provides an upper, lower,
  // or equality bound on the polynomial path. 
  //
  // Example:
  // Given a 3D problem (XYZ), the y-snap of the third node can be
  // constrained to -1 as follows:
  //   PathConstraint(1,2,4,-1)
  //
  struct PathConstraint {
    // The dimension being constrained. If a path has four dimensions
    // (x,y,z,yaw), then 0 = x, 1 = y, etc.
    const size_t dimension_idx;
    // The index of the node being constrained.
    const size_t node_idx;
    // The index of the derivative being constraints. 0 = position, 1 =
    // velocity, etc.
    const size_t derivative_idx;
    // The value that the constraint takes.
    const double value;
  
    PathConstraint(
      const size_t dimension_idx_,
      const size_t node_idx_,
      const size_t derivative_idx_,
      const double value_)
      : dimension_idx(dimension_idx_),
        node_idx(node_idx_),
        derivative_idx(derivative_idx_),
        value(value_) {}
  };

  // Helper types
  using EqualityBound = PathConstraint;
  using LowerBound = PathConstraint;
  using UpperBound = PathConstraint;
  
};
