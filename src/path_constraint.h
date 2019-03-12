// Author: Tucker Haydon

#pragma once

namespace mediation_layer {
  struct PathConstraint {
    const size_t node_idx;
    const size_t dimension_idx;
    const size_t derivative_idx;
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

  using EqualityBound = PathConstraint;
  using LowerBound = PathConstraint;
  using UpperBound = PathConstraint;
  
};
