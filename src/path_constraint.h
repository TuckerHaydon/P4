// Author: Tucker Haydon

#pragma once

namespace mediation_layer {
  // Constrains one dimension for one node in the path
  // Example: constrain the x-position of the 5th node to be 5
  struct PathConstraint {
    const size_t index;
    const size_t dimension;
    const double value;
  
    PathConstraint(
      const size_t index_,
      const size_t dimension_,
      const double value_)
      : index(index_),
        dimension(dimension_),
        value(value_) {}
  };
  
  struct LowerBound : public PathConstraint {};
  struct UpperBound : public PathConstraint {};
  struct EqualityBound : public PathConstraint {};
};
