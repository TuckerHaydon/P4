// Author: Tucker Haydon

#pragma once

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
