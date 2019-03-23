# Piecewise Polynomial Path Planner (P4)

## Description
Generalized piecewise polynomial solver. Given a set of waypoints and associated
arrival times, the solver will find a set of piece-wise polynomials that
smoothly fit through the points and maintain continuity up to a specified
derivative. 

At the core of the polynomial solver is a cost-minimization loop formulated by
as a quadratic programming problem and solved with
[OSQP](https://github.com/oxfordcontrol/osqp). The solver fits the polynomials
while minimizing the squared norm of a specified derivative and adhering to
specified, derivative-based path constraints. This approach follows the
minimum snap problem formulation and solution published by [Mellinger and
Kumar](https://ieeexplore.ieee.org/abstract/document/5980409). The theory behind
the polynomial solver can be found in the [doc/tex](doc/tex) directory. Make the
document by following the instructions below.

Examples are found in the [examples](examples/) directory. 

## Example Usage
```c++
#include <cstdlib>
#include <Eigen/Dense>

#include "polynomial_solver.h"
#include "polynomial_sampler.h"

using namespace mediation_layer;

int main(int argc, char** argv) {

  // Time in seconds
  const std::vector<double> times = {0, 1, 2.5};

  // NodeEqualityBound(dimension_idx, node_idx, derivative_idx, value)
  const std::vector<NodeEqualityBound> node_equality_bounds = {
    // The first node must constrain position, velocity, and acceleration
    // Constraining position, velocity, and acceleration to zero
    NodeEqualityBound(0,0,0,0),
    NodeEqualityBound(1,0,0,0),
    NodeEqualityBound(2,0,0,0),
    NodeEqualityBound(0,0,1,0),
    NodeEqualityBound(1,0,1,0),
    NodeEqualityBound(2,0,1,0),
    NodeEqualityBound(0,0,2,0),
    NodeEqualityBound(1,0,2,0),
    NodeEqualityBound(2,0,2,0),

    // Other nodes may constrain whatever they want
    // The second node is constraining position to (1,0,0)
    NodeEqualityBound(0,1,0,1),
    NodeEqualityBound(1,1,0,0),
    NodeEqualityBound(2,1,0,0),

    // The third node is contraining position to (1,1,free)
    NodeEqualityBound(0,2,0,1),
    NodeEqualityBound(1,2,0,1),
  };

  // NodeInequalityBound(dimension_idx, node_idx, derivative_idx, lower, upper)
  const std::vector<NodeInequalityBound> node_inequality_bounds = {
    // Constraining the z value of the third node above 0.5
    NodeInequalityBound(2,2,0,0.5,NodeInequalityBound::INFTY),
  };

  // SegmentInequalityBound(segment_idx, derivative_idx, mapping, value)
  // Segment inequality bounds constrain a derivative of a segment to 
  //   dot(mapping,dx^n/dt^n) < value
  const std::vector<SegmentInequalityBound> segment_inequality_bounds = {
    // Constraining the x-acceleration of the first segment below 4 m/s^2
    SegmentInequalityBound(0,2,Eigen::Vector3d(1,0,0),4),
  };

  // Configure solver options
  PolynomialSolver::Options solver_options;
  solver_options.num_dimensions   = 3;   // 3D
  solver_options.polynomial_order = 8;   // Fit an 8th-order polynomial
  solver_options.continuity_order = 4;   // Require continuity through the 4th derivative
  solver_options.derivative_order = 2;   // Minimize the 2nd derivative (acceleration)
  solver_options.polish = true;          // Polish the solution

  // Solve
  PolynomialSolver solver(solver_options);
  const PolynomialPath path
    = solver.Run(
        times, 
        node_equality_bounds,
        node_inequality_bounds,
        segment_inequality_bounds);

  // Sample the acceleration profile
  PolynomialSampler::Options sampler_options;
  sampler_options.frequency        = 100;
  sampler_options.derivative_order = 2;

  PolynomialSampler sampler(sampler_options);
  Eigen::MatrixXd samples = sampler.Run(times, path);
```

![](doc/img/trajectory.svg "Trajectory") 
![](doc/img/velocity.svg "Velocity") 
![](doc/img/acceleration.svg "Acceleration")


## Build Requirements
1) Install [Eigen](http://eigen.tuxfamily.org)
2) Install [OSQP](https://github.com/oxfordcontrol/osqp)
3) Install gnuplot and boost
```bash
sudo apt install libboost-all-dev gnuplot
```


## Build
```bash
mkdir build && cd build
cmake ..
make -j4
```


## Run
```bash
cd build/examples
./${EXECUTABLE_OF_CHOICE}
```

## Linking
Builds a shared library called lib_p4.

## Documentation
The theoretical documentation is written in LaTex. Latex must be installed
before building the pdf.

```bash
cd doc/tex
make
# open main.pdf
```

## Credits
- [OSQP](https://github.com/oxfordcontrol/osqp)
- [Eigen](http://eigen.tuxfamily.org)
- [gnuplot](http://www.gnuplot.info)
- [gnuplot-iostream](http://stahlke.org/dan/gnuplot-iostream/)

## Contact
Tucker Haydon (thaydon@utexas.edu)

