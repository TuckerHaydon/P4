# MinSnap

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


## Pre-Installation Requirements
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


## Documentation
```bash
cd doc/tex
make
# open main.pdf
```


## TODO
- Corridor Constraints
- Parallelized QP solver (one for each dimension)

## Contact
Tucker Haydon (thaydon@utexas.edu)

