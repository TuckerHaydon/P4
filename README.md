## MinSnap

# Description
Generalized piecewise polynomial solver. Given a series of waypoints and arrival
times, the solver finds a set of piecewise polynomials that minimize a
derivative of the polynomial. This solver can solve for the minimum snap
trajectory.

The interface requires the following:
- The zeroth, first, and second derivatives for the initial node must be
  specified (position, velocity, acceleration)
- The time of arrival at each node must be specified
- The zeroith derivative of each node must be specified (position)
- Higher derivatives of each node may be optionally specified
- Inequality path constraints for any derivative of any node may be optionally
  specified

# Requirements
https://github.com/oxfordcontrol/osqp


