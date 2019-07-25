// Author: Tucker Haydon

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "polynomial_gradient.h"
#include "common.h"

namespace p4 {
  namespace {
    // Upper-level cost function of the form: 
    //   f(x*, y) = 0.5 * x' P x + ones() * y
    //
    // Want to determine the gradient with respect to y, holding x* constant.
    // Although automatic differentiation is not necessary (gradient is just
    // ones()), this class is included for consistency and in case a new
    // time-dependent function is introduced.
    //
    // TODO: Remove extra computation and just return ones()
    class ObjectiveCostFunction {
     public:
      ObjectiveCostFunction(const PolynomialSolver::Solution& solver_solution)
        : solver_solution_(solver_solution) {}
    
      static ceres::CostFunction* Create(
          const PolynomialSolver::Solution& solver_solution) {
        auto cost_function = 
          new ceres::DynamicAutoDiffCostFunction<ObjectiveCostFunction>(
            new ObjectiveCostFunction(solver_solution));
        // The number of nodes is equal to the number of times to optimize
        cost_function->AddParameterBlock(solver_solution.constants.num_nodes);
        // There is only one residual: the evaluated cost function
        cost_function->SetNumResiduals(1);

        return cost_function;
      }
    
      template <typename T>
      bool operator()(
          T const* const* times, 
          T* residuals) const {
        // Get the size of the time vector
        const int32_t times_size = this->solver_solution_.constants.num_nodes;

        // Cast constant coefficients to type Jet
        const auto x = 
          Eigen::Map<const Eigen::Matrix<c_float, Eigen::Dynamic, 1>>(
              this->solver_solution_.workspace->pol->x,
              this->solver_solution_.data->n).cast<T>();

        // Cast quadratic matrix to type Jet
        Eigen::SparseMatrix<c_float> P_float;
        OSQP2Eigen(
              this->solver_solution_.data->P,
              P_float);
        const Eigen::SparseMatrix<T> P = P_float.cast<T>();

        // Compose times in Eigen vector
        // times is a vector, but is represented as a double pointer due to the
        // DynamicAutoDiffCostFunction interface
        const auto t =
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(times[0], times_size);


        // Fill cost function/residuals
        residuals[0] = (T(0.5) * x.transpose() * P * x + 
          Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(times_size, 1).transpose() * t).eval()(0,0); 
    
        return true;
      }
    
      private:
        PolynomialSolver::Solution solver_solution_;
    };

    class ConstraintCostFunction {
     public:
      ConstraintCostFunction(
          const PolynomialSolver::Solution& solver_solution,
          const std::shared_ptr<const PolynomialSolver>& solver)
        : solver_solution_(solver_solution),
          solver_(solver) {}
    
      static ceres::CostFunction* Create(
          const PolynomialSolver::Solution& solver_solution,
          const std::shared_ptr<const PolynomialSolver>& solver) {
        auto cost_function = 
          new ceres::DynamicAutoDiffCostFunction<ConstraintCostFunction>(
            new ConstraintCostFunction(solver_solution, solver));
        // The number of nodes is equal to the number of times to optimize
        cost_function->AddParameterBlock(solver_solution.constants.num_nodes);
        // Multiply by two to account for upper and lower constraints
        cost_function->SetNumResiduals(2*solver_solution.data->m);

        return cost_function;
      }
    
      template <typename T>
      bool operator()(
          T const* const* times, 
          T* residuals) const {
        // Get the size of the time vector
        const int32_t times_size = this->solver_solution_.constants.num_nodes;

        // Cast constant coefficients to type Jet
        const auto x = 
          Eigen::Map<const Eigen::Matrix<c_float, Eigen::Dynamic, 1>>(
              this->solver_solution_.workspace->pol->x,
              this->solver_solution_.data->n).cast<T>();

        // l <= Ax <= u
        // Ax = z
        // l <= z <= u
        // Cast constant primal solution to type Jet
        // const auto z = 
        //   Eigen::Map<const Eigen::Matrix<c_float, Eigen::Dynamic, 1>>(
        //       this->solver_solution_.workspace->pol->z,
        //       this->solver_solution_.data->m).cast<T>();

        // y+(z - u) = 0
        // y-(z - l) = 0A
        // y+ = max(y, 0)
        // y- = min(y, 0)
        // Cast constant lagrange solution to type Jet
        // const auto y = 
        //   Eigen::Map<const Eigen::Matrix<c_float, Eigen::Dynamic, 1>>(
        //       this->solver_solution_.workspace->pol->y,
        //       this->solver_solution_.data->m).cast<T>();

        // Cast constraint matrix to type Jet
        // Eigen::SparseMatrix<double> A_float;
        // OSQP2Eigen(
        //       this->solver_solution_.data->A,
        //       A_float);
        // const Eigen::SparseMatrix<T> A = A_float.cast<T>();

        // Compose times in std vector
        // times is a vector, but is represented as a double pointer due to the
        // DynamicAutoDiffCostFunction interface
        const std::vector<T> t(times[0], times[0]+times_size);

        // Compose constraints
        Eigen::SparseMatrix<T> A;
        Eigen::Matrix<T, Eigen::Dynamic, 1> l;
        Eigen::Matrix<T, Eigen::Dynamic, 1> u;
        this->solver_->SetConstraints(t, l, u, A);

        // Calculate residuals
        // Residuals are defined as [Ax-u; Ax-l] = 0
        Eigen::Matrix<T, Eigen::Dynamic, 1> upper_residuals = A*x - u;
        Eigen::Matrix<T, Eigen::Dynamic, 1> lower_residuals = A*x - l;

        // Fill residuals
        for(size_t idx = 0; idx < this->solver_solution_.data->m; ++idx) {
          residuals[idx] = upper_residuals(idx);
          residuals[this->solver_solution_.data->m + idx] = lower_residuals(idx);
        }
    
        return true;
      }
    
      private:
        PolynomialSolver::Solution solver_solution_;
        std::shared_ptr<const PolynomialSolver> solver_;
    };

    // RAII buffer for storing a 2D floating point array
    class SmartBuffer2D {
      public:
        SmartBuffer2D(
            const size_t rows, 
            const size_t cols) 
        : rows_(rows),
          cols_(cols) {
          this->data_ = new double*[this->rows_];
          for(size_t row = 0; row < this->rows_; ++row) {
            this->data_[row] = new double[this->cols_];
          }
        }

        ~SmartBuffer2D() {
          for(size_t row = 0; row < this->rows_; ++row) {
            delete[] this->data_[row];
          }
          delete[] this->data_;
        }

        double** Get() const {
          return this->data_;
        }

        size_t Rows() const {
          return this->rows_;
        }

        size_t Cols() const {
          return this->cols_;
        }

      private:
        double** data_;
        size_t rows_;
        size_t cols_;
    };

    // RAII buffer for storing a 1D floating point array
    class SmartBuffer1D {
      public:
        SmartBuffer1D(const size_t rows) 
        : rows_(rows) {
          this->data_ = new double[this->rows_];
        }

        ~SmartBuffer1D() {
          delete[] this->data_;
        }

        double* Get() const {
          return this->data_;
        }

        size_t Rows() const {
          return this->rows_;
        }

      private:
        double* data_;
        size_t rows_;
    };
  }

  void PolynomialGradient::Test(
      const std::vector<double>& initial_times,
      const std::shared_ptr<const PolynomialSolver>& solver,
      const PolynomialSolver::Solution& solver_solution) {

    // TODO: Solution must be polished to have access to z
    // TODO: Verify that z = Ax

    // Notes:
    // http://ceres-solver.org/nnls_modeling.html#_CPPv2N5ceres12CostFunction8EvaluateEPPCdPdPPd
    // jacobians[i][r * parameter_block_sizes_[i] + c] =
    // partial residual[r]
    // -------------------
    // partial parameters[i][c]


    // { // Quadratic graient test
    //   // Structures for autodiff evaluation
    //   const size_t times_size = initial_times.size();
    //   SmartBuffer2D times(1,times_size);
    //   std::memcpy(times.Get()[0], initial_times.data(), times_size * sizeof(double));
    //   SmartBuffer1D residuals(1);
    //   SmartBuffer2D jacobian(1, times_size);

    //   // Cost function
    //   const ceres::CostFunction* cost_function 
    //     = ObjectiveCostFunction::Create(solver_solution);

    //   // Evaluate gradient
    //   bool success = cost_function->Evaluate(
    //       times.Get(),
    //       residuals.Get(),
    //       jacobian.Get());

    //   if(true == success) {
    //     std::cout << "Residual: " << residuals.Get()[0] << std::endl;
    //     for(size_t jacobian_idx = 0; jacobian_idx < solver_solution.constants.num_nodes; ++jacobian_idx) {
    //       std::cout << "Jacobian: " << jacobian.Get()[0][jacobian_idx] << std::endl;
    //     }
    //   } else {
    //     std::cout << "Jacobian Failed!" << std::endl;
    //   }
    // }

    { // Constraint gradient test
      // Structures for autodiff evaluation
      const size_t times_size = initial_times.size();
      SmartBuffer2D times(1,times_size);
      std::memcpy(times.Get()[0], initial_times.data(), times_size * sizeof(double));
      SmartBuffer1D residuals(2*solver_solution.data->m);
      SmartBuffer2D jacobian(1, 2*solver_solution.data->m * times_size);

      // Cost function
      const ceres::CostFunction* cost_function 
        = ConstraintCostFunction::Create(solver_solution, solver);

      // Evaluate gradient
      bool success = cost_function->Evaluate(
          times.Get(),
          residuals.Get(),
          jacobian.Get());

      if(true == success) {
        // Columns contain parameters for individual nodes
        auto residuals_eigen 
          = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 2>>(residuals.Get(), solver_solution.data->m, 2);
        std::cout << residuals_eigen << std::endl;
        // for(size_t jacobian_idx = 0; jacobian_idx < solver_solution.constants.num_nodes; ++jacobian_idx) {
        //   std::cout << "Jacobian: " << jacobian.Get()[0][jacobian_idx] << std::endl;
        // }
      } else {
        std::cout << "Jacobian Failed!" << std::endl;
      }

    }
  }


  PolynomialGradient::Solution PolynomialGradient::Run(
      const std::vector<double>& initial_times,
      const PolynomialSolver::Solution& solver_solution) {

    return PolynomialGradient::Solution();
  }
}
