// Author: Tucker Haydon

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "polynomial_gradient.h"

namespace p4 {
  namespace {
    // Upper-level cost function of the form: 
    //   f(x*, y) = x' P x + ones() * y
    //
    // Want to determine the gradient with respect to y, holding x* constant.
    // Although automatic differentiation is not necessary (gradient is just
    // ones()), this class is included for consistency and in case a new
    // time-dependent function is introduced.
    //
    // TODO: Remove extra computation and just return ones()
    class PolynomialCostFunction {
     public:
      PolynomialCostFunction(const PolynomialSolver::Solution& solver_solution)
        : solver_solution_(solver_solution) {}
    
      static ceres::CostFunction* Create(
          const PolynomialSolver::Solution& solver_solution) {
        auto cost_function = 
          new ceres::DynamicAutoDiffCostFunction<PolynomialCostFunction>(
            new PolynomialCostFunction(solver_solution));
        // The number of nodes is equal to the number of times to optimize
        cost_function->AddParameterBlock(solver_solution.num_nodes);
        // There is only one residual: the evaluated cost function
        cost_function->SetNumResiduals(1);

        return cost_function;
      }
    
      template <typename T>
      bool operator()(
          T const* const* times, 
          T* residuals) const {
        // Residuals are not necessarily zero'd out. This can introduce bugs if
        // a residual slot is incorrectly left unfilled.
        residuals[0] = T(0);

        // Get the size of times
        // const int32_t size_y = parameter_block_sizes()[0];
        const int32_t size_y = this->solver_solution_.num_nodes;

        // Cast constant coefficients to type Jet
        const auto x = 
          Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>(
              this->solver_solution_.workspace->solution->x,
              this->solver_solution_.workspace->data->n).cast<T>();

        // Cast quadratic matrix to type Jet
        // const auto P = 
        //   Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(
        //       this->solver_solution_.workspace->data->A,
        //       this->solver_solution_.workspace->data->n).cast<T>();

        // Compose times in Eigen vector
        // times is a vector, but is represented as a double pointer due to the
        // DynamicAutoDiffCostFunction interface
        const auto y =
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(times[0], size_y);


        // Fill cost function/residuals
        // residuals[0] = x.transpose() * P * x + 
        //   Eigen::Matrix<T, S, 1>::Ones().transpose() * y; 
        residuals[0] = Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(size_y, 1).transpose() * y; 
    
        return true;
      }
    
      private:
        const PolynomialSolver::Solution solver_solution_;
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

      private:
        double* data_;
        size_t rows_;
    };
  }

  void PolynomialGradient::Test(const PolynomialSolver::Solution& solver_solution) {
    // Initial Values
    const std::vector<double> initial_times = {1.0, 2.0, 5.0};
    const size_t times_size = initial_times.size();

    // Structures for autodiff evaluation
    SmartBuffer2D times(1,times_size);
    std::memcpy(times.Get()[0], initial_times.data(), times_size * sizeof(double));
    SmartBuffer1D residuals(1);
    SmartBuffer2D jacobian(1, times_size);

    // Coefficients. Empty atm.
    std::vector<std::vector<Eigen::VectorXd>> coefficients;

    // Cost function
    const ceres::CostFunction* cost_function 
      = PolynomialCostFunction::Create(solver_solution);

    // Evaluate gradient
    bool success = cost_function->Evaluate(
        times.Get(),
        residuals.Get(),
        jacobian.Get());

    if(true == success) {
      std::cout << "Residual: " << residuals.Get()[0] << std::endl;
      for(size_t jacobian_idx = 0; jacobian_idx < solver_solution.num_nodes; ++jacobian_idx) {
        std::cout << "Jacobian: " << jacobian.Get()[0][jacobian_idx] << std::endl;
      }
    } else {
      std::cout << "Jacobian Failed!" << std::endl;
    }
  }


  PolynomialGradient::Solution PolynomialGradient::Run(
      const PolynomialSolver::Solution& solver_solution) {

    return PolynomialGradient::Solution();
  }
}
