// Author: Tucker Haydon

#include <cstdlib>

#include <osqp.h>

#include <Eigen/Core>
#include <iostream>

void Path2PVA(
    const std::vector<Eigen::Matrix<double, 4, 1>>& pos_vec,
    const std::vector<Eigen::Matrix<double, 4, 1>>& vel_vec,
    const std::vector<Eigen::Matrix<double, 4, 1>>& acc_vec,
    const std::vector<double>& time_vec) {
  /*
  * This is the Compressed Sparse Column (CSC) format. It is very common when
  * describing sparse matrix. P->i are the row indices and P->p are the column
  * pointers (not the column indices!). The column pointers are pointer to the
  * first elements of P->x in every column (if there are no elements in that
  * column, they point to next element in the P->x array). The last element of
  * P->p corresponds to the total number of nonzeros in the matrix by
  * construction.
  */
  const size_t num_nodes = pos_vec.size();
  const size_t num_segments = num_nodes - 1;
  const size_t polynomial_order = 8;
  const size_t num_states_per_segment = 4 * polynomial_order;
  const size_t num_states = num_states_per_segment * num_segments;
  const size_t nnz_states_per_segment = 4;
  const size_t nnz_state_offset = 16;

  // 12 constraints of start, end
  // Enforce continuity of 5 derivatives for every intersection
  // TODO: Corridor constraints
  size_t num_constraints = 2*4*3 + 4*5*num_segments;

  // Construct quadratic matrix
  // Very sparse matrix --- zeros everywhere except for ones along the snap
  // elements
  c_int P_nnz = 4 * num_segments;
  std::vector<c_float> P_x(P_nnz, 1.0);

  // Row indices of the matrix elements
  std::vector<c_int> P_i(P_nnz, 0);
  for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
    for(size_t state_idx = 0; state_idx < nnz_states_per_segment; ++state_idx) {
      size_t idx = segment_idx * nnz_states_per_segment + state_idx;
      std::cout << idx << std::endl;
      P_i[idx] 
        = segment_idx * num_states_per_segment + nnz_state_offset + state_idx;
    }
  }

  // The final columns are chopped off because they are zeros
  size_t num_columns = num_states + 1 - (num_states_per_segment - nnz_state_offset - nnz_states_per_segment);
  std::vector<c_int> P_p(num_columns, 0);
  for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
    for(size_t ptr_idx = 0; ptr_idx < num_states_per_segment; ++ptr_idx) {

      // Bound check
      size_t idx = segment_idx * num_states_per_segment + ptr_idx;
      if(idx == num_columns) {
        break;
      }

      // Push the data
      if(ptr_idx < nnz_state_offset) {
        P_p[idx] = segment_idx * nnz_states_per_segment;
      } else if(ptr_idx >= nnz_state_offset && ptr_idx < nnz_state_offset + nnz_states_per_segment) {
        P_p[idx] = segment_idx * nnz_states_per_segment + ptr_idx - nnz_state_offset;
      } else {
        P_p[idx] = (segment_idx + 1) * nnz_states_per_segment;
      }
    }
  }
  P_p.back() = P_nnz;


  // Construct constraint
  // Very sparse matrix --- zeros everywhere except for ones along the snap
  // elements
  c_int A_nnz = num_constraints;
  std::vector<c_float> A_x(A_nnz, 1.0);

  // Row indices of the matrix elements
  // std::vector<c_int> A_i(A_nnz, 0);
  // for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
  //   for(size_t state_idx = 0; state_idx < nnz_states_per_segment; ++state_idx) {
  //     size_t idx = segment_idx * nnz_states_per_segment + state_idx;
  //     std::cout << idx << std::endl;
  //     P_i[idx] 
  //       = segment_idx * num_states_per_segment + nnz_state_offset + state_idx;
  //   }
  // }

  // // The final columns are chopped off because they are zeros
  // size_t num_columns = num_states + 1 - (num_states_per_segment - nnz_state_offset - nnz_states_per_segment);
  // std::vector<c_int> P_p(num_columns, 0);
  // for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
  //   for(size_t ptr_idx = 0; ptr_idx < num_states_per_segment; ++ptr_idx) {

  //     // Bound check
  //     size_t idx = segment_idx * num_states_per_segment + ptr_idx;
  //     if(idx == num_columns) {
  //       break;
  //     }

  //     // Push the data
  //     if(ptr_idx < nnz_state_offset) {
  //       P_p[idx] = segment_idx * nnz_states_per_segment;
  //     } else if(ptr_idx >= nnz_state_offset && ptr_idx < nnz_state_offset + nnz_states_per_segment) {
  //       P_p[idx] = segment_idx * nnz_states_per_segment + ptr_idx - nnz_state_offset;
  //     } else {
  //       P_p[idx] = (segment_idx + 1) * nnz_states_per_segment;
  //     }
  //   }
  // }
  // P_p.back() = P_nnz;
  
  std::vector<c_float> l;

  // No linear cost
  std::vector<c_float> q(num_states, 0.0);

  // Structures
  OSQPWorkspace * work;  // Workspace
  OSQPData * data;  // OSQPData
  data = (OSQPData *)c_malloc(sizeof(OSQPData));


  // Populate data
  data->n = num_states;
  data->m = num_constraints;
  data->P = csc_matrix(data->n, data->n, P_nnz, P_x.data(), P_i.data(), P_p.data());
  data->q = q.data();
  // data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
  data->l = l;
  data->u = u;


  // Define Solver settings as default
  OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);
  settings->alpha = 1.0; // Change alpha parameter

  // Setup workspace
  // work = osqp_setup(data, settings);

  // Solve Problem
  // osqp_solve(work);

  // Cleanup
  // osqp_cleanup(work);
  // c_free(data->A);
  // c_free(data->P);
  // c_free(data);
  // c_free(settings);
}

int main(int argc, char **argv) {
    // Load problem data
    // c_float P_x[4] = {4.00, 1.00, 1.00, 2.00, };
    // c_int P_nnz = 4;
    // c_int P_i[4] = {0, 1, 0, 1, };
    // c_int P_p[3] = {0, 2, 4, };
    // c_float q[2] = {1.00, 1.00, };
    // c_float A_x[4] = {1.00, 1.00, 1.00, 1.00, };
    // c_int A_nnz = 4;
    // c_int A_i[4] = {0, 1, 0, 2, };
    // c_int A_p[3] = {0, 2, 4, };
    // c_float l[3] = {1.00, 0.00, 0.00, };
    // c_float u[3] = {1.00, 0.70, 0.70, };
    // c_int n = 2;
    // c_int m = 3;


    std::vector<Eigen::Matrix<double, 4, 1>> pos_vec, vel_vec, acc_vec;

    pos_vec.emplace_back(0,0,0,0);
    pos_vec.emplace_back(1,0,0,0);

    vel_vec.emplace_back(0,0,0,0);
    vel_vec.emplace_back(0,0,0,0);

    acc_vec.emplace_back(0,0,0,0);
    acc_vec.emplace_back(0,0,0,0);

    std::vector<double> time_vec = {0, 1};

    Path2PVA(pos_vec, vel_vec, acc_vec, time_vec);

    return EXIT_SUCCESS;
};
