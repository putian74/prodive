// Matrix inversion using CPU/Eigen.
#include "kl_divergence.h"
#include <Eigen/Dense>
#include <iostream>
#include <omp.h>

bool check_matrix_condition(const std::vector<float>& matrix, int size) {
    Eigen::MatrixXf A = Eigen::Map<const Eigen::MatrixXf>(matrix.data(), size, size);
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A);
    float condition_number = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    if (condition_number > 1e12) {
        std::cout << "Warning: Matrix condition number " << condition_number
                  << " indicates potential numerical instability" << std::endl;
        return false;
    }
    return true;
}

std::vector<float> invert_matrices_cpu(const std::vector<FragSet>& datasets, int m_states) {
    std::cout << "Computing matrix inversions using CPU (Eigen)..." << std::endl;

    int total_frags = 0;
    for (const auto& ds : datasets) total_frags += ds.nfrag;

    std::vector<float> all_inversions((size_t)total_frags * m_states * m_states);
    int processed = 0;

    for (const auto& ds : datasets) {
        if (ds.m_states != m_states) {
            throw std::runtime_error("All loaded datasets must have the same m_states for one run.");
        }

        #pragma omp parallel for
        for (int frag = 0; frag < ds.nfrag; frag++) {
            Eigen::MatrixXf IA(m_states, m_states);
            for (int i = 0; i < m_states; i++) {
                for (int j = 0; j < m_states; j++) {
                    float identity_val = (i == j) ? 1.0f : 0.0f;
                    float a_val = ds.A[(size_t)frag * m_states * m_states + i * m_states + j];
                    IA(i, j) = identity_val - a_val;
                }
            }

            Eigen::MatrixXf inv_IA;
            bool invertible = true;
            try {
                Eigen::FullPivLU<Eigen::MatrixXf> lu(IA);
                if (lu.isInvertible()) inv_IA = lu.inverse();
                else invertible = false;
            } catch (...) {
                invertible = false;
            }

            size_t result_offset = ((size_t)processed + frag) * m_states * m_states;
            if (invertible) {
                for (int i = 0; i < m_states; i++) {
                    for (int j = 0; j < m_states; j++) {
                        all_inversions[result_offset + i * m_states + j] = inv_IA(i, j);
                    }
                }
            } else {
                for (int i = 0; i < m_states; i++) {
                    for (int j = 0; j < m_states; j++) {
                        all_inversions[result_offset + i * m_states + j] = (i == j) ? 1.0f : 0.0f;
                    }
                }
            }
        }

        processed += ds.nfrag;
        if (processed % 1000 == 0 || processed == total_frags) {
            std::cout << "\rProcessed " << processed << "/" << total_frags
                      << " matrix inversions" << std::flush;
        }
    }

    std::cout << std::endl;
    return all_inversions;
}
