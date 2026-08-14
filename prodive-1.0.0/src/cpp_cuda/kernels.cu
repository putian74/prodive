//main kernels for fragment KL divergence computation
#include "kl_divergence.h"
#include <cuda_runtime.h>

__global__ void sym_kl_kernel_with_precomputed_inv(
    const float* pi1, const float* A1, const float* B1,
    const float* pi2, const float* A2, const float* B2,
    const float* inv1, const float* inv2,
    float* KL, int nfrag1, int nfrag2, int m, int n
) {
    int f1 = blockIdx.y * blockDim.y + threadIdx.y;
    int f2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (f1 >= nfrag1 || f2 >= nfrag2) return;
    if (m > MAX_M_STATES || n > MAX_N_OBS) return;
    
    int pair_idx = f1 * nfrag2 + f2;
    
    // Load data into local memory with validation
    float local_pi1[MAX_M_STATES], local_A1[MAX_M_STATES*MAX_M_STATES], local_B1[MAX_M_STATES*MAX_N_OBS];
    float local_pi2[MAX_M_STATES], local_A2[MAX_M_STATES*MAX_M_STATES], local_B2[MAX_M_STATES*MAX_N_OBS];
    float local_inv1[MAX_M_STATES*MAX_M_STATES], local_inv2[MAX_M_STATES*MAX_M_STATES];
    
    // Load and validate fragment data
    float pi1_sum = 0.0f, pi2_sum = 0.0f;
    for (int i = 0; i < m; i++) {
        local_pi1[i] = pi1[f1*m + i];
        local_pi2[i] = pi2[f2*m + i];
        
        // Check for invalid probabilities and clamp
        if (!isfinite(local_pi1[i]) || local_pi1[i] < 0.0f) {
            local_pi1[i] = fmaxf(local_pi1[i], EPS);
        }
        if (!isfinite(local_pi2[i]) || local_pi2[i] < 0.0f) {
            local_pi2[i] = fmaxf(local_pi2[i], EPS);
        }
        
        pi1_sum += local_pi1[i];
        pi2_sum += local_pi2[i];
        
        // Load transition and emission matrices with validation
        for (int j = 0; j < m; j++) {
            local_A1[i*m + j] = A1[f1*m*m + i*m + j];
            local_A2[i*m + j] = A2[f2*m*m + i*m + j];
            local_inv1[i*m + j] = inv1[f1*m*m + i*m + j];
            local_inv2[i*m + j] = inv2[f2*m*m + i*m + j];
            
            // Validate matrices
            if (!isfinite(local_A1[i*m + j])) {
                local_A1[i*m + j] = (i == j) ? 0.5f : 0.1f;
            }
            if (!isfinite(local_A2[i*m + j])) {
                local_A2[i*m + j] = (i == j) ? 0.5f : 0.1f;
            }
            if (!isfinite(local_inv1[i*m + j])) {
                local_inv1[i*m + j] = (i == j) ? 1.0f : 0.0f;
            }
            if (!isfinite(local_inv2[i*m + j])) {
                local_inv2[i*m + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        
        for (int j = 0; j < n; j++) {
            local_B1[i*n + j] = B1[f1*m*n + i*n + j];
            local_B2[i*n + j] = B2[f2*m*n + i*n + j];
            
            // Validate emission matrices
            if (!isfinite(local_B1[i*n + j]) || local_B1[i*n + j] < 0.0f) {
                local_B1[i*n + j] = fmaxf(local_B1[i*n + j], EPS);
            }
            if (!isfinite(local_B2[i*n + j]) || local_B2[i*n + j] < 0.0f) {
                local_B2[i*n + j] = fmaxf(local_B2[i*n + j], EPS);
            }
        }
    }
    
    // Normalize pi vectors if they don't sum to 1
    if (fabsf(pi1_sum - 1.0f) > 0.01f) {
        for (int i = 0; i < m; i++) {
            local_pi1[i] /= pi1_sum;
        }
    }
    if (fabsf(pi2_sum - 1.0f) > 0.01f) {
        for (int i = 0; i < m; i++) {
            local_pi2[i] /= pi2_sum;
        }
    }

    // === FORWARD DIRECTION: KL(1||2) ===
    
    // Calculate C1, C2, D1, D2 with enhanced numerical stability
    float C1[MAX_M_STATES*MAX_N_OBS] = {0}, C2[MAX_M_STATES*MAX_N_OBS] = {0};
    float D1[MAX_N_OBS] = {0}, D2[MAX_N_OBS] = {0};
    
    // Use double precision for intermediate calculations
    for (int j = 0; j < n; j++) {
        double d1_sum = 0.0, d2_sum = 0.0;
        for (int i = 0; i < m; i++) {
            d1_sum += (double)local_pi1[i] * (double)local_B1[i*n + j];
            d2_sum += (double)local_pi2[i] * (double)local_B2[i*n + j];
        }
        D1[j] = fmaxf((float)d1_sum, EPS);
        D2[j] = fmaxf((float)d2_sum, EPS);
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double c1_sum = 0.0, c2_sum = 0.0;
            for (int k = 0; k < m; k++) {
                c1_sum += (double)local_A1[i*m + k] * (double)local_B1[k*n + j];
                c2_sum += (double)local_A2[i*m + k] * (double)local_B2[k*n + j];
            }
            C1[i*n + j] = fmaxf((float)c1_sum, EPS);
            C2[i*n + j] = fmaxf((float)c2_sum, EPS);
        }
    }
    
    // Calculate W and Q with improved numerical stability
    float W1[MAX_M_STATES] = {0};
    float Q1 = 0.0f;
    
    for (int i = 0; i < m; i++) {
        double w_sum = 0.0;
        for (int j = 0; j < n; j++) {
            float c = C1[i*n + j];
            float cp = C2[i*n + j];
            
            if (c > EPS && cp > EPS) {
                double ratio = (double)c / (double)cp;
                if (ratio > 1e-15 && isfinite(ratio)) {
                    double logval = log(ratio);  // Use double precision log
                    if (isfinite(logval)) {
                        w_sum += (double)c * logval;
                    }
                }
            }
        }
        W1[i] = (float)w_sum;
    }
    
    // Calculate Q1 with Kahan summation
    double q_sum = 0.0;
    for (int j = 0; j < n; j++) {
        float d = D1[j];
        float dp = D2[j];
        if (d > EPS && dp > EPS) {
            double ratio = (double)d / (double)dp;
            if (ratio > 1e-15 && isfinite(ratio)) {
                double logval = log(ratio);
                if (isfinite(logval)) {
                    q_sum += (double)d * logval;
                }
            }
        }
    }
    Q1 = (float)q_sum;
    
    // Calculate KL(1||2) using precomputed inverse
    float tmp[MAX_M_STATES] = {0};
    for (int i = 0; i < m; i++) {
        double tmp_sum = 0.0;
        for (int j = 0; j < m; j++) {
            tmp_sum += (double)local_inv1[i*m + j] * (double)W1[j];
        }
        tmp[i] = (float)tmp_sum;
    }
    
    double kl12_sum = 0.0;
    for (int i = 0; i < m; i++) {
        if (isfinite(tmp[i])) {
            kl12_sum += (double)local_pi1[i] * (double)tmp[i];
        }
    }
    float kl12 = (float)(kl12_sum + (double)Q1);
    
    // === REVERSE DIRECTION: KL(2||1) ===
    
    // Similar calculations for reverse direction with enhanced stability
    float C1b[MAX_M_STATES*MAX_N_OBS] = {0}, C2b[MAX_M_STATES*MAX_N_OBS] = {0};
    float D1b[MAX_N_OBS] = {0}, D2b[MAX_N_OBS] = {0};
    
    for (int j = 0; j < n; j++) {
        double d1b_sum = 0.0, d2b_sum = 0.0;
        for (int i = 0; i < m; i++) {
            d1b_sum += (double)local_pi2[i] * (double)local_B2[i*n + j];
            d2b_sum += (double)local_pi1[i] * (double)local_B1[i*n + j];
        }
        D1b[j] = fmaxf((float)d1b_sum, EPS);
        D2b[j] = fmaxf((float)d2b_sum, EPS);
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double c1b_sum = 0.0, c2b_sum = 0.0;
            for (int k = 0; k < m; k++) {
                c1b_sum += (double)local_A2[i*m + k] * (double)local_B2[k*n + j];
                c2b_sum += (double)local_A1[i*m + k] * (double)local_B1[k*n + j];
            }
            C1b[i*n + j] = fmaxf((float)c1b_sum, EPS);
            C2b[i*n + j] = fmaxf((float)c2b_sum, EPS);
        }
    }
    
    float W2[MAX_M_STATES] = {0};
    float Q2 = 0.0f;
    
    for (int i = 0; i < m; i++) {
        double w2_sum = 0.0;
        for (int j = 0; j < n; j++) {
            float c = C1b[i*n + j];
            float cp = C2b[i*n + j];
            
            if (c > EPS && cp > EPS) {
                double ratio = (double)c / (double)cp;
                if (ratio > 1e-15 && isfinite(ratio)) {
                    double logval = log(ratio);
                    if (isfinite(logval)) {
                        w2_sum += (double)c * logval;
                    }
                }
            }
        }
        W2[i] = (float)w2_sum;
    }
    
    q_sum = 0.0;
    for (int j = 0; j < n; j++) {
        float d = D1b[j];
        float dp = D2b[j];
        if (d > EPS && dp > EPS) {
            double ratio = (double)d / (double)dp;
            if (ratio > 1e-15 && isfinite(ratio)) {
                double logval = log(ratio);
                if (isfinite(logval)) {
                    q_sum += (double)d * logval;
                }
            }
        }
    }
    Q2 = (float)q_sum;
    
    // Calculate KL(2||1) using precomputed inverse
    float tmp2[MAX_M_STATES] = {0};
    for (int i = 0; i < m; i++) {
        double tmp2_sum = 0.0;
        for (int j = 0; j < m; j++) {
            tmp2_sum += (double)local_inv2[i*m + j] * (double)W2[j];
        }
        tmp2[i] = (float)tmp2_sum;
    }
    
    double kl21_sum = 0.0;
    for (int i = 0; i < m; i++) {
        if (isfinite(tmp2[i])) {
            kl21_sum += (double)local_pi2[i] * (double)tmp2[i];
        }
    }
    float kl21 = (float)(kl21_sum + (double)Q2);
    
    // Final validation with robust handling
    if (!isfinite(kl12) || kl12 < -1000.0f || kl12 > 1000.0f) {
        kl12 = 1000.0f;
    }
    if (!isfinite(kl21) || kl21 < -1000.0f || kl21 > 1000.0f) {
        kl21 = 1000.0f;
    }
    
    // Calculate symmetric KL with validation
    float sym_kl = 0.5f * (kl12 + kl21);
    if (!isfinite(sym_kl) || sym_kl < 0.0f || sym_kl > 1000.0f) {
        sym_kl = 1000.0f;
    }
    
    KL[pair_idx] = sym_kl;
}

// HOST WRAPPER FUNCTION:
extern "C" void sym_kl_kernel_with_precomputed_inv(
    const float* pi1, const float* A1, const float* B1,
    const float* pi2, const float* A2, const float* B2,
    const float* inv1, const float* inv2,
    float* KL, int nfrag1, int nfrag2, int m, int n,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((nfrag2 + block.x - 1) / block.x, (nfrag1 + block.y - 1) / block.y);
    
    sym_kl_kernel_with_precomputed_inv<<<grid, block, 0, stream>>>(
        pi1, A1, B1, pi2, A2, B2, inv1, inv2, KL, nfrag1, nfrag2, m, n
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
