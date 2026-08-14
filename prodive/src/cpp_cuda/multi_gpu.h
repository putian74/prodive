#ifndef MULTI_GPU_H
#define MULTI_GPU_H

#include "kl_divergence.h"
#include <vector>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cuda_runtime.h>

class GPUContext {
public:
    int device_id;
    float *d_pi1, *d_A1, *d_B1, *d_pi2, *d_A2, *d_B2, *d_inv1, *d_inv2, *d_result;
    std::vector<cudaStream_t> streams;
    size_t max_frags;
    int m_states;
    int n_obs;
    int num_streams;

    GPUContext(int id, size_t max_f, int m, int obs, int streams_per_gpu);
    ~GPUContext();

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;
};

struct WorkItem {
    int i, j;
    int dataset_i_idx, dataset_j_idx;
    const FragSet* ds1;
    const FragSet* ds2;
    size_t inv_offset1, inv_offset2;

    WorkItem(int _i, int _j, int _di, int _dj, const FragSet* _ds1, const FragSet* _ds2,
             size_t _io1, size_t _io2);
};

class MultiGPUManager {
private:
    std::vector<std::unique_ptr<GPUContext>> gpu_contexts;
    std::queue<WorkItem> work_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> work_done{false};
    std::atomic<int> pairs_completed{0};

    int num_gpus;
    int m_states;
    int n_obs;
    int num_streams;
    int max_gpus;

    void gpu_worker(int gpu_id, const std::vector<float>& all_inv_matrices,
                    const Config& config, int total_pairs);
    std::vector<size_t> calculate_inverse_offsets(const std::vector<FragSet>& datasets);

public:
    explicit MultiGPUManager(const Config& config);
    ~MultiGPUManager();

    bool initialize(size_t max_frags);
    int get_num_gpus() const { return num_gpus; }
    int get_num_streams() const { return num_streams; }

    size_t get_memory_per_gpu(size_t max_frags) const;
    int populate_work_queue(const Config& config, const std::vector<FragSet>& datasets,
                            size_t max_frags);
    void execute_computation(const std::vector<float>& all_inv_matrices,
                             const Config& config, int total_pairs);
    int get_pairs_completed() const { return pairs_completed.load(); }
    void print_gpu_info() const;
};

#endif // MULTI_GPU_H
