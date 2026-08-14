#include "multi_gpu.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <stdexcept>

// GPUContext implementation
GPUContext::GPUContext(int id, size_t max_f, int m, int obs, int streams_per_gpu)
    : device_id(id), d_pi1(nullptr), d_A1(nullptr), d_B1(nullptr),
      d_pi2(nullptr), d_A2(nullptr), d_B2(nullptr), d_inv1(nullptr), d_inv2(nullptr), d_result(nullptr),
      max_frags(max_f), m_states(m), n_obs(obs), num_streams(streams_per_gpu) {
    cudaSetDevice(device_id);
    
    // Check CUDA errors during allocation
    cudaError_t err;
    
    // Allocate GPU memory with error checking
    err = cudaMalloc(&d_pi1, max_frags * m_states * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_pi1 on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    err = cudaMalloc(&d_A1, max_frags * m_states * m_states * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_A1 on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    err = cudaMalloc(&d_B1, max_frags * m_states * n_obs * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_B1 on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    err = cudaMalloc(&d_pi2, max_frags * m_states * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_pi2 on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    err = cudaMalloc(&d_A2, max_frags * m_states * m_states * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_A2 on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    err = cudaMalloc(&d_B2, max_frags * m_states * n_obs * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_B2 on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    err = cudaMalloc(&d_inv1, max_frags * m_states * m_states * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_inv1 on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    err = cudaMalloc(&d_inv2, max_frags * m_states * m_states * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_inv2 on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    err = cudaMalloc(&d_result, max_frags * max_frags * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_result on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
    
    // Create streams with error checking
    streams.resize(num_streams);
    for (int s = 0; s < num_streams; ++s) {
        err = cudaStreamCreate(&streams[s]);
        if (err != cudaSuccess) {
            std::cerr << "Failed to create stream " << s << " on GPU " << device_id << ": " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA stream creation failed");
        }
    }
    
    std::cout << "GPU " << device_id << " initialized successfully" << std::endl;
}

GPUContext::~GPUContext() {
    cudaSetDevice(device_id);
    
    // Destroy streams
    for (auto& stream : streams) {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
    
    // Free GPU memory (check for null pointers)
    if (d_pi1) cudaFree(d_pi1);
    if (d_A1) cudaFree(d_A1);
    if (d_B1) cudaFree(d_B1);
    if (d_pi2) cudaFree(d_pi2);
    if (d_A2) cudaFree(d_A2);
    if (d_B2) cudaFree(d_B2);
    if (d_inv1) cudaFree(d_inv1);
    if (d_inv2) cudaFree(d_inv2);
    if (d_result) cudaFree(d_result);
}

// WorkItem implementation
WorkItem::WorkItem(int _i, int _j, int _di, int _dj, const FragSet* _ds1, const FragSet* _ds2, 
                   size_t _io1, size_t _io2) 
    : i(_i), j(_j), dataset_i_idx(_di), dataset_j_idx(_dj), ds1(_ds1), ds2(_ds2),
      inv_offset1(_io1), inv_offset2(_io2) {}

// MultiGPUManager implementation
MultiGPUManager::MultiGPUManager(const Config& config)
    : num_gpus(0), m_states(config.m_states), n_obs(config.n_obs),
      num_streams(config.num_streams), max_gpus(config.max_gpus) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    num_gpus = std::min(deviceCount, max_gpus);
}

MultiGPUManager::~MultiGPUManager() {
    // GPU contexts will be automatically cleaned up
}

bool MultiGPUManager::initialize(size_t max_frags) {
    if (num_gpus <= 0) {
        std::cerr << "No CUDA devices available!" << std::endl;
        return false;
    }
    
    // Initialize GPU contexts
    gpu_contexts.clear();
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        try {
            gpu_contexts.push_back(std::make_unique<GPUContext>(gpu, max_frags, m_states, n_obs, num_streams));
            std::cout << "Initialized GPU " << gpu << " with " << num_streams << " streams" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize GPU " << gpu << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    return true;
}

size_t MultiGPUManager::get_memory_per_gpu(size_t max_frags) const {
    return max_frags * (2 * m_states + 2 * m_states * m_states + 
                       2 * m_states * n_obs + max_frags) * sizeof(float);
}

void MultiGPUManager::print_gpu_info() const {
    for (int dev = 0; dev < num_gpus; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "GPU " << dev << ": " << prop.name 
                  << " (Compute capability: " << prop.major << "." << prop.minor 
                  << ", Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB)" << std::endl;
    }
}

// Helper function to calculate cumulative inverse matrix offsets
std::vector<size_t> MultiGPUManager::calculate_inverse_offsets(const std::vector<FragSet>& datasets) {
    std::vector<size_t> offsets(datasets.size());
    size_t current_offset = 0;
    
    for (size_t i = 0; i < datasets.size(); ++i) {
        offsets[i] = current_offset;
        current_offset += datasets[i].nfrag * m_states * m_states;
    }
    
    std::cout << "Calculated inverse matrix offsets for " << datasets.size() << " datasets" << std::endl;
    std::cout << "Total inverse matrices size: " << current_offset << " floats" << std::endl;
    
    return offsets;
}

int MultiGPUManager::populate_work_queue(const Config& config, const std::vector<FragSet>& datasets, 
                                         size_t max_frags) {
    std::cout << "Populating work queue..." << std::endl;
    
    // Calculate correct inverse matrix offsets
    std::vector<size_t> inverse_offsets = calculate_inverse_offsets(datasets);
    
    int queue_size = 0;
    
    // Clear any existing work
    std::queue<WorkItem> empty;
    work_queue.swap(empty);

    auto result_file_exists = [&](int i, int j) -> bool {
        std::ostringstream check_fname;
        check_fname << config.outdir << "kl_" 
                    << std::setfill('0') << std::setw(4) << i 
                    << "_" << std::setfill('0') << std::setw(4) << j;
        std::string check_file = check_fname.str() + (config.save_as_text ? ".txt" : ".npy");
        return file_exists(check_file);
    };

    auto enqueue_pair = [&](int i, int j) -> bool {
        // Skip if resuming and file already exists.
        if (config.resume_computation && result_file_exists(i, j)) {
            return false;
        }

        // Check if datasets were loaded.
        if (config.dataset_index_map.find(i) == config.dataset_index_map.end() ||
            config.dataset_index_map.find(j) == config.dataset_index_map.end()) {
            std::cerr << "⚠️  Skipping pair (" << i << ", " << j << "): Dataset not loaded" << std::endl;
            return false;
        }

        int dataset_i_idx = config.dataset_index_map.at(i);
        int dataset_j_idx = config.dataset_index_map.at(j);

        // Validate dataset indices.
        if (dataset_i_idx >= (int)datasets.size() || dataset_j_idx >= (int)datasets.size() ||
            dataset_i_idx >= (int)inverse_offsets.size() || dataset_j_idx >= (int)inverse_offsets.size()) {
            std::cerr << "⚠️  Skipping pair (" << i << ", " << j << "): Invalid dataset indices "
                      << dataset_i_idx << ", " << dataset_j_idx << std::endl;
            return false;
        }

        const FragSet& ds1 = datasets[dataset_i_idx];
        const FragSet& ds2 = datasets[dataset_j_idx];

        // Use correct offsets from cumulative calculation.
        size_t inv_offset1 = inverse_offsets[dataset_i_idx];
        size_t inv_offset2 = inverse_offsets[dataset_j_idx];

        work_queue.emplace(i, j, dataset_i_idx, dataset_j_idx, &ds1, &ds2, inv_offset1, inv_offset2);
        return true;
    };

    if (config.use_pair_list) {
        std::cout << "Pair-list mode: adding explicit sampled pairs to work queue..." << std::endl;
        for (const auto& p : config.pair_list) {
            if (enqueue_pair(p.first, p.second)) {
                queue_size++;
            }
        }
        std::cout << "Work queue populated with " << queue_size
                  << " sampled pair-list tasks" << std::endl;
        return queue_size;
    }
    
    // Original range mode.
    for (int i = config.start_i; i < config.end_i; ++i) {
        // Determine j range based on symmetric computation.
        int j_start, j_end;
        if (config.reconstruct_symmetric) {
            j_start = std::max(i, config.start_j);
            j_end = config.end_j;
        } else {
            j_start = config.start_j;
            j_end = config.end_j;
        }
        
        for (int j = j_start; j < j_end; ++j) {
            if (enqueue_pair(i, j)) {
                queue_size++;
            }
        }
    }
    
    std::cout << "Work queue populated with " << queue_size << " tasks" << std::endl;
    return queue_size;
}

// Also update gpu_worker validation to be more informative
void MultiGPUManager::gpu_worker(int gpu_id, const std::vector<float>& all_inv_matrices, 
                                 const Config& config, int total_pairs) {
    try {
        cudaSetDevice(gpu_id);
        GPUContext& gpu_ctx = *gpu_contexts[gpu_id];
        
        int stream_idx = 0;
        int local_processed = 0;
        
        std::cout << "GPU " << gpu_id << " worker started (total inverse matrices: " 
                  << all_inv_matrices.size() << " floats)" << std::endl;
        
        while (true) {
            WorkItem work_item(0, 0, 0, 0, nullptr, nullptr, 0, 0);
            
            // Get work item from queue (thread-safe)
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this]{ return !work_queue.empty() || work_done.load(); });
                
                if (work_done.load() && work_queue.empty()) {
                    break; // No more work
                }
                
                if (!work_queue.empty()) {
                    work_item = work_queue.front();
                    work_queue.pop();
                } else {
                    continue; // Spurious wakeup
                }
            }
            
            // Validate work item
            if (work_item.ds1 == nullptr || work_item.ds2 == nullptr) {
                std::cerr << "GPU " << gpu_id << ": Invalid work item - null dataset pointers" << std::endl;
                continue;
            }
            
            // Use round-robin stream selection
            cudaStream_t current_stream = gpu_ctx.streams[stream_idx];
            stream_idx = (stream_idx + 1) % num_streams;
            
            int nfrag1 = work_item.ds1->nfrag;
            int nfrag2 = work_item.ds2->nfrag;
            
            // Validate fragment counts
            if (nfrag1 <= 0 || nfrag2 <= 0) {
                std::cerr << "GPU " << gpu_id << ": Invalid fragment counts: " << nfrag1 << ", " << nfrag2 << std::endl;
                continue;
            }
            
            // Check memory bounds
            if (nfrag1 > gpu_ctx.max_frags || nfrag2 > gpu_ctx.max_frags) {
                std::cerr << "GPU " << gpu_id << ": Fragment count exceeds allocated memory: " 
                          << nfrag1 << " or " << nfrag2 << " > " << gpu_ctx.max_frags << std::endl;
                continue;
            }
            
            // Validate data sizes
            if (work_item.ds1->pi.size() < (size_t)(nfrag1 * m_states) ||
                work_item.ds1->A.size() < (size_t)(nfrag1 * m_states * m_states) ||
                work_item.ds1->B.size() < (size_t)(nfrag1 * m_states * n_obs) ||
                work_item.ds2->pi.size() < (size_t)(nfrag2 * m_states) ||
                work_item.ds2->A.size() < (size_t)(nfrag2 * m_states * m_states) ||
                work_item.ds2->B.size() < (size_t)(nfrag2 * m_states * n_obs)) {
                std::cerr << "GPU " << gpu_id << ": Dataset vector size mismatch for pair (" 
                          << work_item.i << ", " << work_item.j << ")" << std::endl;
                std::cerr << "  ds1 sizes: pi=" << work_item.ds1->pi.size() 
                          << " A=" << work_item.ds1->A.size() << " B=" << work_item.ds1->B.size() << std::endl;
                std::cerr << "  ds2 sizes: pi=" << work_item.ds2->pi.size() 
                          << " A=" << work_item.ds2->A.size() << " B=" << work_item.ds2->B.size() << std::endl;
                std::cerr << "  Required: pi=" << (nfrag1 * m_states) 
                          << " A=" << (nfrag1 * m_states * m_states) 
                          << " B=" << (nfrag1 * m_states * n_obs) << std::endl;
                continue;
            }
            
            // Validate inverse matrix offsets with detailed debugging
            size_t required_inv_size1 = work_item.inv_offset1 + nfrag1 * m_states * m_states;
            size_t required_inv_size2 = work_item.inv_offset2 + nfrag2 * m_states * m_states;
            
            if (required_inv_size1 > all_inv_matrices.size() || 
                required_inv_size2 > all_inv_matrices.size()) {
                std::cerr << "GPU " << gpu_id << ": Inverse matrix offset out of bounds for pair (" 
                          << work_item.i << ", " << work_item.j << ")" << std::endl;
                std::cerr << "  Dataset indices: " << work_item.dataset_i_idx << ", " << work_item.dataset_j_idx << std::endl;
                std::cerr << "  Fragment counts: " << nfrag1 << ", " << nfrag2 << std::endl;
                std::cerr << "  Offsets: " << work_item.inv_offset1 << ", " << work_item.inv_offset2 << std::endl;
                std::cerr << "  Required sizes: " << required_inv_size1 << ", " << required_inv_size2 << std::endl;
                std::cerr << "  Available size: " << all_inv_matrices.size() << std::endl;
                continue;
            }
            
            // Asynchronous data upload with error checking
            cudaError_t err;
            
            err = cudaMemcpyAsync(gpu_ctx.d_pi1, work_item.ds1->pi.data(), 
                                 nfrag1 * m_states * sizeof(float), 
                                 cudaMemcpyHostToDevice, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for pi1: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            err = cudaMemcpyAsync(gpu_ctx.d_A1, work_item.ds1->A.data(), 
                                 nfrag1 * m_states * m_states * sizeof(float), 
                                 cudaMemcpyHostToDevice, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for A1: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            err = cudaMemcpyAsync(gpu_ctx.d_B1, work_item.ds1->B.data(), 
                                 nfrag1 * m_states * n_obs * sizeof(float), 
                                 cudaMemcpyHostToDevice, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for B1: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            err = cudaMemcpyAsync(gpu_ctx.d_pi2, work_item.ds2->pi.data(), 
                                 nfrag2 * m_states * sizeof(float), 
                                 cudaMemcpyHostToDevice, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for pi2: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            err = cudaMemcpyAsync(gpu_ctx.d_A2, work_item.ds2->A.data(), 
                                 nfrag2 * m_states * m_states * sizeof(float), 
                                 cudaMemcpyHostToDevice, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for A2: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            err = cudaMemcpyAsync(gpu_ctx.d_B2, work_item.ds2->B.data(), 
                                 nfrag2 * m_states * n_obs * sizeof(float), 
                                 cudaMemcpyHostToDevice, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for B2: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            // Upload precomputed inverse matrices
            err = cudaMemcpyAsync(gpu_ctx.d_inv1, &all_inv_matrices[work_item.inv_offset1], 
                                 nfrag1 * m_states * m_states * sizeof(float), 
                                 cudaMemcpyHostToDevice, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for inv1: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            err = cudaMemcpyAsync(gpu_ctx.d_inv2, &all_inv_matrices[work_item.inv_offset2], 
                                 nfrag2 * m_states * m_states * sizeof(float), 
                                 cudaMemcpyHostToDevice, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for inv2: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            // Launch kernel asynchronously
            sym_kl_kernel_with_precomputed_inv(
                gpu_ctx.d_pi1, gpu_ctx.d_A1, gpu_ctx.d_B1, 
                gpu_ctx.d_pi2, gpu_ctx.d_A2, gpu_ctx.d_B2, 
                gpu_ctx.d_inv1, gpu_ctx.d_inv2,
                gpu_ctx.d_result, nfrag1, nfrag2, m_states, n_obs,
                current_stream
            );
            
            // Check for kernel launch errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            // Allocate result buffer
            std::vector<float> result_data(nfrag1 * nfrag2);
            
            // Asynchronous data download
            err = cudaMemcpyAsync(result_data.data(), gpu_ctx.d_result, 
                                 nfrag1 * nfrag2 * sizeof(float), 
                                 cudaMemcpyDeviceToHost, current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaMemcpyAsync failed for result: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            // Wait for this stream to complete
            err = cudaStreamSynchronize(current_stream);
            if (err != cudaSuccess) {
                std::cerr << "GPU " << gpu_id << ": cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            // Save result
            std::ostringstream fname;
            fname << config.outdir << "kl_" 
                  << std::setfill('0') << std::setw(4) << work_item.i 
                  << "_" << std::setfill('0') << std::setw(4) << work_item.j;
            
            try {
                if (config.save_as_text) {
                    save_matrix_text(fname.str() + ".txt", result_data.data(), nfrag1, nfrag2);
                } else {
                    save_matrix_npy(fname.str() + ".npy", result_data.data(), nfrag1, nfrag2);
                }
            } catch (const std::exception& e) {
                std::cerr << "GPU " << gpu_id << ": Failed to save result for pair (" 
                          << work_item.i << ", " << work_item.j << "): " << e.what() << std::endl;
                continue;
            }
            
            // Update progress (thread-safe)
            local_processed++;
            int completed = pairs_completed.fetch_add(1) + 1;
            if (completed % 50 == 0) {  // Reduce progress update frequency
                std::cout << "\rGPU " << gpu_id << " - Processed " << completed 
                          << "/" << total_pairs << " pairs (local: " << local_processed << ")" << std::flush;
            }
        }
        
        std::cout << "\nGPU " << gpu_id << " worker finished - processed " << local_processed << " pairs" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "GPU " << gpu_id << " worker crashed: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "GPU " << gpu_id << " worker crashed with unknown exception" << std::endl;
    }
}

void MultiGPUManager::execute_computation(const std::vector<float>& all_inv_matrices, 
                                         const Config& config, int total_pairs) {
    // Reset completion counter
    pairs_completed.store(0);
    work_done.store(false);
    
    // Start GPU worker threads
    std::cout << "Starting " << num_gpus << " GPU worker threads..." << std::endl;
    
    std::vector<std::thread> workers;
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        workers.emplace_back(&MultiGPUManager::gpu_worker, this, gpu, 
                            std::cref(all_inv_matrices), std::cref(config), total_pairs);
    }
    
    // Signal work completion when queue is empty
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        work_done.store(true);
    }
    queue_cv.notify_all();
    
    // Wait for all workers to finish
    for (auto& worker : workers) {
        worker.join();
    }
}