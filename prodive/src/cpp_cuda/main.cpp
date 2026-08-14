// main function for computing KL divergence through CUDA kernels
#include "kl_divergence.h"
#include "multi_gpu.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <set>
#include <algorithm>

int main(int argc, char* argv[]) {
    // Parse command line arguments
    Config config = parse_command_line(argc, argv);

    // Load explicit sampled pair list before dataset loading.
    if (config.use_pair_list) {
        try {
            config.pair_list = load_pair_list(config.pair_list_file);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load pair list: " << e.what() << std::endl;
            return 1;
        }
    }
    
    PackedDatasetIndex packed_index;
    if (config.use_packed_db) {
        try {
            PackedRuntimeInfo runtime_info = load_packed_runtime_info_from_metadata(config.packed_db_dir);
            apply_packed_runtime_info(config, runtime_info);
            packed_index = load_packed_dataset_index(config.packed_db_dir);
            std::cout << "Packed DB runtime info from metadata.json: fragment=" << config.fragment
                      << ", m_states=" << config.m_states
                      << ", n_obs=" << config.n_obs
                      << ", num_datasets=" << config.num_datasets
                      << ", min_frag=" << config.min_frag
                      << ", max_frag=" << config.max_frag << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load packed database metadata/index: " << e.what() << std::endl;
            return 1;
        }
    }

    if (config.end_i < 0) config.end_i = config.num_datasets;
    if (config.end_j < 0) config.end_j = config.num_datasets;

    if (config.use_pair_list) {
        try {
            validate_pair_list_against_num_datasets(config.pair_list, config.num_datasets);
        } catch (const std::exception& e) {
            std::cerr << "Pair list validation failed: " << e.what() << std::endl;
            return 1;
        }
    } else {
        if (config.start_i >= config.end_i || config.start_j >= config.end_j) {
            std::cerr << "Error: invalid range after runtime metadata applied." << std::endl;
            return 1;
        }
        if (config.start_i < 0 || config.start_j < 0 ||
            config.end_i > config.num_datasets || config.end_j > config.num_datasets) {
            std::cerr << "Error: range indices must be within [0, " << config.num_datasets << ")." << std::endl;
            return 1;
        }
    }

    // Validate and fix paths
    if (config.outdir.substr(0, 2) == "./") {
        std::cerr << "Warning: Converting relative path to absolute path" << std::endl;
        std::cerr << "Original: " << config.outdir << std::endl;
        config.outdir = config.outdir.substr(2); // Remove "./"
        std::cerr << "Fixed: " << config.outdir << std::endl;
    }
    
    // Ensure outdir ends with slash
    if (config.outdir.back() != '/') {
        config.outdir += '/';
    }
    
    // Create output directory
    if (!create_directory_if_not_exists(config.outdir)) {
        std::cerr << "Failed to create output directory: " << config.outdir << std::endl;
        return 1;
    }
    
    // Initialize multi-GPU manager after runtime dimensions are known
    MultiGPUManager gpu_manager(config);
    
    // Print configuration summary
    std::cout << "=== Configuration Summary ===" << std::endl;
    std::cout << "Output directory: " << config.outdir << std::endl;
    if (config.use_packed_db) {
        std::cout << "Data source: packed database" << std::endl;
        std::cout << "Packed DB directory: " << config.packed_db_dir << std::endl;
    } else {
        std::cout << "Data source: small NPY files" << std::endl;
        std::cout << "Data directory: " << config.datadir << std::endl;
    }

    if (config.use_pair_list) {
        std::cout << "Computation mode: explicit pair list" << std::endl;
        std::cout << "Pair list file: " << config.pair_list_file << std::endl;
        std::cout << "Pair list size: " << config.pair_list.size() << " unique pairs" << std::endl;
    } else {
        std::cout << "Computation mode: range" << std::endl;
        std::cout << "Computation range: i=[" << config.start_i << "," << config.end_i 
                  << "), j=[" << config.start_j << "," << config.end_j << ")" << std::endl;
    }

    std::cout << "Fragment length: " << config.fragment << std::endl;
    std::cout << "Runtime m_states: " << config.m_states << std::endl;
    std::cout << "Runtime n_obs: " << config.n_obs << std::endl;
    std::cout << "Runtime num_datasets: " << config.num_datasets << std::endl;
    std::cout << "Runtime min/max nfrag: " << config.min_frag << "/" << config.max_frag << std::endl;
    std::cout << "Matrix inversion: CPU (Eigen) with precomputed matrices" << std::endl;
    std::cout << "Memory efficient: " << (config.memory_efficient ? "Yes" : "No") << std::endl;
    std::cout << "Resume computation: " << (config.resume_computation ? "Yes" : "No") << std::endl;
    std::cout << "Save format: " << (config.save_as_text ? "Text" : "NPY") << std::endl;
    std::cout << "Reconstruct symmetric: " << (config.reconstruct_symmetric ? "Yes" : "No") << std::endl;
    std::cout << "GPUs: " << gpu_manager.get_num_gpus() << " devices, " << gpu_manager.get_num_streams() << " streams per GPU" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Print GPU information
    gpu_manager.print_gpu_info();
    
    // Load datasets
    std::cout << "Loading datasets..." << std::endl;
    
    std::vector<FragSet> datasets;
    std::vector<int> loaded_indices;

    auto has_dataset = [&](int idx) -> bool {
        if (config.use_packed_db) {
            return check_packed_dataset(packed_index, idx);
        }
        return check_dataset_files(config.datadir, idx);
    };

    auto load_one_dataset = [&](int idx) -> FragSet {
        if (config.use_packed_db) {
            return load_packed_dataset(config.packed_db_dir, packed_index, idx, config.m_states, config.n_obs);
        }
        return load_dataset(config.datadir, idx, config.m_states, config.n_obs);
    };

    if (config.use_pair_list) {
        // Pair-list mode always loads only families that appear in the sampled pairs.
        std::cout << "Pair-list mode: Loading only datasets used by sampled pairs..." << std::endl;

        std::set<int> needed_indices;
        for (const auto& p : config.pair_list) {
            needed_indices.insert(p.first);
            needed_indices.insert(p.second);
        }

        int loaded_idx = 0;
        int missing_count = 0;
        int load_error_count = 0;
        for (int idx : needed_indices) {
            if (!has_dataset(idx)) {
                missing_count++;
                std::cerr << "Warning: dataset missing for family index " << idx << std::endl;
                continue;
            }

            try {
                datasets.push_back(load_one_dataset(idx));
                loaded_indices.push_back(idx);
                config.dataset_index_map[idx] = loaded_idx;
                config.reverse_dataset_map.push_back(idx);
                loaded_idx++;
            } catch (const std::exception& e) {
                load_error_count++;
                std::cerr << "Warning: failed to load dataset " << idx << ": " << e.what() << std::endl;
            }
        }

        std::cout << "Loaded " << datasets.size() << "/" << needed_indices.size()
                  << " datasets needed by pair list" << std::endl;
        if (missing_count > 0 || load_error_count > 0) {
            std::cerr << "Warning: " << missing_count << " missing and " << load_error_count
                      << " failed pair-list family IDs. Pairs containing them will be skipped." << std::endl;
        }
    } else if (config.memory_efficient) {
        // Memory efficient mode - load only needed datasets for range mode.
        std::cout << "Memory-efficient mode: Loading datasets for computation range..." << std::endl;
        
        std::set<int> needed_indices;
        for (int i = config.start_i; i < config.end_i; ++i) {
            needed_indices.insert(i);
        }
        for (int j = config.start_j; j < config.end_j; ++j) {
            needed_indices.insert(j);
        }
        
        int loaded_idx = 0;
        int missing_count = 0;
        int load_error_count = 0;
        for (int idx : needed_indices) {
            if (!has_dataset(idx)) {
                missing_count++;
                continue;
            }

            try {
                datasets.push_back(load_one_dataset(idx));
                loaded_indices.push_back(idx);
                config.dataset_index_map[idx] = loaded_idx;
                config.reverse_dataset_map.push_back(idx);
                loaded_idx++;
            } catch (const std::exception& e) {
                load_error_count++;
                std::cerr << "Warning: failed to load dataset " << idx << ": " << e.what() << std::endl;
            }
        }
        
        std::cout << "Loaded " << datasets.size() << "/" << needed_indices.size() << " datasets" << std::endl;
        if (missing_count > 0 || load_error_count > 0) {
            std::cerr << "Warning: " << missing_count << " missing and " << load_error_count
                      << " failed datasets in requested range." << std::endl;
        }
    } else {
        // Load all datasets in range. This mode can be very memory-heavy.
        std::cout << "Standard mode: Loading all datasets for computation range..." << std::endl;
        
        int max_idx = std::max(config.end_i, config.end_j);
        int load_error_count = 0;
        for (int idx = 0; idx < max_idx; ++idx) {
            if (!has_dataset(idx)) {
                continue;
            }

            try {
                datasets.push_back(load_one_dataset(idx));
                loaded_indices.push_back(idx);
                config.dataset_index_map[idx] = datasets.size() - 1;
            } catch (const std::exception& e) {
                load_error_count++;
                std::cerr << "Warning: failed to load dataset " << idx << ": " << e.what() << std::endl;
            }
        }
        
        std::cout << "Loaded " << datasets.size() << "/" << max_idx << " datasets" << std::endl;
        if (load_error_count > 0) {
            std::cerr << "Warning: " << load_error_count << " datasets failed during loading." << std::endl;
        }
    }
    
    if (datasets.empty()) {
        std::cerr << "No datasets loaded! Check data directory: " << config.datadir << std::endl;
        return 1;
    }
    
    // Show sample fragment counts
    std::cout << "Loaded " << datasets.size() << " datasets with fragment counts: ";
    for (int i = 0; i < std::min(10, (int)datasets.size()); ++i) {
        std::cout << datasets[i].nfrag;
        if (i < std::min(9, (int)datasets.size() - 1)) std::cout << ", ";
    }
    if (datasets.size() > 10) std::cout << "...";
    std::cout << std::endl;
    
    // Find maximum fragment count for memory allocation
    size_t max_frags = 0;
    for (const auto& ds : datasets) {
        max_frags = std::max(max_frags, (size_t)ds.nfrag);
    }
    
    // Calculate memory requirements
    size_t memory_per_gpu = gpu_manager.get_memory_per_gpu(max_frags);
    std::cout << "GPU memory per device: ";
    print_memory_usage(memory_per_gpu);
    std::cout << std::endl;
    
    // Initialize GPU contexts
    if (!gpu_manager.initialize(max_frags)) {
        std::cerr << "Failed to initialize GPU contexts!" << std::endl;
        return 1;
    }
    
    // Compute matrix inversions using CPU (always)
    std::cout << "Computing matrix inversions using CPU (Eigen)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<float> all_inv_matrices = invert_matrices_cpu(datasets, config.m_states);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Calculate total matrices processed
    int total_matrices = 0;
    for (const auto& ds : datasets) {
        total_matrices += ds.nfrag;
    }
    std::cout << "Processed " << total_matrices << "/" << total_matrices << " matrix inversions" << std::endl;
    std::cout << "CPU matrix inversion completed in " << duration.count() << " ms" << std::endl;
    
    // Calculate total computation pairs
    int total_pairs = 0;
    if (config.use_pair_list) {
        total_pairs = (int)config.pair_list.size();
        std::cout << "Pair-list mode: Requested " << total_pairs << " explicit pairs" << std::endl;
    } else if (config.reconstruct_symmetric) {
        for (int i = config.start_i; i < config.end_i; ++i) {
            int j_start = std::max(i, config.start_j);
            if (j_start < config.end_j) {
                total_pairs += (config.end_j - j_start);
            }
        }
        std::cout << "Symmetric mode: Computing " << total_pairs << " pairs (upper triangle)" << std::endl;
    } else {
        total_pairs = (config.end_i - config.start_i) * (config.end_j - config.start_j);
        std::cout << "Full mode: Computing " << total_pairs << " pairs (full matrix)" << std::endl;
    }
    
    // Populate work queue
    int planned_pair_count = total_pairs;
    int queue_size = gpu_manager.populate_work_queue(config, datasets, max_frags);
    if (queue_size == 0) {
        write_computation_plan_files(config, planned_pair_count, queue_size);
        std::vector<std::pair<int, int>> missing_outputs_when_empty;
        long long missing_when_empty = verify_expected_output_files(config, missing_outputs_when_empty, 1000);
        if (config.resume_computation && missing_when_empty == 0) {
            std::cout << "No new work items were queued because --resume found all expected outputs already present." << std::endl;
            write_run_manifest(config, "completed", planned_pair_count, queue_size, 0, 0, 0);
            if (!write_run_complete_marker(config)) {
                std::cerr << "Warning: failed to write RUN_COMPLETE marker in " << config.outdir << std::endl;
            }
            return 0;
        }
        std::cerr << "No work items were added to the queue. Check pair list/range and dataset availability." << std::endl;
        write_run_manifest(config, "failed_no_work_items", planned_pair_count, queue_size, 0, -1, (int)missing_when_empty);
        return 1;
    }

    // Record the declared computation plan before launching GPU workers.  Downstream
    // filtering uses this manifest instead of guessing expected pairs from the
    // current set of kl_*.npy files in the output directory.
    write_computation_plan_files(config, planned_pair_count, queue_size);
    write_run_manifest(config, "running", planned_pair_count, queue_size, 0, -1, -1);

    // Use the actual queue size for progress reporting, because resume/missing datasets may reduce it.
    total_pairs = queue_size;
    
    // Execute multi-GPU computation
    if (config.use_pair_list) {
        std::cout << "Starting KL divergence computation for pair list with "
                  << queue_size << " queued tasks..." << std::endl;
    } else {
        std::cout << "Starting KL divergence computation for range i=[" 
                  << config.start_i << "," << config.end_i << "), j=[" 
                  << config.start_j << "," << config.end_j << ")..." << std::endl;
    }
    
    auto computation_start = std::chrono::high_resolution_clock::now();
    
    gpu_manager.execute_computation(all_inv_matrices, config, total_pairs);
    
    auto computation_end = std::chrono::high_resolution_clock::now();
    auto computation_duration = std::chrono::duration_cast<std::chrono::seconds>(computation_end - computation_start);
    
    // Print final results
    int pairs_completed = gpu_manager.get_pairs_completed();
    
    std::cout << "\nKL divergence computation completed!" << std::endl;
    std::cout << "Total pairs computed: " << pairs_completed << std::endl;
    std::cout << "Total computation time: " << format_time_duration(computation_duration.count()) << std::endl;
    std::cout << "Results saved as " << (config.save_as_text ? "text" : "NPY") << " files" << std::endl;
    
    if (computation_duration.count() > 0) {
        std::cout << "Average throughput: " << (pairs_completed / (double)computation_duration.count()) 
                  << " pairs/second" << std::endl;
    }

    std::vector<std::pair<int, int>> missing_outputs;
    long long missing_output_count = verify_expected_output_files(config, missing_outputs, 1000);

    if (pairs_completed != queue_size || missing_output_count > 0) {
        std::cerr << "C++ run did not finish cleanly: completed " << pairs_completed
                  << " queued tasks out of " << queue_size
                  << "; missing expected output files: " << missing_output_count << std::endl;
        if (!missing_outputs.empty()) {
            std::cerr << "First missing output pair(s):" << std::endl;
            for (size_t k = 0; k < std::min<size_t>(missing_outputs.size(), 20); ++k) {
                std::cerr << "  (" << missing_outputs[k].first << ", " << missing_outputs[k].second << ")" << std::endl;
            }
        }
        write_run_manifest(config, "failed_incomplete_outputs", planned_pair_count, queue_size,
                           pairs_completed, computation_duration.count(), (int)missing_output_count);
        return 1;
    }

    write_run_manifest(config, "completed", planned_pair_count, queue_size,
                       pairs_completed, computation_duration.count(), 0);
    if (!write_run_complete_marker(config)) {
        std::cerr << "Warning: failed to write RUN_COMPLETE marker in " << config.outdir << std::endl;
    }
    
    return 0;
}
