#ifndef KL_DIVERGENCE_H
#define KL_DIVERGENCE_H

#include <vector>
#include <string>
#include <map>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cuda_runtime.h>

// These are compile-time safety capacities for CUDA local arrays.
// Runtime m_states and n_obs are read from packed database metadata.json.
// For fragment lengths whose 3*fragment+1 exceeds MAX_M_STATES,
// increase MAX_M_STATES and recompile.
constexpr int MAX_M_STATES = 64;
constexpr int MAX_N_OBS = 32;
constexpr float EPS = 1e-8f;

// Structure definitions
struct FragSet {
    int nfrag;
    int m_states;
    int n_obs;
    std::vector<float> pi, A, B;

    FragSet() : nfrag(0), m_states(0), n_obs(0) {}

    FragSet(int n, int m, int obs)
        : nfrag(n), m_states(m), n_obs(obs),
          pi((size_t)n * m),
          A((size_t)n * m * m),
          B((size_t)n * m * obs) {}
};

// One row from packed fragment database index.csv.
// Offsets and counts are measured in float32 elements, not bytes.
struct PackedDatasetIndexEntry {
    int family_id = -1;
    int hmm_length = 0;
    int fragment = 0;
    int nfrag = 0;

    size_t pi_offset = 0;
    size_t pi_count = 0;
    size_t A_offset = 0;
    size_t A_count = 0;
    size_t B_offset = 0;
    size_t B_count = 0;
};

using PackedDatasetIndex = std::map<int, PackedDatasetIndexEntry>;

struct PackedRuntimeInfo {
    int fragment = 6;
    int m_states = 19;
    int n_obs = 21;
    int num_datasets = 0;
    int min_frag = 0;
    int max_frag = 0;
};

struct Config {
    bool memory_efficient = false;
    bool resume_computation = false;
    bool reconstruct_symmetric = true;
    bool save_as_text = false;  // Option to save as text for debugging
    
    std::string outdir = "./kl_output/";
    std::string datadir = "./small_npy_input";

    // Packed database mode reads these files from packed_db_dir:
    //   pi_all.float32.bin
    //   A_all.float32.bin
    //   B_all.float32.bin
    //   index.csv
    bool use_packed_db = false;
    std::string packed_db_dir = "";
    
    int start_i = 0;
    int end_i = -1;   // -1 means "use runtime num_datasets" after DB metadata is known.
    int start_j = 0;
    int end_j = -1;

    // Runtime dimensions. In packed-db mode these are read from metadata.json.
    // In small-NPY mode they can be set by --fragment/--m_states/--n_obs.
    int fragment = 6;
    int m_states = 19;
    int n_obs = 21;
    int num_datasets = 25545;
    int min_frag = 0;
    int max_frag = 0;

    // Runtime GPU controls.
    int num_streams = 8;
    int max_gpus = 4;

    // Pair-list mode: compute only explicitly listed family pairs.
    bool use_pair_list = false;
    std::string pair_list_file = "";
    std::vector<std::pair<int, int>> pair_list;
    
    // Memory-efficient mode mappings
    mutable std::map<int, int> dataset_index_map;    // original_idx -> loaded_idx
    std::vector<int> reverse_dataset_map;    // loaded_idx -> original_idx

    // Compression options
    enum class OutputFormat {
        FLOAT_ONLY,
        INT8_ONLY,
        BOTH
    };

    OutputFormat output_format = OutputFormat::FLOAT_ONLY;
    bool enable_adaptive_compression = false;
    float compression_coverage = 0.98f;
    std::string compression_params_file = "";
};

// CUDA kernel wrapper function
extern "C" void sym_kl_kernel_with_precomputed_inv(
    const float* pi1, const float* A1, const float* B1,
    const float* pi2, const float* A2, const float* B2,
    const float* inv1, const float* inv2,
    float* KL, int nfrag1, int nfrag2, int m, int n,
    cudaStream_t stream
);

// Matrix inversion functions
std::vector<float> invert_matrices_cpu(const std::vector<FragSet>& datasets, int m_states);

// Data loading/saving functions (from util.cpp)
bool check_dataset_files(const std::string& datadir, int dataset_idx);
FragSet load_dataset(const std::string& datadir, int dataset_idx, int expected_m_states, int expected_n_obs);

PackedDatasetIndex load_packed_dataset_index(const std::string& packed_db_dir);
PackedRuntimeInfo load_packed_runtime_info_from_metadata(const std::string& packed_db_dir);
PackedRuntimeInfo infer_packed_runtime_info(const PackedDatasetIndex& index); // fallback/debug helper; metadata.json is authoritative in main().
void apply_packed_runtime_info(Config& config, const PackedRuntimeInfo& info);
bool check_packed_dataset(const PackedDatasetIndex& index, int dataset_idx);
FragSet load_packed_dataset(const std::string& packed_db_dir, const PackedDatasetIndex& index, int dataset_idx, int expected_m_states, int expected_n_obs);

void save_dataset(const FragSet& fs, const std::string& data_dir, int idx);
void save_frag_counts(const std::vector<int>& frag_counts, const std::string& data_dir);
std::vector<int> load_frag_counts(const std::string& data_dir);
std::vector<std::pair<int, int>> load_pair_list(const std::string& pair_list_file);
void validate_pair_list_against_num_datasets(const std::vector<std::pair<int, int>>& pairs, int num_datasets);


// C++ run-plan / completion manifest helpers (from util.cpp)
void write_computation_plan_files(const Config& config, int planned_pair_count, int queued_pair_count);
void write_run_manifest(const Config& config, const std::string& status,
                        int planned_pair_count, int queued_pair_count,
                        int pairs_completed, long computation_seconds,
                        int missing_output_count);
long long verify_expected_output_files(const Config& config, std::vector<std::pair<int, int>>& missing_outputs,
                                       int max_missing_to_store = 1000);
bool write_run_complete_marker(const Config& config);

// Command line parsing (from command_line.cpp)
Config parse_command_line(int argc, char* argv[]);
void print_usage(const char* program_name);

// Utility functions (from util.cpp)
bool create_directory_if_not_exists(const std::string& path);
bool file_exists(const std::string& path);
void save_matrix_npy(const std::string& filename, const float* data, int nfrag1, int nfrag2);
void save_matrix_text(const std::string& filename, const float* data, int nfrag1, int nfrag2);
std::string format_time_duration(long seconds);
void print_memory_usage(size_t bytes);

#endif // KL_DIVERGENCE_H
