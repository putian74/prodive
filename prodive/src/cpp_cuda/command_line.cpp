// parsing command line options

#include "kl_divergence.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "\nOptions:\n"
              << "  --start_i N          Start index for first dataset (default: 0)\n"
              << "  --end_i N            End index for first dataset (default: runtime num_datasets)\n"
              << "  --start_j N          Start index for second dataset (default: 0)\n"
              << "  --end_j N            End index for second dataset (default: runtime num_datasets)\n"
              << "  --pair_list PATH     Compute only pairs listed in CSV/TSV/plain text file.\n"
              << "                       Accepted formats: i,j; or wider CSV with columns named i and j.\n"
              << "  --outdir PATH        Output directory (default: ./kl_output/)\n"
              << "  --datadir PATH       Small-NPY data directory (old mode).\n"
              << "  --packed_db PATH     Packed DB directory containing index.csv and *_all.float32.bin files.\n"
              << "                       If provided, runtime fragment/m_states/n_obs/num_datasets are read from metadata.json.\n"
              << "  --fragment N         Fragment length for small-NPY mode; m_states becomes 3*N+1.\n"
              << "  --m_states N         Manually set HMM state count for small-NPY mode.\n"
              << "  --n_obs N            Observation alphabet size for small-NPY mode (default: 21).\n"
              << "  --num_datasets N     Dataset count for range validation in small-NPY mode.\n"
              << "  --num_streams N      CUDA streams per GPU (default: 8).\n"
              << "  --max_gpus N         Maximum GPUs to use (default: 4).\n"
              << "  --memory_efficient   Use memory-efficient mode (load only needed datasets).\n"
              << "                       Pair-list mode always loads only sampled family IDs.\n"
              << "  --resume             Resume computation (skip existing files)\n"
              << "  --no_symmetric       Disable symmetric reconstruction for range mode.\n"
              << "                       Pair-list mode always computes the listed pairs exactly.\n"
              << "  --save_text          Save results as text files (default: NPY format)\n"
              << "  --help, -h           Show this help message\n"
              << "\nExamples:\n"
              << "  " << program_name << " --start_i 0 --end_i 100 --packed_db /path/to/fragment_6_packed\n"
              << "  " << program_name << " --pair_list sampled_pairs.csv --packed_db /path/to/fragment_6_packed --memory_efficient\n"
              << "  " << program_name << " --pair_list sampled_pairs.csv --packed_db /path/to/fragment_7_packed --max_gpus 4 --num_streams 8\n"
              << "\nNotes:\n"
              << "  Packed-db mode is recommended. It reads runtime constants from metadata.json.\n"
              << "  Runtime m_states must be <= MAX_M_STATES=" << MAX_M_STATES << "; n_obs must be <= MAX_N_OBS=" << MAX_N_OBS << ".\n"
              << "  In small-NPY mode, provide --fragment or --m_states when not using fragment=6.\n";
}

static int parse_positive_int_or_exit(const std::string& arg, const char* name) {
    int v = std::atoi(arg.c_str());
    if (v <= 0) {
        std::cerr << "Error: " << name << " must be positive. Got: " << arg << std::endl;
        exit(1);
    }
    return v;
}

Config parse_command_line(int argc, char* argv[]) {
    Config config;
    bool m_states_explicit = false;
    bool fragment_explicit = false;
    bool end_i_explicit = false;
    bool end_j_explicit = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        }
        else if (arg == "--start_i" && i + 1 < argc) {
            config.start_i = std::atoi(argv[++i]);
        }
        else if (arg == "--end_i" && i + 1 < argc) {
            config.end_i = std::atoi(argv[++i]);
            end_i_explicit = true;
        }
        else if (arg == "--start_j" && i + 1 < argc) {
            config.start_j = std::atoi(argv[++i]);
        }
        else if (arg == "--end_j" && i + 1 < argc) {
            config.end_j = std::atoi(argv[++i]);
            end_j_explicit = true;
        }
        else if (arg == "--pair_list" && i + 1 < argc) {
            config.pair_list_file = argv[++i];
            config.use_pair_list = true;
        }
        else if (arg == "--outdir" && i + 1 < argc) {
            config.outdir = argv[++i];
        }
        else if (arg == "--datadir" && i + 1 < argc) {
            config.datadir = argv[++i];
        }
        else if (arg == "--packed_db" && i + 1 < argc) {
            config.packed_db_dir = argv[++i];
            config.use_packed_db = true;
        }
        else if (arg == "--fragment" && i + 1 < argc) {
            config.fragment = parse_positive_int_or_exit(argv[++i], "--fragment");
            config.m_states = 3 * config.fragment + 1;
            fragment_explicit = true;
        }
        else if (arg == "--m_states" && i + 1 < argc) {
            config.m_states = parse_positive_int_or_exit(argv[++i], "--m_states");
            m_states_explicit = true;
        }
        else if (arg == "--n_obs" && i + 1 < argc) {
            config.n_obs = parse_positive_int_or_exit(argv[++i], "--n_obs");
        }
        else if (arg == "--num_datasets" && i + 1 < argc) {
            config.num_datasets = parse_positive_int_or_exit(argv[++i], "--num_datasets");
        }
        else if (arg == "--num_streams" && i + 1 < argc) {
            config.num_streams = parse_positive_int_or_exit(argv[++i], "--num_streams");
        }
        else if (arg == "--max_gpus" && i + 1 < argc) {
            config.max_gpus = parse_positive_int_or_exit(argv[++i], "--max_gpus");
        }
        else if (arg == "--memory_efficient") {
            config.memory_efficient = true;
        }
        else if (arg == "--resume") {
            config.resume_computation = true;
        }
        else if (arg == "--no_symmetric") {
            config.reconstruct_symmetric = false;
        }
        else if (arg == "--save_text") {
            config.save_as_text = true;
        }
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            std::cerr << "Use --help for usage information." << std::endl;
            exit(1);
        }
    }

    if (!m_states_explicit && fragment_explicit) {
        config.m_states = 3 * config.fragment + 1;
    }

    if (config.m_states <= 0 || config.m_states > MAX_M_STATES) {
        std::cerr << "Error: runtime m_states=" << config.m_states
                  << " exceeds supported range 1.." << MAX_M_STATES << std::endl;
        exit(1);
    }
    if (config.n_obs <= 0 || config.n_obs > MAX_N_OBS) {
        std::cerr << "Error: runtime n_obs=" << config.n_obs
                  << " exceeds supported range 1.." << MAX_N_OBS << std::endl;
        exit(1);
    }
    if (config.num_streams <= 0) {
        std::cerr << "Error: --num_streams must be positive." << std::endl;
        exit(1);
    }
    if (config.max_gpus <= 0) {
        std::cerr << "Error: --max_gpus must be positive." << std::endl;
        exit(1);
    }

    if (config.use_packed_db) {
        if (config.packed_db_dir.empty()) {
            std::cerr << "Error: --packed_db requires a directory path." << std::endl;
            exit(1);
        }
        std::string base_dir = config.packed_db_dir;
        if (!base_dir.empty() && base_dir.back() != '/') {
            base_dir += "/";
        }
        std::string index_file = base_dir + "index.csv";
        std::string metadata_file = base_dir + "metadata.json";
        std::ifstream test_index(index_file);
        if (!test_index.good()) {
            std::cerr << "Error: packed database index cannot be opened: "
                      << index_file << std::endl;
            exit(1);
        }
        std::ifstream test_metadata(metadata_file);
        if (!test_metadata.good()) {
            std::cerr << "Error: packed database metadata cannot be opened: "
                      << metadata_file << std::endl;
            exit(1);
        }
    }

    if (config.use_pair_list) {
        if (config.pair_list_file.empty()) {
            std::cerr << "Error: --pair_list requires a file path." << std::endl;
            exit(1);
        }
        std::ifstream test(config.pair_list_file);
        if (!test.good()) {
            std::cerr << "Error: pair list file cannot be opened: "
                      << config.pair_list_file << std::endl;
            exit(1);
        }
    }

    // Range validation is completed in main() after packed DB metadata has been applied.
    if (end_i_explicit && config.end_i <= config.start_i) {
        std::cerr << "Error: start_i must be < end_i." << std::endl;
        exit(1);
    }
    if (end_j_explicit && config.end_j <= config.start_j) {
        std::cerr << "Error: start_j must be < end_j." << std::endl;
        exit(1);
    }
    if (config.start_i < 0 || config.start_j < 0) {
        std::cerr << "Error: start_i/start_j must be non-negative." << std::endl;
        exit(1);
    }

    return config;
}
