#include "kl_divergence.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <set>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <limits>
#include <cnpy.h>

// ============================================================================
// BASIC UTILITIES
// ============================================================================

bool create_directory_if_not_exists(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        return mkdir(path.c_str(), 0755) == 0;
    }
    return true;
}

bool file_exists(const std::string& path) {
    struct stat st = {0};
    return stat(path.c_str(), &st) == 0;
}

void save_matrix_npy(const std::string& filename, const float* data, int nfrag1, int nfrag2) {
    cnpy::npy_save(filename, data, {(size_t)nfrag1, (size_t)nfrag2}, "w");
}

void save_matrix_text(const std::string& filename, const float* data, int nfrag1, int nfrag2) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    file << "# KL divergence matrix\n";
    file << "# Shape: " << nfrag1 << " x " << nfrag2 << "\n";
    for (int f1 = 0; f1 < nfrag1; ++f1) {
        for (int f2 = 0; f2 < nfrag2; ++f2) {
            int idx = f1 * nfrag2 + f2;
            file << std::scientific << std::setprecision(6) << data[idx];
            if (f2 < nfrag2 - 1) file << " ";
        }
        file << "\n";
    }
}

std::string format_time_duration(long seconds) {
    if (seconds < 60) {
        return std::to_string(seconds) + " seconds";
    } else if (seconds < 3600) {
        int mins = seconds / 60;
        int secs = seconds % 60;
        return std::to_string(mins) + "m " + std::to_string(secs) + "s";
    } else {
        int hours = seconds / 3600;
        int mins = (seconds % 3600) / 60;
        int secs = seconds % 60;
        return std::to_string(hours) + "h " + std::to_string(mins) + "m " + std::to_string(secs) + "s";
    }
}

void print_memory_usage(size_t bytes) {
    if (bytes < 1024) {
        std::cout << bytes << " B";
    } else if (bytes < 1024 * 1024) {
        std::cout << (bytes / 1024) << " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        std::cout << (bytes / (1024 * 1024)) << " MB";
    } else {
        std::cout << (bytes / (1024 * 1024 * 1024)) << " GB";
    }
}

static std::string trim_string_local(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

static std::string to_lower_local(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::vector<std::string> split_csv_or_ws_local(const std::string& line) {
    std::vector<std::string> tokens;
    if (line.find(',') != std::string::npos) {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) tokens.push_back(trim_string_local(token));
    } else {
        std::stringstream ss(line);
        std::string token;
        while (ss >> token) tokens.push_back(trim_string_local(token));
    }
    return tokens;
}

static bool is_integer_token_local(const std::string& s) {
    if (s.empty()) return false;
    size_t pos = 0;
    if (s[0] == '+' || s[0] == '-') pos = 1;
    if (pos >= s.size()) return false;
    for (; pos < s.size(); ++pos) {
        if (!std::isdigit(static_cast<unsigned char>(s[pos]))) return false;
    }
    return true;
}

static long long parse_ll_local(const std::vector<std::string>& tokens, int col, int line_no, const std::string& name) {
    if (col < 0 || col >= (int)tokens.size()) {
        std::ostringstream oss;
        oss << "Line " << line_no << " missing column " << name;
        throw std::runtime_error(oss.str());
    }
    try {
        return std::stoll(tokens[col]);
    } catch (...) {
        std::ostringstream oss;
        oss << "Line " << line_no << " has invalid integer in column " << name << ": " << tokens[col];
        throw std::runtime_error(oss.str());
    }
}

static int find_col_or_throw(const std::vector<std::string>& header, const std::string& name) {
    for (int i = 0; i < (int)header.size(); ++i) {
        if (to_lower_local(header[i]) == to_lower_local(name)) return i;
    }
    throw std::runtime_error("index.csv missing required column: " + name);
}

static std::string join_path_local(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    if (dir.back() == '/') return dir + file;
    return dir + "/" + file;
}

// ============================================================================
// OLD SMALL-NPY DATA LOADING/SAVING
// ============================================================================

bool check_dataset_files(const std::string& data_dir, int idx) {
    std::string pi_file = data_dir + "/pi_" + std::to_string(idx) + ".npy";
    std::string A_file = data_dir + "/A_" + std::to_string(idx) + ".npy";
    std::string B_file = data_dir + "/B_" + std::to_string(idx) + ".npy";
    std::ifstream pi_check(pi_file);
    std::ifstream A_check(A_file);
    std::ifstream B_check(B_file);
    return pi_check.good() && A_check.good() && B_check.good();
}

FragSet load_dataset(const std::string& data_dir, int idx, int expected_m_states, int expected_n_obs) {
    std::string pi_file = data_dir + "/pi_" + std::to_string(idx) + ".npy";
    std::string A_file = data_dir + "/A_" + std::to_string(idx) + ".npy";
    std::string B_file = data_dir + "/B_" + std::to_string(idx) + ".npy";
    cnpy::NpyArray pi_arr = cnpy::npy_load(pi_file);
    cnpy::NpyArray A_arr = cnpy::npy_load(A_file);
    cnpy::NpyArray B_arr = cnpy::npy_load(B_file);

    if (pi_arr.shape.size() != 2 || A_arr.shape.size() != 3 || B_arr.shape.size() != 3) {
        throw std::runtime_error("Unexpected NPY shape rank for dataset " + std::to_string(idx));
    }
    int nfrag = (int)pi_arr.shape[0];
    int m_states = (int)pi_arr.shape[1];
    int n_obs = (int)B_arr.shape[2];
    if (expected_m_states > 0 && m_states != expected_m_states) {
        throw std::runtime_error("Dataset " + std::to_string(idx) + " m_states mismatch: got " +
                                 std::to_string(m_states) + ", expected " + std::to_string(expected_m_states));
    }
    if (expected_n_obs > 0 && n_obs != expected_n_obs) {
        throw std::runtime_error("Dataset " + std::to_string(idx) + " n_obs mismatch: got " +
                                 std::to_string(n_obs) + ", expected " + std::to_string(expected_n_obs));
    }
    if ((int)A_arr.shape[0] != nfrag || (int)A_arr.shape[1] != m_states || (int)A_arr.shape[2] != m_states ||
        (int)B_arr.shape[0] != nfrag || (int)B_arr.shape[1] != m_states) {
        throw std::runtime_error("Dataset " + std::to_string(idx) + " pi/A/B shapes are inconsistent.");
    }
    if (m_states > MAX_M_STATES || n_obs > MAX_N_OBS) {
        throw std::runtime_error("Dataset " + std::to_string(idx) + " exceeds compiled CUDA capacity.");
    }

    FragSet fs(nfrag, m_states, n_obs);
    float* pi_data = pi_arr.data<float>();
    float* A_data = A_arr.data<float>();
    float* B_data = B_arr.data<float>();
    std::copy(pi_data, pi_data + pi_arr.num_vals, fs.pi.begin());
    std::copy(A_data, A_data + A_arr.num_vals, fs.A.begin());
    std::copy(B_data, B_data + B_arr.num_vals, fs.B.begin());
    return fs;
}

void save_dataset(const FragSet& fs, const std::string& data_dir, int idx) {
    std::string pi_file = data_dir + "/pi_" + std::to_string(idx) + ".npy";
    std::string A_file = data_dir + "/A_" + std::to_string(idx) + ".npy";
    std::string B_file = data_dir + "/B_" + std::to_string(idx) + ".npy";
    cnpy::npy_save(pi_file, fs.pi.data(), {(size_t)fs.nfrag, (size_t)fs.m_states}, "w");
    cnpy::npy_save(A_file, fs.A.data(), {(size_t)fs.nfrag, (size_t)fs.m_states, (size_t)fs.m_states}, "w");
    cnpy::npy_save(B_file, fs.B.data(), {(size_t)fs.nfrag, (size_t)fs.m_states, (size_t)fs.n_obs}, "w");
}

void save_frag_counts(const std::vector<int>& frag_counts, const std::string& data_dir) {
    std::string file = data_dir + "/frag_counts.txt";
    std::ofstream out(file);
    if (!out.is_open()) return;
    for (size_t i = 0; i < frag_counts.size(); ++i) {
        out << i << " " << frag_counts[i] << "\n";
    }
}

std::vector<int> load_frag_counts(const std::string& data_dir) {
    std::string file = data_dir + "/frag_counts.txt";
    std::ifstream in(file);
    std::vector<int> counts;
    int idx, count;
    while (in >> idx >> count) {
        if ((int)counts.size() <= idx) counts.resize(idx + 1, 0);
        counts[idx] = count;
    }
    return counts;
}

// ============================================================================
// PACKED DATABASE LOADING
// ============================================================================


static std::string read_text_file_local(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open text file: " + filename);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static long long extract_json_int_local(const std::string& json, const std::string& key) {
    const std::string quoted_key = "\"" + key + "\"";
    size_t pos = json.find(quoted_key);
    if (pos == std::string::npos) {
        throw std::runtime_error("metadata.json missing required integer field: " + key);
    }

    pos = json.find(':', pos + quoted_key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("metadata.json malformed near field: " + key);
    }
    ++pos;

    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }

    size_t start = pos;
    if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) {
        ++pos;
    }
    bool has_digit = false;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
        has_digit = true;
        ++pos;
    }
    if (!has_digit) {
        throw std::runtime_error("metadata.json field is not an integer: " + key);
    }

    return std::stoll(json.substr(start, pos - start));
}

PackedRuntimeInfo load_packed_runtime_info_from_metadata(const std::string& packed_db_dir) {
    const std::string metadata_file = join_path_local(packed_db_dir, "metadata.json");
    const std::string json = read_text_file_local(metadata_file);

    PackedRuntimeInfo info;
    info.fragment = (int)extract_json_int_local(json, "fragment");
    info.m_states = (int)extract_json_int_local(json, "m_states");
    info.n_obs = (int)extract_json_int_local(json, "n_obs");
    info.num_datasets = (int)extract_json_int_local(json, "num_datasets");
    info.min_frag = (int)extract_json_int_local(json, "min_frag");
    info.max_frag = (int)extract_json_int_local(json, "max_frag");

    if (info.fragment <= 0) {
        throw std::runtime_error("metadata.json has invalid fragment <= 0.");
    }
    const int expected_m_states = 3 * info.fragment + 1;
    if (info.m_states != expected_m_states) {
        std::ostringstream oss;
        oss << "metadata.json m_states mismatch: got " << info.m_states
            << ", expected 3*fragment+1=" << expected_m_states;
        throw std::runtime_error(oss.str());
    }
    if (info.m_states <= 0 || info.m_states > MAX_M_STATES) {
        std::ostringstream oss;
        oss << "metadata.json m_states=" << info.m_states
            << " exceeds compiled MAX_M_STATES=" << MAX_M_STATES
            << ". Increase MAX_M_STATES in kl_divergence.h and recompile.";
        throw std::runtime_error(oss.str());
    }
    if (info.n_obs <= 0 || info.n_obs > MAX_N_OBS) {
        std::ostringstream oss;
        oss << "metadata.json n_obs=" << info.n_obs
            << " exceeds compiled MAX_N_OBS=" << MAX_N_OBS
            << ". Increase MAX_N_OBS in kl_divergence.h and recompile.";
        throw std::runtime_error(oss.str());
    }
    if (info.num_datasets <= 0) {
        throw std::runtime_error("metadata.json has invalid num_datasets <= 0.");
    }
    if (info.min_frag <= 0 || info.max_frag < info.min_frag) {
        throw std::runtime_error("metadata.json has invalid min_frag/max_frag.");
    }

    std::cout << "Loaded packed database metadata: " << metadata_file << std::endl;
    std::cout << "  fragment=" << info.fragment
              << ", m_states=" << info.m_states
              << ", n_obs=" << info.n_obs
              << ", num_datasets=" << info.num_datasets
              << ", min_frag=" << info.min_frag
              << ", max_frag=" << info.max_frag << std::endl;
    return info;
}

PackedDatasetIndex load_packed_dataset_index(const std::string& packed_db_dir) {
    std::string index_file = join_path_local(packed_db_dir, "index.csv");
    std::ifstream file(index_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open packed database index: " + index_file);
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("Packed database index is empty: " + index_file);
    }
    if (!header_line.empty() && (unsigned char)header_line[0] == 0xEF) {
        // Strip UTF-8 BOM when present.
        if (header_line.size() >= 3) header_line = header_line.substr(3);
    }
    std::vector<std::string> header = split_csv_or_ws_local(trim_string_local(header_line));
    int c_family_id = find_col_or_throw(header, "family_id");
    int c_hmm_length = find_col_or_throw(header, "hmm_length");
    int c_fragment = find_col_or_throw(header, "fragment");
    int c_nfrag = find_col_or_throw(header, "nfrag");
    int c_pi_offset = find_col_or_throw(header, "pi_offset");
    int c_pi_count = find_col_or_throw(header, "pi_count");
    int c_A_offset = find_col_or_throw(header, "A_offset");
    int c_A_count = find_col_or_throw(header, "A_count");
    int c_B_offset = find_col_or_throw(header, "B_offset");
    int c_B_count = find_col_or_throw(header, "B_count");

    PackedDatasetIndex index;
    int duplicate_count = 0;
    int line_no = 1;
    std::string line;
    while (std::getline(file, line)) {
        line_no++;
        line = trim_string_local(line);
        if (line.empty()) continue;
        std::vector<std::string> tokens = split_csv_or_ws_local(line);
        if (tokens.empty()) continue;

        PackedDatasetIndexEntry e;
        e.family_id = (int)parse_ll_local(tokens, c_family_id, line_no, "family_id");
        e.hmm_length = (int)parse_ll_local(tokens, c_hmm_length, line_no, "hmm_length");
        e.fragment = (int)parse_ll_local(tokens, c_fragment, line_no, "fragment");
        e.nfrag = (int)parse_ll_local(tokens, c_nfrag, line_no, "nfrag");
        e.pi_offset = (size_t)parse_ll_local(tokens, c_pi_offset, line_no, "pi_offset");
        e.pi_count = (size_t)parse_ll_local(tokens, c_pi_count, line_no, "pi_count");
        e.A_offset = (size_t)parse_ll_local(tokens, c_A_offset, line_no, "A_offset");
        e.A_count = (size_t)parse_ll_local(tokens, c_A_count, line_no, "A_count");
        e.B_offset = (size_t)parse_ll_local(tokens, c_B_offset, line_no, "B_offset");
        e.B_count = (size_t)parse_ll_local(tokens, c_B_count, line_no, "B_count");

        if (e.family_id < 0 || e.fragment <= 0 || e.nfrag <= 0) {
            std::ostringstream oss;
            oss << "Packed index line " << line_no << " has invalid family_id/fragment/nfrag.";
            throw std::runtime_error(oss.str());
        }
        const int m_states = 3 * e.fragment + 1;
        if (m_states <= 0 || m_states > MAX_M_STATES) {
            std::ostringstream oss;
            oss << "Packed index line " << line_no << " has m_states=" << m_states
                << " exceeding MAX_M_STATES=" << MAX_M_STATES;
            throw std::runtime_error(oss.str());
        }
        const size_t expected_pi = (size_t)e.nfrag * m_states;
        const size_t expected_A = (size_t)e.nfrag * m_states * m_states;
        if (e.pi_count != expected_pi || e.A_count != expected_A) {
            std::ostringstream oss;
            oss << "Packed index line " << line_no << " pi/A count mismatch for family " << e.family_id;
            throw std::runtime_error(oss.str());
        }
        const size_t denom = (size_t)e.nfrag * m_states;
        if (denom == 0 || e.B_count % denom != 0) {
            std::ostringstream oss;
            oss << "Packed index line " << line_no << " cannot infer integer n_obs for family " << e.family_id;
            throw std::runtime_error(oss.str());
        }
        int n_obs = (int)(e.B_count / denom);
        if (n_obs <= 0 || n_obs > MAX_N_OBS) {
            std::ostringstream oss;
            oss << "Packed index line " << line_no << " has n_obs=" << n_obs
                << " exceeding MAX_N_OBS=" << MAX_N_OBS;
            throw std::runtime_error(oss.str());
        }

        if (!index.emplace(e.family_id, e).second) {
            duplicate_count++;
            index[e.family_id] = e;
        }
    }

    std::cout << "Loaded packed database index: " << index.size() << " families from " << index_file;
    if (duplicate_count > 0) std::cout << " (overwrote " << duplicate_count << " duplicate family_id rows)";
    std::cout << std::endl;
    return index;
}

PackedRuntimeInfo infer_packed_runtime_info(const PackedDatasetIndex& index) {
    if (index.empty()) throw std::runtime_error("Cannot infer runtime info from empty packed index.");

    PackedRuntimeInfo info;
    bool first = true;
    int max_family_id = -1;
    int min_frag = std::numeric_limits<int>::max();
    int max_frag = 0;

    for (const auto& kv : index) {
        const auto& e = kv.second;
        int m_states = 3 * e.fragment + 1;
        int n_obs = (int)(e.B_count / ((size_t)e.nfrag * m_states));
        if (first) {
            info.fragment = e.fragment;
            info.m_states = m_states;
            info.n_obs = n_obs;
            first = false;
        } else {
            if (e.fragment != info.fragment || m_states != info.m_states || n_obs != info.n_obs) {
                throw std::runtime_error("Packed database mixes multiple fragment/m_states/n_obs configurations. Use one database per fragment length.");
            }
        }
        max_family_id = std::max(max_family_id, e.family_id);
        min_frag = std::min(min_frag, e.nfrag);
        max_frag = std::max(max_frag, e.nfrag);
    }

    info.num_datasets = max_family_id + 1;
    info.min_frag = (min_frag == std::numeric_limits<int>::max()) ? 0 : min_frag;
    info.max_frag = max_frag;
    return info;
}

void apply_packed_runtime_info(Config& config, const PackedRuntimeInfo& info) {
    config.fragment = info.fragment;
    config.m_states = info.m_states;
    config.n_obs = info.n_obs;
    config.num_datasets = info.num_datasets;
    config.min_frag = info.min_frag;
    config.max_frag = info.max_frag;

    if (config.m_states <= 0 || config.m_states > MAX_M_STATES) {
        throw std::runtime_error("Runtime m_states from packed DB exceeds compiled MAX_M_STATES.");
    }
    if (config.n_obs <= 0 || config.n_obs > MAX_N_OBS) {
        throw std::runtime_error("Runtime n_obs from packed DB exceeds compiled MAX_N_OBS.");
    }
}

bool check_packed_dataset(const PackedDatasetIndex& index, int dataset_idx) {
    return index.find(dataset_idx) != index.end();
}

static std::vector<float> read_float32_block_local(const std::string& filename, size_t float_offset, size_t float_count) {
    std::vector<float> data(float_count);
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open packed binary file: " + filename);
    const std::streamoff byte_offset = (std::streamoff)(float_offset * sizeof(float));
    const std::streamsize byte_count = (std::streamsize)(float_count * sizeof(float));
    file.seekg(byte_offset, std::ios::beg);
    if (!file.good()) throw std::runtime_error("Failed to seek packed binary file: " + filename);
    file.read(reinterpret_cast<char*>(data.data()), byte_count);
    if (file.gcount() != byte_count) {
        std::ostringstream oss;
        oss << "Short read from " << filename << ": requested " << byte_count << " bytes, got " << file.gcount() << " bytes.";
        throw std::runtime_error(oss.str());
    }
    return data;
}

FragSet load_packed_dataset(const std::string& packed_db_dir, const PackedDatasetIndex& index, int dataset_idx, int expected_m_states, int expected_n_obs) {
    auto it = index.find(dataset_idx);
    if (it == index.end()) throw std::runtime_error("Family index not found in packed database: " + std::to_string(dataset_idx));
    const PackedDatasetIndexEntry& e = it->second;
    const int m_states = 3 * e.fragment + 1;
    const int n_obs = (int)(e.B_count / ((size_t)e.nfrag * m_states));
    if (expected_m_states > 0 && m_states != expected_m_states) {
        throw std::runtime_error("Packed dataset " + std::to_string(dataset_idx) + " m_states mismatch.");
    }
    if (expected_n_obs > 0 && n_obs != expected_n_obs) {
        throw std::runtime_error("Packed dataset " + std::to_string(dataset_idx) + " n_obs mismatch.");
    }
    FragSet fs(e.nfrag, m_states, n_obs);
    fs.pi = read_float32_block_local(join_path_local(packed_db_dir, "pi_all.float32.bin"), e.pi_offset, e.pi_count);
    fs.A  = read_float32_block_local(join_path_local(packed_db_dir, "A_all.float32.bin"),  e.A_offset,  e.A_count);
    fs.B  = read_float32_block_local(join_path_local(packed_db_dir, "B_all.float32.bin"),  e.B_offset,  e.B_count);
    return fs;
}

// ============================================================================
// PAIR-LIST LOADING
// ============================================================================

std::vector<std::pair<int, int>> load_pair_list(const std::string& pair_list_file) {
    std::ifstream file(pair_list_file);
    if (!file.is_open()) throw std::runtime_error("Failed to open pair list file: " + pair_list_file);

    std::vector<std::pair<int, int>> pairs;
    std::set<std::pair<int, int>> seen_pairs;
    bool header_processed = false;
    int i_col = 0;
    int j_col = 1;
    int line_no = 0;
    int duplicate_count = 0;
    std::string line;

    while (std::getline(file, line)) {
        line_no++;
        line = trim_string_local(line);
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> tokens = split_csv_or_ws_local(line);
        if (tokens.empty()) continue;

        if (!header_processed) {
            header_processed = true;
            bool first_two_integer = tokens.size() >= 2 && is_integer_token_local(tokens[0]) && is_integer_token_local(tokens[1]);
            if (!first_two_integer) {
                i_col = -1;
                j_col = -1;
                for (int c = 0; c < (int)tokens.size(); ++c) {
                    std::string name = to_lower_local(tokens[c]);
                    if (name == "i" || name == "family_i" || name == "idx_i" || name == "start_i") {
                        if (i_col < 0 || name == "i") i_col = c;
                    }
                    if (name == "j" || name == "family_j" || name == "idx_j" || name == "start_j") {
                        if (j_col < 0 || name == "j") j_col = c;
                    }
                }
                if (i_col < 0 || j_col < 0) {
                    throw std::runtime_error("Pair list header must contain columns named i and j, or use a two-column no-header file.");
                }
                continue;
            }
        }

        if ((int)tokens.size() <= std::max(i_col, j_col)) {
            std::ostringstream oss;
            oss << "Malformed pair-list line " << line_no << ": not enough columns.";
            throw std::runtime_error(oss.str());
        }
        if (!is_integer_token_local(tokens[i_col]) || !is_integer_token_local(tokens[j_col])) {
            std::ostringstream oss;
            oss << "Malformed pair-list line " << line_no << ": i/j are not integers.";
            throw std::runtime_error(oss.str());
        }
        int i = std::stoi(tokens[i_col]);
        int j = std::stoi(tokens[j_col]);
        if (i < 0 || j < 0) {
            std::ostringstream oss;
            oss << "Invalid pair-list line " << line_no << ": indices must be non-negative. Got (" << i << ", " << j << ").";
            throw std::runtime_error(oss.str());
        }
        std::pair<int, int> p(i, j);
        if (seen_pairs.insert(p).second) pairs.push_back(p);
        else duplicate_count++;
    }

    if (pairs.empty()) throw std::runtime_error("Pair list contains no valid pairs: " + pair_list_file);
    std::cout << "Loaded pair list: " << pairs.size() << " unique pairs";
    if (duplicate_count > 0) std::cout << " (skipped " << duplicate_count << " duplicate rows)";
    std::cout << std::endl;
    return pairs;
}

void validate_pair_list_against_num_datasets(const std::vector<std::pair<int, int>>& pairs, int num_datasets) {
    for (size_t row = 0; row < pairs.size(); ++row) {
        int i = pairs[row].first;
        int j = pairs[row].second;
        if (i < 0 || j < 0 || i >= num_datasets || j >= num_datasets) {
            std::ostringstream oss;
            oss << "Pair-list row " << (row + 1) << " has indices outside [0, " << num_datasets
                << "): (" << i << ", " << j << ").";
            throw std::runtime_error(oss.str());
        }
    }
}

// ============================================================================
// RUN PLAN AND COMPLETION MANIFEST UTILITIES
// ============================================================================

static std::string json_escape_manifest(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

static std::string result_extension_manifest(const Config& config) {
    return config.save_as_text ? ".txt" : ".npy";
}

static std::string result_filename_manifest(const Config& config, int i, int j) {
    std::ostringstream fname;
    fname << config.outdir << "kl_"
          << std::setfill('0') << std::setw(4) << i
          << "_" << std::setfill('0') << std::setw(4) << j
          << result_extension_manifest(config);
    return fname.str();
}

static long long count_planned_pairs_manifest(const Config& config) {
    if (config.use_pair_list) {
        return static_cast<long long>(config.pair_list.size());
    }
    long long total = 0;
    if (config.reconstruct_symmetric) {
        for (int i = config.start_i; i < config.end_i; ++i) {
            int j_start = std::max(i, config.start_j);
            if (j_start < config.end_j) {
                total += static_cast<long long>(config.end_j - j_start);
            }
        }
    } else {
        total = static_cast<long long>(config.end_i - config.start_i) *
                static_cast<long long>(config.end_j - config.start_j);
    }
    return total;
}

void write_computation_plan_files(const Config& config, int planned_pair_count, int queued_pair_count) {
    // The downstream Python filtering stage should not infer the C++ computation plan
    // by scanning whatever files happen to exist in the output directory.  These files
    // record the family-pair plan declared by the C++ run.
    if (config.use_pair_list) {
        std::string pair_path = config.outdir + "prodive_computed_pairs.csv";
        std::ofstream pair_file(pair_path);
        if (!pair_file.is_open()) {
            std::cerr << "Warning: failed to write pair manifest: " << pair_path << std::endl;
        } else {
            pair_file << "i,j\n";
            for (const auto& p : config.pair_list) {
                pair_file << p.first << "," << p.second << "\n";
            }
        }
    } else {
        std::string range_path = config.outdir + "prodive_computed_ranges.csv";
        std::ofstream range_file(range_path);
        if (!range_file.is_open()) {
            std::cerr << "Warning: failed to write range manifest: " << range_path << std::endl;
        } else {
            range_file << "mode,start_i,end_i,start_j,end_j,range_semantics,reconstruct_symmetric,planned_pair_count,queued_pair_count\n";
            range_file << "range," << config.start_i << "," << config.end_i << ","
                       << config.start_j << "," << config.end_j << ",half_open,"
                       << (config.reconstruct_symmetric ? "true" : "false") << ","
                       << planned_pair_count << "," << queued_pair_count << "\n";
        }
    }
}

void write_run_manifest(const Config& config, const std::string& status,
                        int planned_pair_count, int queued_pair_count,
                        int pairs_completed, long computation_seconds,
                        int missing_output_count) {
    std::string path = config.outdir + "prodive_run_manifest.json";
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: failed to write run manifest: " << path << std::endl;
        return;
    }

    f << "{\n";
    f << "  \"status\": \"" << json_escape_manifest(status) << "\",\n";
    f << "  \"mode\": \"" << (config.use_pair_list ? "pair_list" : "range") << "\",\n";
    f << "  \"output_dir\": \"" << json_escape_manifest(config.outdir) << "\",\n";
    f << "  \"data_source\": \"" << (config.use_packed_db ? "packed_db" : "small_npy") << "\",\n";
    f << "  \"packed_db\": \"" << json_escape_manifest(config.packed_db_dir) << "\",\n";
    f << "  \"datadir\": \"" << json_escape_manifest(config.datadir) << "\",\n";
    f << "  \"pair_list_file\": \"" << json_escape_manifest(config.pair_list_file) << "\",\n";
    f << "  \"start_i\": " << config.start_i << ",\n";
    f << "  \"end_i\": " << config.end_i << ",\n";
    f << "  \"start_j\": " << config.start_j << ",\n";
    f << "  \"end_j\": " << config.end_j << ",\n";
    f << "  \"range_semantics\": \"half_open_[start,end)\",\n";
    f << "  \"reconstruct_symmetric\": " << (config.reconstruct_symmetric ? "true" : "false") << ",\n";
    f << "  \"planned_pair_count\": " << planned_pair_count << ",\n";
    f << "  \"queued_pair_count\": " << queued_pair_count << ",\n";
    f << "  \"pairs_completed\": " << pairs_completed << ",\n";
    f << "  \"missing_output_count\": " << missing_output_count << ",\n";
    f << "  \"completion_marker\": \"RUN_COMPLETE\",\n";
    f << "  \"output_pattern\": \"kl_<i:04d>_<j:04d>" << result_extension_manifest(config) << "\",\n";
    f << "  \"fragment\": " << config.fragment << ",\n";
    f << "  \"m_states\": " << config.m_states << ",\n";
    f << "  \"n_obs\": " << config.n_obs << ",\n";
    f << "  \"num_datasets\": " << config.num_datasets << ",\n";
    f << "  \"dtype\": \"float32\",\n";
    f << "  \"save_format\": \"" << (config.save_as_text ? "text" : "npy") << "\",\n";
    f << "  \"resume\": " << (config.resume_computation ? "true" : "false") << ",\n";
    f << "  \"memory_efficient\": " << (config.memory_efficient ? "true" : "false") << ",\n";
    f << "  \"computation_seconds\": " << computation_seconds << ",\n";
    f << "  \"computed_pairs_file\": \"" << (config.use_pair_list ? "prodive_computed_pairs.csv" : "") << "\",\n";
    f << "  \"computed_ranges_file\": \"" << (config.use_pair_list ? "" : "prodive_computed_ranges.csv") << "\"\n";
    f << "}\n";
}

long long verify_expected_output_files(const Config& config, std::vector<std::pair<int, int>>& missing_outputs,
                                       int max_missing_to_store) {
    missing_outputs.clear();
    long long missing_count = 0;

    auto check_pair = [&](int i, int j) {
        if (!file_exists(result_filename_manifest(config, i, j))) {
            missing_count++;
            if ((int)missing_outputs.size() < max_missing_to_store) {
                missing_outputs.emplace_back(i, j);
            }
        }
    };

    if (config.use_pair_list) {
        for (const auto& p : config.pair_list) {
            check_pair(p.first, p.second);
        }
    } else if (config.reconstruct_symmetric) {
        for (int i = config.start_i; i < config.end_i; ++i) {
            int j_start = std::max(i, config.start_j);
            for (int j = j_start; j < config.end_j; ++j) {
                check_pair(i, j);
            }
        }
    } else {
        for (int i = config.start_i; i < config.end_i; ++i) {
            for (int j = config.start_j; j < config.end_j; ++j) {
                check_pair(i, j);
            }
        }
    }

    if (missing_count > 0) {
        std::string missing_path = config.outdir + "prodive_missing_outputs.csv";
        std::ofstream mf(missing_path);
        if (mf.is_open()) {
            mf << "i,j,expected_file\n";
            for (const auto& p : missing_outputs) {
                mf << p.first << "," << p.second << "," << result_filename_manifest(config, p.first, p.second) << "\n";
            }
            if (missing_count > (long long)missing_outputs.size()) {
                mf << "# omitted_additional_missing," << (missing_count - (long long)missing_outputs.size()) << ",\n";
            }
        }
    }

    return missing_count;
}

bool write_run_complete_marker(const Config& config) {
    std::string path = config.outdir + "RUN_COMPLETE";
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "completed\n";
    return true;
}
