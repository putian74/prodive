#ifndef KL_COMPRESSION_H
#define KL_COMPRESSION_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

class AdaptiveKLCompressor {
private:
    // Fixed compression parameters
    static constexpr float SCALE_FACTOR = 8.0f; // Due to PHMM model length normalization
    static constexpr float DEFAULT_LOG_MIN = -2.0f; // Corresponds to KL ~0.01
    static constexpr float DEFAULT_LOG_MAX = 0.5f;  // Corresponds to KL ~3.16
    
    // Adaptive parameters (can be calibrated)
    float log_min_, log_max_;
    float bin_width_;
    
    // Bin assignments
    static constexpr uint8_t UNDERFLOW_BIN = 0;
    static constexpr uint8_t OVERFLOW_BIN = 126;
    static constexpr uint8_t NUM_REGULAR_BINS = 125;
    static constexpr uint8_t REGULAR_BIN_START = 1;
    
    // Random number generator for decompression noise
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<float> noise_dist_;
    
    bool calibrated_;
    
public:
    AdaptiveKLCompressor() 
        : log_min_(DEFAULT_LOG_MIN), log_max_(DEFAULT_LOG_MAX), 
          rng_(std::random_device{}()), noise_dist_(-0.5f, 0.5f),
          calibrated_(false) {
        update_bin_width();
    }
    
    // Calibrate compression parameters based on sample data
    void calibrate(const std::vector<float>& sample_data, float coverage = 0.98f) {
        std::cout << "Calibrating adaptive KL compressor..." << std::endl;
        
        std::vector<float> log_values;
        log_values.reserve(sample_data.size());
        
        // Convert to log scale
        for (float val : sample_data) {
            if (val > 0.0f && std::isfinite(val)) {
                log_values.push_back(std::log10(val / SCALE_FACTOR));
            }
        }
        
        if (log_values.empty()) {
            std::cout << "Warning: No valid values for calibration, using defaults" << std::endl;
            return;
        }
        
        std::sort(log_values.begin(), log_values.end());
        
        // Use percentiles to determine range for adaptive binning
        float lower_percentile = (1.0f - coverage) / 2.0f;
        float upper_percentile = coverage + lower_percentile;
        
        size_t lower_idx = static_cast<size_t>(log_values.size() * lower_percentile);
        size_t upper_idx = static_cast<size_t>(log_values.size() * upper_percentile);
        
        lower_idx = std::max(0UL, std::min(lower_idx, log_values.size() - 1));
        upper_idx = std::max(0UL, std::min(upper_idx, log_values.size() - 1));
        
        log_min_ = log_values[lower_idx];
        log_max_ = log_values[upper_idx];
        
        // Ensure reasonable range
        if (log_max_ - log_min_ < 0.1f) {
            log_min_ = DEFAULT_LOG_MIN;
            log_max_ = DEFAULT_LOG_MAX;
            std::cout << "Warning: Data range too small, using default range" << std::endl;
        }
        
        update_bin_width();
        calibrated_ = true;
        
        // Print calibration results
        float kl_min = SCALE_FACTOR * std::pow(10.0f, log_min_);
        float kl_max = SCALE_FACTOR * std::pow(10.0f, log_max_);
        
        std::cout << "Calibration complete:" << std::endl;
        std::cout << "  Sample size: " << sample_data.size() << " values" << std::endl;
        std::cout << "  Valid values: " << log_values.size() << " values" << std::endl;
        std::cout << "  Coverage: " << (coverage * 100.0f) << "%" << std::endl;
        std::cout << "  Log10(KL/8) range: [" << log_min_ << ", " << log_max_ << "]" << std::endl;
        std::cout << "  KL value range: [" << kl_min << ", " << kl_max << "]" << std::endl;
        std::cout << "  Bin width: " << bin_width_ << std::endl;
        
        // Analyze compression quality
        analyze_compression_quality(log_values);
    }
    
    // Compress single KL divergence value
    uint8_t compress_value(float kl_value) const {
        // Handle special cases
        if (kl_value <= 0.0f || !std::isfinite(kl_value)) {
            return UNDERFLOW_BIN;
        }
        
        // Apply log transformation
        float log_val = std::log10(kl_value / SCALE_FACTOR);
        
        // Handle underflow
        if (log_val < log_min_) {
            return UNDERFLOW_BIN;
        }
        
        // Handle overflow
        if (log_val > log_max_) {
            return OVERFLOW_BIN;
        }
        
        // Map to regular bins [1, 125]
        float normalized = (log_val - log_min_) / (log_max_ - log_min_);
        int bin = static_cast<int>(normalized * NUM_REGULAR_BINS);
        bin = std::max(0, std::min(static_cast<int>(NUM_REGULAR_BINS - 1), bin));
        
        return static_cast<uint8_t>(REGULAR_BIN_START + bin);
    }
    
    // Decompress single value with optional noise
    float decompress_value(uint8_t compressed, bool add_noise = true) const {
        if (compressed == UNDERFLOW_BIN) {
            // Return a small positive value
            float base_val = SCALE_FACTOR * std::pow(10.0f, log_min_ - 0.1f);
            return add_noise ? base_val * (1.0f + 0.1f * noise_dist_(rng_)) : base_val;
        }
        
        if (compressed == OVERFLOW_BIN) {
            // Return a large value
            float base_val = SCALE_FACTOR * std::pow(10.0f, log_max_ + 0.1f);
            return add_noise ? base_val * (1.0f + 0.1f * noise_dist_(rng_)) : base_val;
        }
        
        // Regular bins [1, 125]
        if (compressed >= REGULAR_BIN_START && compressed < REGULAR_BIN_START + NUM_REGULAR_BINS) {
            int bin_idx = compressed - REGULAR_BIN_START;
            
            // Calculate bin center
            float normalized = (static_cast<float>(bin_idx) + 0.5f) / NUM_REGULAR_BINS;
            float log_val = log_min_ + normalized * (log_max_ - log_min_);
            
            // Add small random noise to avoid quantization artifacts
            if (add_noise) {
                log_val += noise_dist_(rng_) * bin_width_ * 0.5f; // Half bin width noise
            }
            
            return SCALE_FACTOR * std::pow(10.0f, log_val);
        }
        
        // Invalid compressed value
        std::cerr << "Warning: Invalid compressed value " << static_cast<int>(compressed) << std::endl;
        return 0.0f;
    }
    
    // Compress vector of KL divergence values
    std::vector<uint8_t> compress_vector(const std::vector<float>& kl_values) const {
        std::vector<uint8_t> compressed;
        compressed.reserve(kl_values.size());
        
        for (float val : kl_values) {
            compressed.push_back(compress_value(val));
        }
        
        return compressed;
    }
    
    // Decompress vector of compressed values
    std::vector<float> decompress_vector(const std::vector<uint8_t>& compressed, bool add_noise = true) const {
        std::vector<float> decompressed;
        decompressed.reserve(compressed.size());
        
        for (uint8_t val : compressed) {
            decompressed.push_back(decompress_value(val, add_noise));
        }
        
        return decompressed;
    }
    
    // Save compression parameters to file
    void save_parameters(const std::string& filename) const {
        std::ofstream file(filename + ".compression_params", std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot write compression parameters to: " + filename);
        }
        
        // Write magic number and version
        uint32_t magic = 0x4B4C5041; // "KLPA" = KL Parameters Adaptive
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Write parameters
        file.write(reinterpret_cast<const char*>(&log_min_), sizeof(log_min_));
        file.write(reinterpret_cast<const char*>(&log_max_), sizeof(log_max_));
        file.write(reinterpret_cast<const char*>(&bin_width_), sizeof(bin_width_));
        
        uint8_t calibrated_flag = calibrated_ ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&calibrated_flag), sizeof(calibrated_flag));
    }
    
    // Load compression parameters from file
    bool load_parameters(const std::string& filename) {
        std::ifstream file(filename + ".compression_params", std::ios::binary);
        if (!file) {
            return false; // File doesn't exist, use defaults
        }
        
        // Read magic number and version
        uint32_t magic, version;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        
        if (magic != 0x4B4C5041 || version != 1) {
            std::cout << "Warning: Invalid compression parameters file, using defaults" << std::endl;
            return false;
        }
        
        // Read parameters
        file.read(reinterpret_cast<char*>(&log_min_), sizeof(log_min_));
        file.read(reinterpret_cast<char*>(&log_max_), sizeof(log_max_));
        file.read(reinterpret_cast<char*>(&bin_width_), sizeof(bin_width_));
        
        uint8_t calibrated_flag;
        file.read(reinterpret_cast<char*>(&calibrated_flag), sizeof(calibrated_flag));
        calibrated_ = (calibrated_flag == 1);
        
        std::cout << "Loaded compression parameters:" << std::endl;
        std::cout << "  Log range: [" << log_min_ << ", " << log_max_ << "]" << std::endl;
        std::cout << "  Calibrated: " << (calibrated_ ? "Yes" : "No") << std::endl;
        
        return true;
    }
    
    // Get compression parameters for metadata
    void get_parameters(float& scale_factor, float& log_min, float& log_max, 
                       float& bin_width, bool& is_calibrated) const {
        scale_factor = SCALE_FACTOR;
        log_min = log_min_;
        log_max = log_max_;
        bin_width = bin_width_;
        is_calibrated = calibrated_;
    }
    
private:
    void update_bin_width() {
        bin_width_ = (log_max_ - log_min_) / NUM_REGULAR_BINS;
    }
    
    void analyze_compression_quality(const std::vector<float>& log_values) const {
        int underflow = 0, overflow = 0, regular = 0;
        
        for (float log_val : log_values) {
            if (log_val < log_min_) underflow++;
            else if (log_val > log_max_) overflow++;
            else regular++;
        }
        
        float coverage = static_cast<float>(regular) / log_values.size();
        
        std::cout << "Compression quality analysis:" << std::endl;
        std::cout << "  Underflow (bin 0): " << underflow << " (" 
                  << (100.0f * underflow / log_values.size()) << "%)" << std::endl;
        std::cout << "  Regular bins (1-125): " << regular << " (" 
                  << (100.0f * coverage) << "%)" << std::endl;
        std::cout << "  Overflow (bin 126): " << overflow << " (" 
                  << (100.0f * overflow / log_values.size()) << "%)" << std::endl;
        
        if (coverage < 0.95f) {
            std::cout << "Warning: Low coverage (" << (coverage * 100.0f) 
                      << "%), consider recalibration" << std::endl;
        }
    }
};

#endif // KL_COMPRESSION_H