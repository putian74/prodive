#include "kl_divergence.h"
#include "debug_array.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <map>
#include <fstream>
#include <algorithm>
#include <numeric>

struct DebugArrays {
    std::vector<float> W1, Q1, W2, Q2;
    std::vector<float> kl12, kl21;
    std::vector<int> flags;
    
    DebugArrays(int nfrag1, int nfrag2) {
        int total_pairs = nfrag1 * nfrag2;
        W1.resize(total_pairs * M_STATES);
        W2.resize(total_pairs * M_STATES);
        Q1.resize(total_pairs);
        Q2.resize(total_pairs);
        kl12.resize(total_pairs);
        kl21.resize(total_pairs);
        flags.resize(total_pairs);
    }
};

// Function to save debug data as text files (avoiding cnpy segfault)
void save_debug_data(const std::vector<float>& debug_W1, 
                    const std::vector<float>& debug_Q1,
                    const std::vector<float>& debug_W2, 
                    const std::vector<float>& debug_Q2,
                    const std::vector<float>& debug_kl12, 
                    const std::vector<float>& debug_kl21,
                    const std::vector<int>& debug_flags,
                    int i, int j, int nfrag1, int nfrag2, 
                    const std::string& debug_dir) {
    
    try {
        std::ostringstream fname_base;
        fname_base << debug_dir << "/debug_" << std::setfill('0') << std::setw(4) << i
                   << "_" << std::setfill('0') << std::setw(4) << j;
        
        std::string base = fname_base.str();
        
        // Save W1 debug data (nfrag1 * nfrag2 * M_STATES)
        {
            std::ofstream file(base + "_W1.txt");
            if (file.is_open()) {
                file << "# W1 debug data for pair (" << i << ", " << j << ")\n";
                file << "# Shape: " << nfrag1 << " x " << nfrag2 << " x " << M_STATES << "\n";
                for (int f1 = 0; f1 < nfrag1; ++f1) {
                    for (int f2 = 0; f2 < nfrag2; ++f2) {
                        file << "Frag(" << f1 << "," << f2 << "): ";
                        for (int m = 0; m < M_STATES; ++m) {
                            int idx = (f1 * nfrag2 + f2) * M_STATES + m;
                            if (idx < debug_W1.size()) {
                                file << std::scientific << std::setprecision(6) << debug_W1[idx] << " ";
                            }
                        }
                        file << "\n";
                    }
                }
                file.close();
            }
        }
        
        // Save W2 debug data
        {
            std::ofstream file(base + "_W2.txt");
            if (file.is_open()) {
                file << "# W2 debug data for pair (" << i << ", " << j << ")\n";
                file << "# Shape: " << nfrag1 << " x " << nfrag2 << " x " << M_STATES << "\n";
                for (int f1 = 0; f1 < nfrag1; ++f1) {
                    for (int f2 = 0; f2 < nfrag2; ++f2) {
                        file << "Frag(" << f1 << "," << f2 << "): ";
                        for (int m = 0; m < M_STATES; ++m) {
                            int idx = (f1 * nfrag2 + f2) * M_STATES + m;
                            if (idx < debug_W2.size()) {
                                file << std::scientific << std::setprecision(6) << debug_W2[idx] << " ";
                            }
                        }
                        file << "\n";
                    }
                }
                file.close();
            }
        }
        
        // Save scalar debug data (Q1, Q2, kl12, kl21, flags)
        {
            std::ofstream file(base + "_scalars.txt");
            if (file.is_open()) {
                file << "# Scalar debug data for pair (" << i << ", " << j << ")\n";
                file << "# Format: frag1 frag2 Q1 Q2 kl12 kl21 flags\n";
                for (int f1 = 0; f1 < nfrag1; ++f1) {
                    for (int f2 = 0; f2 < nfrag2; ++f2) {
                        int idx = f1 * nfrag2 + f2;
                        if (idx < debug_Q1.size()) {
                            file << f1 << " " << f2 << " "
                                 << std::scientific << std::setprecision(6)
                                 << debug_Q1[idx] << " "
                                 << debug_Q2[idx] << " "
                                 << debug_kl12[idx] << " "
                                 << debug_kl21[idx] << " "
                                 << std::hex << debug_flags[idx] << std::dec << "\n";
                        }
                    }
                }
                file.close();
            }
        }
        
        // Save summary statistics
        {
            std::ofstream file(base + "_summary.txt");
            if (file.is_open()) {
                file << "Debug Summary for pair (" << i << ", " << j << ")\n";
                file << "Matrix size: " << nfrag1 << " x " << nfrag2 << "\n\n";
                
                // Statistics for Q1
                if (!debug_Q1.empty()) {
                    auto q1_minmax = std::minmax_element(debug_Q1.begin(), debug_Q1.end());
                    double q1_sum = std::accumulate(debug_Q1.begin(), debug_Q1.end(), 0.0);
                    file << "Q1 statistics:\n";
                    file << "  Min: " << *q1_minmax.first << "\n";
                    file << "  Max: " << *q1_minmax.second << "\n";
                    file << "  Mean: " << (q1_sum / debug_Q1.size()) << "\n\n";
                }
                
                // Statistics for Q2
                if (!debug_Q2.empty()) {
                    auto q2_minmax = std::minmax_element(debug_Q2.begin(), debug_Q2.end());
                    double q2_sum = std::accumulate(debug_Q2.begin(), debug_Q2.end(), 0.0);
                    file << "Q2 statistics:\n";
                    file << "  Min: " << *q2_minmax.first << "\n";
                    file << "  Max: " << *q2_minmax.second << "\n";
                    file << "  Mean: " << (q2_sum / debug_Q2.size()) << "\n\n";
                }
                
                // Statistics for KL12
                if (!debug_kl12.empty()) {
                    auto kl12_minmax = std::minmax_element(debug_kl12.begin(), debug_kl12.end());
                    double kl12_sum = std::accumulate(debug_kl12.begin(), debug_kl12.end(), 0.0);
                    file << "KL(1||2) statistics:\n";
                    file << "  Min: " << *kl12_minmax.first << "\n";
                    file << "  Max: " << *kl12_minmax.second << "\n";
                    file << "  Mean: " << (kl12_sum / debug_kl12.size()) << "\n\n";
                }
                
                // Statistics for KL21
                if (!debug_kl21.empty()) {
                    auto kl21_minmax = std::minmax_element(debug_kl21.begin(), debug_kl21.end());
                    double kl21_sum = std::accumulate(debug_kl21.begin(), debug_kl21.end(), 0.0);
                    file << "KL(2||1) statistics:\n";
                    file << "  Min: " << *kl21_minmax.first << "\n";
                    file << "  Max: " << *kl21_minmax.second << "\n";
                    file << "  Mean: " << (kl21_sum / debug_kl21.size()) << "\n\n";
                }
                
                // Flag statistics
                int total_flags = std::count_if(debug_flags.begin(), debug_flags.end(), 
                                              [](int flag) { return flag != 0; });
                file << "Flags: " << total_flags << " / " << debug_flags.size() 
                     << " pairs had issues (" << (100.0 * total_flags / debug_flags.size()) << "%)\n";
                file.close();
            }
        }
        
        std::cout << "Debug data saved to " << base << "_*.txt" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving debug data: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred while saving debug data" << std::endl;
    }
}

// New overloaded function to match main.cpp calls (with i, j parameters)
void analyze_debug_flags(const std::vector<int>& debug_flags, int nfrag1, int nfrag2, int i, int j) {
    std::cout << "\n=== Debug Analysis for pair (" << i << ", " << j << ") ===" << std::endl;
    std::cout << "Matrix size: " << nfrag1 << " x " << nfrag2 << std::endl;
    
    std::map<std::string, int> issue_counts;
    issue_counts["Invalid pi1"] = 0;
    issue_counts["Invalid pi2"] = 0;
    issue_counts["Invalid A1"] = 0;
    issue_counts["Invalid A2"] = 0;
    issue_counts["Invalid inv1"] = 0;
    issue_counts["Invalid inv2"] = 0;
    issue_counts["Invalid B1"] = 0;
    issue_counts["Invalid B2"] = 0;
    issue_counts["Unnormalized pi1"] = 0;
    issue_counts["Unnormalized pi2"] = 0;
    issue_counts["Invalid kl12"] = 0;
    issue_counts["Invalid kl21"] = 0;
    issue_counts["Invalid sym_kl"] = 0;
    
    int total_issues = 0;
    for (size_t idx = 0; idx < debug_flags.size(); ++idx) {
        int flag = debug_flags[idx];
        if (flag != 0) {
            total_issues++;
            if (flag & 0x01) issue_counts["Invalid pi1"]++;
            if (flag & 0x02) issue_counts["Invalid pi2"]++;
            if (flag & 0x04) issue_counts["Invalid A1"]++;
            if (flag & 0x08) issue_counts["Invalid A2"]++;
            if (flag & 0x10) issue_counts["Invalid inv1"]++;
            if (flag & 0x20) issue_counts["Invalid inv2"]++;
            if (flag & 0x40) issue_counts["Invalid B1"]++;
            if (flag & 0x80) issue_counts["Invalid B2"]++;
            if (flag & 0x100) issue_counts["Unnormalized pi1"]++;
            if (flag & 0x200) issue_counts["Unnormalized pi2"]++;
            if (flag & 0x400) issue_counts["Invalid kl12"]++;
            if (flag & 0x800) issue_counts["Invalid kl21"]++;
            if (flag & 0x1000) issue_counts["Invalid sym_kl"]++;
        }
    }
    
    std::cout << "Total problematic pairs: " << total_issues << " / " << debug_flags.size() 
              << " (" << (100.0 * total_issues / debug_flags.size()) << "%)" << std::endl;
    
    if (total_issues > 0) {
        std::cout << "Issue breakdown:" << std::endl;
        for (const auto& pair : issue_counts) {
            if (pair.second > 0) {
                std::cout << "  " << pair.first << ": " << pair.second << " cases" << std::endl;
            }
        }
    }
    std::cout << "=========================================" << std::endl;
}

// Keep original function for backward compatibility if needed
void save_debug_data(const DebugArrays& debug, int i, int j, int nfrag1, int nfrag2, 
                     const std::string& output_dir) {
    // Call the new function
    save_debug_data(debug.W1, debug.Q1, debug.W2, debug.Q2, 
                   debug.kl12, debug.kl21, debug.flags,
                   i, j, nfrag1, nfrag2, output_dir);
}

void analyze_debug_flags(const std::vector<int>& flags, int nfrag1, int nfrag2) {
    // Call the new function with dummy i, j values
    analyze_debug_flags(flags, nfrag1, nfrag2, -1, -1);
}