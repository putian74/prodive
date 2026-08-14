#ifndef DEBUG_ARRAY_H
#define DEBUG_ARRAY_H

#include <vector>
#include <string>

#ifndef M_STATES
#define M_STATES 3
#endif

// Forward declaration for DebugArrays struct
struct DebugArrays;

// Original functions
void analyze_debug_flags(const std::vector<int>& flags, int nfrag1, int nfrag2);
void save_debug_data(const DebugArrays& debug, int i, int j, int nfrag1, int nfrag2, 
                     const std::string& output_dir);

// Overloaded functions to match main.cpp calls
void analyze_debug_flags(const std::vector<int>& debug_flags, int nfrag1, int nfrag2, int i, int j);
void save_debug_data(const std::vector<float>& debug_W1, 
                    const std::vector<float>& debug_Q1,
                    const std::vector<float>& debug_W2, 
                    const std::vector<float>& debug_Q2,
                    const std::vector<float>& debug_kl12, 
                    const std::vector<float>& debug_kl21,
                    const std::vector<int>& debug_flags,
                    int i, int j, int nfrag1, int nfrag2, 
                    const std::string& debug_dir);

#endif // DEBUG_ARRAY_H