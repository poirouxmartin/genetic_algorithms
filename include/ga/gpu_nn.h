#pragma once

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ga {

class GpuNN {
    bool available_ = false;
    bool initialized_ = false;

#ifdef HAS_CUDA
    float* d_genes_ = nullptr;
    float* d_states_ = nullptr;
    int* d_actions_ = nullptr;
    float* d_scores_ = nullptr;
    int* d_valid_moves_ = nullptr;
    int* d_move_counts_ = nullptr;
    int* d_best_actions_ = nullptr;
    size_t allocated_N_ = 0;
    size_t allocated_gene_size_ = 0;
    size_t allocated_input_size_ = 0;
    size_t allocated_max_moves_ = 0;

    void ensure_alloc(size_t N, size_t gene_size, size_t input_size, size_t max_moves = 0) {
        if (N <= allocated_N_ && gene_size <= allocated_gene_size_ &&
            input_size <= allocated_input_size_ && max_moves <= allocated_max_moves_)
            return;

        free_gpu();

        cudaMalloc(&d_genes_, N * gene_size * sizeof(float));
        cudaMalloc(&d_states_, N * input_size * sizeof(float));
        cudaMalloc(&d_actions_, N * sizeof(int));
        cudaMalloc(&d_scores_, N * sizeof(float));
        if (max_moves > 0) {
            cudaMalloc(&d_valid_moves_, N * max_moves * sizeof(int));
            cudaMalloc(&d_move_counts_, N * sizeof(int));
            cudaMalloc(&d_best_actions_, N * sizeof(int));
        }

        allocated_N_ = N;
        allocated_gene_size_ = gene_size;
        allocated_input_size_ = input_size;
        allocated_max_moves_ = max_moves;
    }

    void free_gpu() {
        if (d_genes_) cudaFree(d_genes_);
        if (d_states_) cudaFree(d_states_);
        if (d_actions_) cudaFree(d_actions_);
        if (d_scores_) cudaFree(d_scores_);
        if (d_valid_moves_) cudaFree(d_valid_moves_);
        if (d_move_counts_) cudaFree(d_move_counts_);
        if (d_best_actions_) cudaFree(d_best_actions_);
        d_genes_ = d_states_ = d_scores_ = nullptr;
        d_actions_ = d_valid_moves_ = d_move_counts_ = d_best_actions_ = nullptr;
    }
#endif

public:
    GpuNN() {
#ifdef HAS_CUDA
        cudaError_t err = cudaSetDevice(0);
        if (err == cudaSuccess) {
            available_ = true;
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
        }
#endif
    }

    ~GpuNN() {
#ifdef HAS_CUDA
        free_gpu();
#endif
    }

    bool is_available() const { return available_; }
    bool is_initialized() const { return initialized_; }

    // Batch PolicyNetwork: decide actions for N genomes
    std::vector<int> batch_policy_decide(
        const std::vector<std::vector<float>>& genes,
        const std::vector<float>& states_flat,
        int input_size, int hidden_size, int output_size)
    {
        size_t N = genes.size();
        size_t gene_size = genes[0].size();

#ifdef HAS_CUDA
        if (available_ && N >= 32) {
            ensure_alloc(N, gene_size, input_size);

            // Pack genes into contiguous buffer
            std::vector<float> flat_genes(N * gene_size);
            for (size_t i = 0; i < N; ++i)
                std::memcpy(flat_genes.data() + i * gene_size, genes[i].data(), gene_size * sizeof(float));

            cudaMemcpy(d_genes_, flat_genes.data(), N * gene_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_states_, states_flat.data(), N * input_size * sizeof(float), cudaMemcpyHostToDevice);

            cuda_batch_policy_decide(d_genes_, d_states_, d_actions_,
                                     N, input_size, hidden_size, output_size, gene_size);

            std::vector<int> actions(N);
            cudaMemcpy(actions.data(), d_actions_, N * sizeof(int), cudaMemcpyDeviceToHost);
            return actions;
        }
#endif
        // CPU fallback
        return cpu_policy_decide(genes, states_flat, input_size, hidden_size, output_size);
    }

    // Batch ScoringNetwork: score all valid moves per genome, return best
    std::vector<int> batch_scoring_decide(
        const std::vector<std::vector<float>>& genes,
        const std::vector<std::vector<std::vector<float>>>& move_states,
        const std::vector<std::vector<int>>& valid_moves,
        int input_size, int hidden_size)
    {
        size_t N = genes.size();
        size_t gene_size = genes[0].size();
        size_t max_moves = 0;
        for (auto& vm : valid_moves) max_moves = std::max(max_moves, vm.size());
        if (max_moves == 0) return std::vector<int>(N, 0);

#ifdef HAS_CUDA
        if (available_ && N >= 32) {
            ensure_alloc(N, gene_size, input_size, max_moves);

            std::vector<float> flat_genes(N * gene_size);
            for (size_t i = 0; i < N; ++i)
                std::memcpy(flat_genes.data() + i * gene_size, genes[i].data(), gene_size * sizeof(float));

            std::vector<float> flat_states(N * max_moves * input_size, 0.0f);
            std::vector<int> flat_moves(N * max_moves, 0);
            std::vector<int> move_counts(N);

            for (size_t i = 0; i < N; ++i) {
                move_counts[i] = static_cast<int>(valid_moves[i].size());
                for (size_t m = 0; m < valid_moves[i].size(); ++m) {
                    flat_moves[i * max_moves + m] = valid_moves[i][m];
                    std::memcpy(flat_states.data() + (i * max_moves + m) * input_size,
                                move_states[i][m].data(), input_size * sizeof(float));
                }
            }

            cudaMemcpy(d_genes_, flat_genes.data(), N * gene_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_states_, flat_states.data(), N * max_moves * input_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_valid_moves_, flat_moves.data(), N * max_moves * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_move_counts_, move_counts.data(), N * sizeof(int), cudaMemcpyHostToDevice);

            cuda_batch_scoring_decide(d_genes_, d_states_, d_valid_moves_, d_move_counts_, d_best_actions_,
                                      N, input_size, hidden_size, gene_size, max_moves);

            std::vector<int> actions(N);
            cudaMemcpy(actions.data(), d_best_actions_, N * sizeof(int), cudaMemcpyDeviceToHost);
            return actions;
        }
#endif
        return cpu_scoring_decide(genes, move_states, valid_moves, input_size, hidden_size);
    }

private:
    std::vector<int> cpu_policy_decide(
        const std::vector<std::vector<float>>& genes,
        const std::vector<float>& states_flat,
        int input_size, int hidden_size, int output_size)
    {
        size_t N = genes.size();
        std::vector<int> actions(N);
        for (size_t idx = 0; idx < N; ++idx) {
            const float* g = genes[idx].data();
            const float* s = states_flat.data() + idx * input_size;

            std::vector<float> hidden(hidden_size);
            size_t offset = 0;
            for (int h = 0; h < hidden_size; ++h) {
                float sum = g[input_size * hidden_size + h];
                for (int i = 0; i < input_size; ++i)
                    sum += g[offset + i] * s[i];
                hidden[h] = std::tanh(sum);
                offset += input_size;
            }

            offset = input_size * hidden_size + hidden_size;
            int best = 0;
            float best_val = -1e30f;
            for (int o = 0; o < output_size; ++o) {
                float sum = g[offset + hidden_size * output_size + o];
                for (int h = 0; h < hidden_size; ++h)
                    sum += g[offset + o * hidden_size + h] * hidden[h];
                if (sum > best_val) { best_val = sum; best = o; }
            }
            actions[idx] = best;
        }
        return actions;
    }

    std::vector<int> cpu_scoring_decide(
        const std::vector<std::vector<float>>& genes,
        const std::vector<std::vector<std::vector<float>>>& move_states,
        const std::vector<std::vector<int>>& valid_moves,
        int input_size, int hidden_size)
    {
        size_t N = genes.size();
        std::vector<int> actions(N);
        for (size_t idx = 0; idx < N; ++idx) {
            const float* g = genes[idx].data();
            int best_action = valid_moves[idx].empty() ? 0 : valid_moves[idx][0];
            float best_score = -1e30f;

            for (size_t m = 0; m < valid_moves[idx].size(); ++m) {
                const float* s = move_states[idx][m].data();

                std::vector<float> hidden(hidden_size);
                size_t offset = 0;
                for (int h = 0; h < hidden_size; ++h) {
                    float sum = g[input_size * hidden_size + h];
                    for (int i = 0; i < input_size; ++i)
                        sum += g[offset + i] * s[i];
                    hidden[h] = std::tanh(sum);
                    offset += input_size;
                }

                offset = input_size * hidden_size + hidden_size;
                float score = g[offset + hidden_size];
                for (int h = 0; h < hidden_size; ++h)
                    score += g[offset + h] * hidden[h];

                if (score > best_score) {
                    best_score = score;
                    best_action = valid_moves[idx][m];
                }
            }
            actions[idx] = best_action;
        }
        return actions;
    }
};

} // namespace ga
