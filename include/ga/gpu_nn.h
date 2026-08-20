#pragma once

#ifdef HAS_CUDA

extern "C" {
void cuda_batch_policy_decide(
    const float* h_genes, const float* h_states, int* h_actions,
    int N, int input_size, int hidden_size, int output_size, int gene_size);

void cuda_batch_score(
    const float* h_genes, const float* h_states, float* h_scores,
    int N, int input_size, int hidden_size, int gene_size);

void cuda_batch_scoring_decide(
    const float* h_genes, const float* h_states, const int* h_valid_moves,
    const int* h_move_counts, int* h_best_actions,
    int N, int input_size, int hidden_size, int gene_size, int max_moves);
}

#endif

#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ga {

class GpuNN {
    bool available_ = false;

public:
    GpuNN() {
#ifdef HAS_CUDA
        available_ = true;
#endif
    }

    bool is_available() const { return available_; }

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
            std::vector<float> flat_genes(N * gene_size);
            for (size_t i = 0; i < N; ++i)
                std::memcpy(flat_genes.data() + i * gene_size, genes[i].data(), gene_size * sizeof(float));

            std::vector<int> actions(N);
            cuda_batch_policy_decide(flat_genes.data(), states_flat.data(), actions.data(),
                                     static_cast<int>(N), input_size, hidden_size, output_size,
                                     static_cast<int>(gene_size));
            return actions;
        }
#endif
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

            std::vector<int> actions(N);
            cuda_batch_scoring_decide(flat_genes.data(), flat_states.data(), flat_moves.data(),
                                       move_counts.data(), actions.data(),
                                       static_cast<int>(N), input_size, hidden_size,
                                       static_cast<int>(gene_size), static_cast<int>(max_moves));
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
