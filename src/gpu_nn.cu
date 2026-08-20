#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <cmath>

__device__ float tanhf_approx(float x) {
    return tanhf(x);
}

// Batch PolicyNetwork forward pass
// Each thread processes one genome's forward pass
// genome_genes: [N * gene_size] - all genomes' weights concatenated
// states: [N * input_size] - all genomes' current states
// actions: [N] - output actions (argmax of output layer)
__global__ void batch_policy_decide_kernel(
    const float* __restrict__ genome_genes,
    const float* __restrict__ states,
    int* __restrict__ actions,
    int N, int input_size, int hidden_size, int output_size,
    int gene_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* genes = genome_genes + idx * gene_size;
    const float* state = states + idx * input_size;

    extern __shared__ float shared_mem[];
    float* hidden = shared_mem;
    float* outs = shared_mem + hidden_size;

    // Hidden layer: tanh(W*h + b)
    int offset = 0;
    for (int h = 0; h < hidden_size; ++h) {
        float sum = genes[input_size * hidden_size + h]; // bias
        for (int i = 0; i < input_size; ++i) {
            sum += genes[offset + i] * state[i];
        }
        hidden[h] = tanhf_approx(sum);
        offset += input_size;
    }

    // Output layer: argmax
    offset = input_size * hidden_size + hidden_size;
    int best_action = 0;
    float best_val = -1e30f;
    for (int o = 0; o < output_size; ++o) {
        float sum = genes[offset + hidden_size * output_size + o]; // bias
        for (int h = 0; h < hidden_size; ++h) {
            sum += genes[offset + o * hidden_size + h] * hidden[h];
        }
        outs[o] = sum;
        if (sum > best_val) {
            best_val = sum;
            best_action = o;
        }
    }
    actions[idx] = best_action;
}

// Batch ScoringNetwork forward pass
// Each thread scores one (genome, state) pair
// scores: [N] - output scores
__global__ void batch_score_kernel(
    const float* __restrict__ genome_genes,
    const float* __restrict__ states,
    float* __restrict__ scores,
    int N, int input_size, int hidden_size,
    int gene_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* genes = genome_genes + idx * gene_size;
    const float* state = states + idx * input_size;

    extern __shared__ float hidden[];

    int offset = 0;
    for (int h = 0; h < hidden_size; ++h) {
        float sum = genes[input_size * hidden_size + h];
        for (int i = 0; i < input_size; ++i) {
            sum += genes[offset + i] * state[i];
        }
        hidden[h] = tanhf_approx(sum);
        offset += input_size;
    }

    offset = input_size * hidden_size + hidden_size;
    float sum = genes[offset + hidden_size]; // output bias
    for (int h = 0; h < hidden_size; ++h) {
        sum += genes[offset + h] * hidden[h];
    }
    scores[idx] = sum;
}

// Batch scoring: for each genome, score all valid moves, return best
// For scoring networks: N genomes × M valid moves
// genome_genes: [N genomes × gene_size]
// states: [N × M × input_size] (state after each move)
// valid_moves: [N × M] (valid move indices)
// best_actions: [N] - output best action per genome
__global__ void batch_scoring_decide_kernel(
    const float* __restrict__ genome_genes,
    const float* __restrict__ states,
    const int* __restrict__ valid_moves,
    const int* __restrict__ move_counts,
    int* __restrict__ best_actions,
    int N, int input_size, int hidden_size,
    int gene_size, int max_moves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* genes = genome_genes + idx * gene_size;
    int move_count = move_counts[idx];

    extern __shared__ float shared_mem[];
    float* hidden = shared_mem;
    float* move_scores = shared_mem + hidden_size;

    // Precompute hidden layer from genome weights (same for all moves)
    int offset = 0;
    for (int h = 0; h < hidden_size; ++h) {
        float sum = genes[input_size * hidden_size + h];
        for (int i = 0; i < input_size; ++i) {
            // Use state[0] as placeholder for bias computation
            // actual state passed per-move below
        }
        // We need to compute hidden per-move since states differ
        // This is handled differently - see below
    }

    // For scoring: compute hidden for each move's state, then score
    float best_score = -1e30f;
    int best_action = valid_moves[idx * max_moves]; // default to first

    for (int m = 0; m < move_count; ++m) {
        const float* move_state = states + (idx * max_moves + m) * input_size;

        // Hidden layer
        offset = 0;
        for (int h = 0; h < hidden_size; ++h) {
            float sum = genes[input_size * hidden_size + h];
            for (int i = 0; i < input_size; ++i) {
                sum += genes[offset + i] * move_state[i];
            }
            hidden[h] = tanhf_approx(sum);
            offset += input_size;
        }

        // Output (single scalar)
        offset = input_size * hidden_size + hidden_size;
        float score = genes[offset + hidden_size];
        for (int h = 0; h < hidden_size; ++h) {
            score += genes[offset + h] * hidden[h];
        }

        if (score > best_score) {
            best_score = score;
            best_action = valid_moves[idx * max_moves + m];
        }
    }

    best_actions[idx] = best_action;
}

extern "C" {

void cuda_batch_policy_decide(
    const float* d_genes, const float* d_states, int* d_actions,
    int N, int input_size, int hidden_size, int output_size, int gene_size)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = (hidden_size + output_size) * sizeof(float);
    batch_policy_decide_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_actions, N, input_size, hidden_size, output_size, gene_size);
}

void cuda_batch_score(
    const float* d_genes, const float* d_states, float* d_scores,
    int N, int input_size, int hidden_size, int gene_size)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = hidden_size * sizeof(float);
    batch_score_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_scores, N, input_size, hidden_size, gene_size);
}

void cuda_batch_scoring_decide(
    const float* d_genes, const float* d_states, const int* d_valid_moves,
    const int* d_move_counts, int* d_best_actions,
    int N, int input_size, int hidden_size, int gene_size, int max_moves)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = (hidden_size + max_moves) * sizeof(float);
    batch_scoring_decide_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_valid_moves, d_move_counts, d_best_actions,
        N, input_size, hidden_size, gene_size, max_moves);
}

} // extern "C"

#endif // HAS_CUDA
