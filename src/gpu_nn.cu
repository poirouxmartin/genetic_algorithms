#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <cstring>

__device__ float tanhf_approx(float x) {
    return tanhf(x);
}

// Batch PolicyNetwork forward pass
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
    int best_action = 0;
    float best_val = -1e30f;
    for (int o = 0; o < output_size; ++o) {
        float sum = genes[offset + hidden_size * output_size + o];
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
    float sum = genes[offset + hidden_size];
    for (int h = 0; h < hidden_size; ++h) {
        sum += genes[offset + h] * hidden[h];
    }
    scores[idx] = sum;
}

// Batch scoring decide: for each genome, score all valid moves
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

    float best_score = -1e30f;
    int best_action = valid_moves[idx * max_moves];

    for (int m = 0; m < move_count; ++m) {
        const float* move_state = states + (idx * max_moves + m) * input_size;

        int offset = 0;
        for (int h = 0; h < hidden_size; ++h) {
            float sum = genes[input_size * hidden_size + h];
            for (int i = 0; i < input_size; ++i) {
                sum += genes[offset + i] * move_state[i];
            }
            hidden[h] = tanhf_approx(sum);
            offset += input_size;
        }

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

// Host wrappers with CUDA memory management
extern "C" {

void cuda_batch_policy_decide(
    const float* h_genes, const float* h_states, int* h_actions,
    int N, int input_size, int hidden_size, int output_size, int gene_size)
{
    float *d_genes, *d_states;
    int* d_actions;
    cudaMalloc(&d_genes, N * gene_size * sizeof(float));
    cudaMalloc(&d_states, N * input_size * sizeof(float));
    cudaMalloc(&d_actions, N * sizeof(int));

    cudaMemcpy(d_genes, h_genes, N * gene_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, h_states, N * input_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = (hidden_size + output_size) * sizeof(float);
    batch_policy_decide_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_actions, N, input_size, hidden_size, output_size, gene_size);

    cudaMemcpy(h_actions, d_actions, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_genes);
    cudaFree(d_states);
    cudaFree(d_actions);
}

void cuda_batch_score(
    const float* h_genes, const float* h_states, float* h_scores,
    int N, int input_size, int hidden_size, int gene_size)
{
    float *d_genes, *d_states, *d_scores;
    cudaMalloc(&d_genes, N * gene_size * sizeof(float));
    cudaMalloc(&d_states, N * input_size * sizeof(float));
    cudaMalloc(&d_scores, N * sizeof(float));

    cudaMemcpy(d_genes, h_genes, N * gene_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, h_states, N * input_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = hidden_size * sizeof(float);
    batch_score_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_scores, N, input_size, hidden_size, gene_size);

    cudaMemcpy(h_scores, d_scores, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_genes);
    cudaFree(d_states);
    cudaFree(d_scores);
}

void cuda_batch_scoring_decide(
    const float* h_genes, const float* h_states, const int* h_valid_moves,
    const int* h_move_counts, int* h_best_actions,
    int N, int input_size, int hidden_size, int gene_size, int max_moves)
{
    float *d_genes, *d_states;
    int *d_valid_moves, *d_move_counts, *d_best_actions;
    cudaMalloc(&d_genes, N * gene_size * sizeof(float));
    cudaMalloc(&d_states, N * max_moves * input_size * sizeof(float));
    cudaMalloc(&d_valid_moves, N * max_moves * sizeof(int));
    cudaMalloc(&d_move_counts, N * sizeof(int));
    cudaMalloc(&d_best_actions, N * sizeof(int));

    cudaMemcpy(d_genes, h_genes, N * gene_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, h_states, N * max_moves * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid_moves, h_valid_moves, N * max_moves * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_move_counts, h_move_counts, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = (hidden_size + max_moves) * sizeof(float);
    batch_scoring_decide_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_valid_moves, d_move_counts, d_best_actions,
        N, input_size, hidden_size, gene_size, max_moves);

    cudaMemcpy(h_best_actions, d_best_actions, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_genes);
    cudaFree(d_states);
    cudaFree(d_valid_moves);
    cudaFree(d_move_counts);
    cudaFree(d_best_actions);
}

} // extern "C"

#endif // HAS_CUDA
