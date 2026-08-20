#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <cstring>

__device__ float tanhf_approx(float x) {
    return tanhf(x);
}

// Multi-layer forward pass: returns output into `output` buffer.
// layers: [input, h1, h2, ..., hn, output]
// genes: weights+biases packed sequentially per layer pair.
__device__ void forward_pass_device(
    const float* genes, const float* input_vec,
    const int* layers, int num_layers,
    float* hidden_buf, float* output_buf)
{
    const float* cur = input_vec;
    float* nxt = hidden_buf;
    size_t offset = 0;
    for (int l = 0; l < num_layers - 1; ++l) {
        int fan_in = layers[l];
        int fan_out = layers[l + 1];
        bool is_output = (l + 2 == num_layers);
        for (int o = 0; o < fan_out; ++o) {
            float sum = genes[offset + fan_in * fan_out + o];
            for (int i = 0; i < fan_in; ++i)
                sum += genes[offset + o * fan_in + i] * cur[i];
            nxt[o] = is_output ? sum : tanhf_approx(sum);
        }
        cur = nxt;
        nxt = (nxt == hidden_buf) ? output_buf : hidden_buf;
        offset += fan_in * fan_out + fan_out;
    }
    // Result is in cur, which points to either hidden_buf or output_buf
    // Copy to output_buf if needed
    if (cur != output_buf) {
        int out_size = layers[num_layers - 1];
        for (int i = 0; i < out_size; ++i) output_buf[i] = cur[i];
    }
}

// Batch PolicyNetwork forward pass
__global__ void batch_policy_decide_kernel(
    const float* __restrict__ genome_genes,
    const float* __restrict__ states,
    int* __restrict__ actions,
    int N, const int* __restrict__ layers, int num_layers, int gene_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* genes = genome_genes + idx * gene_size;
    const float* state = states + idx * layers[0];

    extern __shared__ float shared_mem[];
    float* hidden_buf = shared_mem;
    float* output_buf = shared_mem + gene_size; // temporary, sized to max layer
    float* result_buf = shared_mem + 2 * gene_size;

    forward_pass_device(genes, state, layers, num_layers, hidden_buf, result_buf);

    int output_size = layers[num_layers - 1];
    int best_action = 0;
    float best_val = -1e30f;
    for (int o = 0; o < output_size; ++o) {
        if (result_buf[o] > best_val) { best_val = result_buf[o]; best_action = o; }
    }
    actions[idx] = best_action;
}

// Batch ScoringNetwork forward pass
__global__ void batch_score_kernel(
    const float* __restrict__ genome_genes,
    const float* __restrict__ states,
    float* __restrict__ scores,
    int N, const int* __restrict__ layers, int num_layers, int gene_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* genes = genome_genes + idx * gene_size;
    const float* state = states + idx * layers[0];

    extern __shared__ float shared_mem[];
    float* hidden_buf = shared_mem;
    float* output_buf = shared_mem + gene_size;
    float* result_buf = shared_mem + 2 * gene_size;

    forward_pass_device(genes, state, layers, num_layers, hidden_buf, result_buf);
    scores[idx] = result_buf[0];
}

// Batch scoring decide: for each genome, score all valid moves
__global__ void batch_scoring_decide_kernel(
    const float* __restrict__ genome_genes,
    const float* __restrict__ states,
    const int* __restrict__ valid_moves,
    const int* __restrict__ move_counts,
    int* __restrict__ best_actions,
    int N, const int* __restrict__ layers, int num_layers,
    int gene_size, int max_moves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* genes = genome_genes + idx * gene_size;
    int move_count = move_counts[idx];

    extern __shared__ float shared_mem[];
    float* hidden_buf = shared_mem;
    float* output_buf = shared_mem + gene_size;
    float* result_buf = shared_mem + 2 * gene_size;

    float best_score = -1e30f;
    int best_action = valid_moves[idx * max_moves];

    for (int m = 0; m < move_count; ++m) {
        const float* move_state = states + (idx * max_moves + m) * layers[0];
        forward_pass_device(genes, move_state, layers, num_layers, hidden_buf, result_buf);
        float score = result_buf[0];
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
    int N, const int* h_layers, int num_layers, int gene_size)
{
    float *d_genes, *d_states;
    int *d_actions, *d_layers;
    cudaMalloc(&d_genes, N * gene_size * sizeof(float));
    cudaMalloc(&d_states, N * h_layers[0] * sizeof(float));
    cudaMalloc(&d_actions, N * sizeof(int));
    cudaMalloc(&d_layers, num_layers * sizeof(int));

    cudaMemcpy(d_genes, h_genes, N * gene_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, h_states, N * h_layers[0] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layers, h_layers, num_layers * sizeof(int), cudaMemcpyHostToDevice);

    int max_dim = 0;
    for (int i = 0; i < num_layers; ++i)
        if (h_layers[i] > max_dim) max_dim = h_layers[i];

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = 3 * max_dim * sizeof(float);
    batch_policy_decide_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_actions, N, d_layers, num_layers, gene_size);

    cudaMemcpy(h_actions, d_actions, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_genes);
    cudaFree(d_states);
    cudaFree(d_actions);
    cudaFree(d_layers);
}

void cuda_batch_score(
    const float* h_genes, const float* h_states, float* h_scores,
    int N, const int* h_layers, int num_layers, int gene_size)
{
    float *d_genes, *d_states, *d_scores;
    int *d_layers;
    cudaMalloc(&d_genes, N * gene_size * sizeof(float));
    cudaMalloc(&d_states, N * h_layers[0] * sizeof(float));
    cudaMalloc(&d_scores, N * sizeof(float));
    cudaMalloc(&d_layers, num_layers * sizeof(int));

    cudaMemcpy(d_genes, h_genes, N * gene_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, h_states, N * h_layers[0] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layers, h_layers, num_layers * sizeof(int), cudaMemcpyHostToDevice);

    int max_dim = 0;
    for (int i = 0; i < num_layers; ++i)
        if (h_layers[i] > max_dim) max_dim = h_layers[i];

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = 3 * max_dim * sizeof(float);
    batch_score_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_scores, N, d_layers, num_layers, gene_size);

    cudaMemcpy(h_scores, d_scores, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_genes);
    cudaFree(d_states);
    cudaFree(d_scores);
    cudaFree(d_layers);
}

void cuda_batch_scoring_decide(
    const float* h_genes, const float* h_states, const int* h_valid_moves,
    const int* h_move_counts, int* h_best_actions,
    int N, const int* h_layers, int num_layers, int gene_size, int max_moves)
{
    float *d_genes, *d_states;
    int *d_valid_moves, *d_move_counts, *d_best_actions, *d_layers;
    cudaMalloc(&d_genes, N * gene_size * sizeof(float));
    cudaMalloc(&d_states, N * max_moves * h_layers[0] * sizeof(float));
    cudaMalloc(&d_valid_moves, N * max_moves * sizeof(int));
    cudaMalloc(&d_move_counts, N * sizeof(int));
    cudaMalloc(&d_best_actions, N * sizeof(int));
    cudaMalloc(&d_layers, num_layers * sizeof(int));

    cudaMemcpy(d_genes, h_genes, N * gene_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, h_states, N * max_moves * h_layers[0] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid_moves, h_valid_moves, N * max_moves * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_move_counts, h_move_counts, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layers, h_layers, num_layers * sizeof(int), cudaMemcpyHostToDevice);

    int max_dim = 0;
    for (int i = 0; i < num_layers; ++i)
        if (h_layers[i] > max_dim) max_dim = h_layers[i];

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_size = 3 * max_dim * sizeof(float);
    batch_scoring_decide_kernel<<<blocks, threads, shared_size>>>(
        d_genes, d_states, d_valid_moves, d_move_counts, d_best_actions,
        N, d_layers, num_layers, gene_size, max_moves);

    cudaMemcpy(h_best_actions, d_best_actions, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_genes);
    cudaFree(d_states);
    cudaFree(d_valid_moves);
    cudaFree(d_move_counts);
    cudaFree(d_best_actions);
    cudaFree(d_layers);
}

} // extern "C"

#endif // HAS_CUDA
