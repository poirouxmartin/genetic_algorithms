#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace ga {

struct Genome {
    std::vector<float> genes;
    size_t gene_count = 0;
    float fitness = 0.0f;

    Genome() = default;

    explicit Genome(size_t count) : genes(count, 0.0f), gene_count(count) {
        randomize();
    }

    Genome(size_t count, float min_val, float max_val) : genes(count, 0.0f), gene_count(count) {
        randomize(min_val, max_val);
    }

    void randomize() {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < gene_count; ++i) {
            genes[i] = dist(rng);
        }
    }

    void randomize(float min_val, float max_val) {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < gene_count; ++i) {
            genes[i] = dist(rng);
        }
    }

    float& operator[](size_t i) { return genes[i]; }
    float operator[](size_t i) const { return genes[i]; }

    bool operator>(const Genome& o) const { return fitness > o.fitness; }
    bool operator<(const Genome& o) const { return fitness < o.fitness; }
};

} // namespace ga
