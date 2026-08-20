#pragma once

#include "genome.h"
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <execution>
#include <cassert>
#include <functional>

namespace ga {

class Population {
public:
    std::vector<Genome> individuals_;
    std::vector<Genome> buffer_;
    size_t size_ = 0;
    size_t gene_count_ = 0;
    float best_fitness_ = -1e30f;
    size_t stagnation_ = 0;
    float prev_best_ = -1e30f;

    Population() = default;

    Population(size_t pop_size, size_t genes) : size_(pop_size), gene_count_(genes) {
        individuals_.resize(pop_size);
        buffer_.resize(pop_size);
        for (auto& g : individuals_) {
            g.genes.resize(genes, 0.0f);
            g.gene_count = genes;
            g.randomize();
        }
    }

    Population(size_t pop_size, size_t genes, float min_val, float max_val) : size_(pop_size), gene_count_(genes) {
        individuals_.resize(pop_size);
        buffer_.resize(pop_size);
        for (auto& g : individuals_) {
            g.genes.resize(genes, 0.0f);
            g.gene_count = genes;
            g.randomize(min_val, max_val);
        }
    }

    void evaluate(std::function<float(const Genome&)> fitness_fn) {
        std::for_each(std::execution::par_unseq, individuals_.begin(), individuals_.end(),
            [&fitness_fn](Genome& g) {
                g.fitness = fitness_fn(g);
            });
    }

    void evaluate_batch(std::function<void(Genome&)> fitness_fn) {
        std::for_each(std::execution::par_unseq, individuals_.begin(), individuals_.end(), fitness_fn);
    }

    Genome& best() {
        return *std::max_element(individuals_.begin(), individuals_.end(),
            [](const Genome& a, const Genome& b) { return a.fitness < b.fitness; });
    }

    const Genome& best() const {
        return *std::max_element(individuals_.begin(), individuals_.end(),
            [](const Genome& a, const Genome& b) { return a.fitness < b.fitness; });
    }

    float avg_fitness() const {
        float sum = 0.0f;
        for (const auto& g : individuals_) sum += g.fitness;
        return sum / static_cast<float>(individuals_.size());
    }

    float worst_fitness() const {
        return std::min_element(individuals_.begin(), individuals_.end(),
            [](const Genome& a, const Genome& b) { return a.fitness < b.fitness; })->fitness;
    }

    float diversity() const {
        if (individuals_.size() < 2) return 0.0f;
        float mean = avg_fitness();
        float variance = 0.0f;
        for (const auto& g : individuals_) {
            float d = g.fitness - mean;
            variance += d * d;
        }
        return std::sqrt(variance / individuals_.size());
    }

    void sort_by_fitness() {
        std::sort(individuals_.begin(), individuals_.end(),
            [](const Genome& a, const Genome& b) { return a.fitness > b.fitness; });
    }

    void swap_buffers() {
        individuals_.swap(buffer_);
    }

    void copy_to_buffer() {
        buffer_ = individuals_;
    }

    void update_stagnation() {
        float current_best = best().fitness;
        if (current_best <= prev_best_ + 1e-6f) {
            stagnation_++;
        } else {
            stagnation_ = 0;
        }
        prev_best_ = current_best;
        best_fitness_ = current_best;
    }

    size_t stagnation() const { return stagnation_; }

    std::vector<Genome>& individuals() { return individuals_; }
    const std::vector<Genome>& individuals() const { return individuals_; }
    size_t size() const { return size_; }
};

} // namespace ga
