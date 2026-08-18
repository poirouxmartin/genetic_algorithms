#pragma once

#include "genome.h"
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cassert>

namespace ga {

class SelectionStrategy {
public:
    virtual ~SelectionStrategy() = default;
    virtual size_t select(const std::vector<Genome>& pop, std::mt19937& rng) = 0;
};

class TournamentSelection : public SelectionStrategy {
    size_t tournament_size_;
public:
    explicit TournamentSelection(size_t k = 3) : tournament_size_(k) {}

    size_t select(const std::vector<Genome>& pop, std::mt19937& rng) override {
        std::uniform_int_distribution<size_t> dist(0, pop.size() - 1);
        size_t best_idx = dist(rng);
        float best_fit = pop[best_idx].fitness;
        for (size_t i = 1; i < tournament_size_; ++i) {
            size_t idx = dist(rng);
            if (pop[idx].fitness > best_fit) {
                best_fit = pop[idx].fitness;
                best_idx = idx;
            }
        }
        return best_idx;
    }
};

class RouletteSelection : public SelectionStrategy {
public:
    size_t select(const std::vector<Genome>& pop, std::mt19937& rng) override {
        float total = 0.0f;
        for (const auto& g : pop) total += g.fitness;
        if (total <= 0.0f) {
            std::uniform_int_distribution<size_t> dist(0, pop.size() - 1);
            return dist(rng);
        }
        std::uniform_real_distribution<float> dist(0.0f, total);
        float r = dist(rng);
        float cumulative = 0.0f;
        for (size_t i = 0; i < pop.size(); ++i) {
            cumulative += pop[i].fitness;
            if (cumulative >= r) return i;
        }
        return pop.size() - 1;
    }
};

class RankSelection : public SelectionStrategy {
public:
    size_t select(const std::vector<Genome>& pop, std::mt19937& rng) override {
        size_t n = pop.size();
        float total = static_cast<float>(n * (n + 1)) / 2.0f;
        std::uniform_real_distribution<float> dist(0.0f, total);
        float r = dist(rng);
        float cumulative = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            cumulative += static_cast<float>(n - i);
            if (cumulative >= r) return i;
        }
        return n - 1;
    }
};

class EliteSelection : public SelectionStrategy {
    size_t elite_count_;
public:
    explicit EliteSelection(size_t elite_count = 1) : elite_count_(elite_count) {}

    size_t select(const std::vector<Genome>& pop, std::mt19937& rng) override {
        std::uniform_int_distribution<size_t> dist(0, elite_count_ - 1);
        return dist(rng);
    }
};

} // namespace ga
