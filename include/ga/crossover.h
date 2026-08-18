#pragma once

#include "genome.h"
#include <random>
#include <algorithm>

namespace ga {

class CrossoverStrategy {
public:
    virtual ~CrossoverStrategy() = default;
    virtual void crossover(const Genome& a, const Genome& b, Genome& child1, Genome& child2, std::mt19937& rng) = 0;
};

class SinglePointCrossover : public CrossoverStrategy {
    float rate_;
public:
    explicit SinglePointCrossover(float rate = 0.8f) : rate_(rate) {}

    void crossover(const Genome& a, const Genome& b, Genome& child1, Genome& child2, std::mt19937& rng) override {
        child1.gene_count = a.gene_count;
        child2.gene_count = a.gene_count;
        child1.fitness = 0.0f;
        child2.fitness = 0.0f;

        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        if (prob(rng) > rate_ || a.gene_count <= 1) {
            std::copy(a.genes.begin(), a.genes.begin() + a.gene_count, child1.genes.begin());
            std::copy(b.genes.begin(), b.genes.begin() + b.gene_count, child2.genes.begin());
            return;
        }

        std::uniform_int_distribution<size_t> dist(1, a.gene_count - 1);
        size_t point = dist(rng);

        std::copy(a.genes.begin(), a.genes.begin() + point, child1.genes.begin());
        std::copy(b.genes.begin() + point, b.genes.begin() + a.gene_count, child1.genes.begin() + point);

        std::copy(b.genes.begin(), b.genes.begin() + point, child2.genes.begin());
        std::copy(a.genes.begin() + point, a.genes.begin() + a.gene_count, child2.genes.begin() + point);
    }
};

class UniformCrossover : public CrossoverStrategy {
    float rate_;
public:
    explicit UniformCrossover(float swap_prob = 0.5f) : rate_(swap_prob) {}

    void crossover(const Genome& a, const Genome& b, Genome& child1, Genome& child2, std::mt19937& rng) override {
        child1.gene_count = a.gene_count;
        child2.gene_count = a.gene_count;
        child1.fitness = 0.0f;
        child2.fitness = 0.0f;

        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        for (size_t i = 0; i < a.gene_count; ++i) {
            if (prob(rng) < rate_) {
                child1.genes[i] = b.genes[i];
                child2.genes[i] = a.genes[i];
            } else {
                child1.genes[i] = a.genes[i];
                child2.genes[i] = b.genes[i];
            }
        }
    }
};

class BlendCrossover : public CrossoverStrategy {
    float alpha_;
public:
    explicit BlendCrossover(float alpha = 0.5f) : alpha_(alpha) {}

    void crossover(const Genome& a, const Genome& b, Genome& child1, Genome& child2, std::mt19937& rng) override {
        child1.gene_count = a.gene_count;
        child2.gene_count = a.gene_count;
        child1.fitness = 0.0f;
        child2.fitness = 0.0f;

        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        for (size_t i = 0; i < a.gene_count; ++i) {
            float lo = std::min(a.genes[i], b.genes[i]);
            float hi = std::max(a.genes[i], b.genes[i]);
            float range = hi - lo;
            float t1 = lo - alpha_ * range;
            float t2 = hi + alpha_ * range;
            child1.genes[i] = std::clamp(std::uniform_real_distribution<float>(t1, t2)(rng), -10.0f, 10.0f);
            child2.genes[i] = std::clamp(std::uniform_real_distribution<float>(t1, t2)(rng), -10.0f, 10.0f);
        }
    }
};

} // namespace ga
