#pragma once

#include "genome.h"
#include <random>
#include <algorithm>

namespace ga {

class MutationStrategy {
public:
    virtual ~MutationStrategy() = default;
    virtual void mutate(Genome& g, std::mt19937& rng) = 0;
};

class GaussianMutation : public MutationStrategy {
    float rate_;
    float sigma_;
public:
    explicit GaussianMutation(float rate = 0.1f, float sigma = 0.2f) : rate_(rate), sigma_(sigma) {}

    void mutate(Genome& g, std::mt19937& rng) override {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        std::normal_distribution<float> noise(0.0f, sigma_);
        for (size_t i = 0; i < g.gene_count; ++i) {
            if (prob(rng) < rate_) {
                g.genes[i] += noise(rng);
                g.genes[i] = std::clamp(g.genes[i], -10.0f, 10.0f);
            }
        }
    }
};

class UniformMutation : public MutationStrategy {
    float rate_;
    float range_;
public:
    explicit UniformMutation(float rate = 0.1f, float range = 0.5f) : rate_(rate), range_(range) {}

    void mutate(Genome& g, std::mt19937& rng) override {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        std::uniform_real_distribution<float> noise(-range_, range_);
        for (size_t i = 0; i < g.gene_count; ++i) {
            if (prob(rng) < rate_) {
                g.genes[i] += noise(rng);
                g.genes[i] = std::clamp(g.genes[i], -10.0f, 10.0f);
            }
        }
    }
};

class BitFlipMutation : public MutationStrategy {
    float rate_;
public:
    explicit BitFlipMutation(float rate = 0.05f) : rate_(rate) {}

    void mutate(Genome& g, std::mt19937& rng) override {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        for (size_t i = 0; i < g.gene_count; ++i) {
            if (prob(rng) < rate_) {
                g.genes[i] = -g.genes[i];
            }
        }
    }
};

class AdaptiveMutation : public MutationStrategy {
    float base_rate_;
    float base_sigma_;
    float mutation_rate_decay_ = 0.995f;
    float current_rate_;
    float current_sigma_;
public:
    AdaptiveMutation(float rate = 0.15f, float sigma = 0.3f)
        : base_rate_(rate), base_sigma_(sigma), current_rate_(rate), current_sigma_(sigma) {}

    void mutate(Genome& g, std::mt19937& rng) override {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        std::uniform_real_distribution<float> noise(-current_sigma_ * 4.0f, current_sigma_ * 4.0f);
        for (size_t i = 0; i < g.gene_count; ++i) {
            if (prob(rng) < current_rate_) {
                g.genes[i] += noise(rng);
                if (g.genes[i] > 10.0f) g.genes[i] = 10.0f;
                else if (g.genes[i] < -10.0f) g.genes[i] = -10.0f;
            }
        }
    }

    void decay() {
        current_rate_ *= mutation_rate_decay_;
        current_sigma_ *= mutation_rate_decay_;
        current_rate_ = std::max(current_rate_, 0.001f);
        current_sigma_ = std::max(current_sigma_, 0.01f);
    }

    void reset() { current_rate_ = base_rate_; current_sigma_ = base_sigma_; }
};

} // namespace ga
