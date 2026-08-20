#pragma once

#include "genome.h"
#include "population.h"
#include "selection.h"
#include "crossover.h"
#include "mutation.h"
#include <thread>
#include <functional>
#include <chrono>
#include <vector>
#include <cmath>

namespace ga {

struct GAConfig {
    size_t population_size = 500;
    size_t gene_count = 64;
    float min_gene = -1.0f;
    float max_gene = 1.0f;
    float crossover_rate = 0.85f;
    float mutation_rate = 0.1f;
    float mutation_sigma = 0.2f;
    float elitism_ratio = 0.05f;
    size_t tournament_size = 3;
    int max_generations = 1000;
    bool adaptive_mutation = true;
    bool parallel_eval = true;
    std::vector<std::pair<int,int>> symmetry_pairs;
    float l2_lambda = 0.0f; // L2 weight penalty: fitness -= l2_lambda * sum(w^2)
    int input_size = 0;     // for He init: sigma = sqrt(2/input_size) per layer
    int hidden_size = 0;
};

struct GAResult {
    Genome best_genome;
    float best_fitness = 0.0f;
    float avg_fitness = 0.0f;
    int generations_run = 0;
    double elapsed_ms = 0.0f;
    std::vector<float> fitness_history;
    std::vector<float> avg_history;
    std::vector<float> diversity_history;
};

class GAEngine {
    Population pop_;
    GAConfig config_;
    std::unique_ptr<SelectionStrategy> selection_;
    std::unique_ptr<CrossoverStrategy> crossover_;
    std::unique_ptr<MutationStrategy> mutation_;
    std::mt19937 rng_;
    bool stop_flag_ = false;
    std::chrono::high_resolution_clock::time_point start_time_;

public:
    explicit GAEngine(const GAConfig& cfg = GAConfig{})
        : config_(cfg), rng_(std::random_device{}()),
          start_time_(std::chrono::high_resolution_clock::now()) {

        pop_ = Population(cfg.population_size, cfg.gene_count, cfg.min_gene, cfg.max_gene);

        if (cfg.input_size > 0 && cfg.hidden_size > 0) {
            he_init(cfg.input_size, cfg.hidden_size);
        }

        selection_ = std::make_unique<TournamentSelection>(cfg.tournament_size);
        crossover_ = std::make_unique<UniformCrossover>(cfg.crossover_rate);

        if (cfg.adaptive_mutation) {
            mutation_ = std::make_unique<AdaptiveMutation>(cfg.mutation_rate, cfg.mutation_sigma);
        } else {
            mutation_ = std::make_unique<GaussianMutation>(cfg.mutation_rate, cfg.mutation_sigma);
        }
    }

    GAEngine(GAEngine&&) = default;
    GAEngine& operator=(GAEngine&&) = default;
    GAEngine(const GAEngine&) = delete;
    GAEngine& operator=(const GAEngine&) = delete;

    void set_selection(std::unique_ptr<SelectionStrategy> s) { selection_ = std::move(s); }
    void set_crossover(std::unique_ptr<CrossoverStrategy> c) { crossover_ = std::move(c); }
    void set_mutation(std::unique_ptr<MutationStrategy> m) { mutation_ = std::move(m); }

    Population& population() { return pop_; }
    const Population& population() const { return pop_; }
    const GAConfig& config() const { return config_; }

    void request_stop() { stop_flag_ = true; }

    void create_next_generation_mutable() { create_next_generation(); }

    // Non-blocking single generation step. Returns true if a generation was run.
    bool step(std::function<void(Genome&)> fitness_fn, GAResult& result) {
        if (result.generations_run >= config_.max_generations) return false;

        auto gen = result.generations_run;
        pop_.evaluate_batch(fitness_fn);

        float best_fit = -1e30f;
        size_t best_idx = 0;
        float sum = 0.0f;
        for (size_t i = 0; i < pop_.individuals_.size(); ++i) {
            float f = pop_.individuals_[i].fitness;
            sum += f;
            if (f > best_fit) { best_fit = f; best_idx = i; }
        }
        float avg = sum / static_cast<float>(pop_.individuals_.size());
        float div = pop_.diversity();

        if (best_fit > result.best_fitness || result.generations_run == 0) {
            result.best_genome = pop_.individuals_[best_idx];
            result.best_fitness = best_fit;
        }

        result.fitness_history.push_back(best_fit);
        result.avg_history.push_back(avg);
        result.diversity_history.push_back(div);
        result.generations_run = gen + 1;
        result.avg_fitness = avg;

        pop_.prev_best_ = best_fit;
        if (best_fit <= pop_.best_fitness_ + 1e-6f) {
            pop_.stagnation_++;
            if (pop_.stagnation_ > 50 && config_.adaptive_mutation) {
                auto* adapt = dynamic_cast<AdaptiveMutation*>(mutation_.get());
                if (adapt) adapt->decay();
                inject_random(0.02f);
            }
        } else {
            pop_.stagnation_ = 0;
            pop_.best_fitness_ = best_fit;
        }

        pop_.copy_to_buffer();
        create_next_generation();
        pop_.swap_buffers();

        auto now = std::chrono::high_resolution_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time_).count();

        return true;
    }

    GAResult run(std::function<float(const Genome&)> fitness_fn, std::function<void(int, const GAResult&)> callback = nullptr) {
        GAResult result;
        result.fitness_history.reserve(config_.max_generations);
        result.avg_history.reserve(config_.max_generations);
        result.diversity_history.reserve(config_.max_generations);

        auto start = std::chrono::high_resolution_clock::now();
        stop_flag_ = false;

        for (int gen = 0; gen < config_.max_generations && !stop_flag_; ++gen) {
            pop_.evaluate(fitness_fn);

            const auto& best = pop_.best();
            float avg = pop_.avg_fitness();
            float div = pop_.diversity();

            result.fitness_history.push_back(best.fitness);
            result.avg_history.push_back(avg);
            result.diversity_history.push_back(div);
            result.generations_run = gen + 1;
            result.best_fitness = best.fitness;
            result.avg_fitness = avg;

            pop_.update_stagnation();
            if (pop_.stagnation() > 50 && config_.adaptive_mutation) {
                auto* adapt = dynamic_cast<AdaptiveMutation*>(mutation_.get());
                if (adapt) adapt->decay();
                inject_random(0.02f);
            }

            pop_.copy_to_buffer();
            create_next_generation();
            pop_.swap_buffers();

            if (callback) callback(gen + 1, result);
        }

        pop_.evaluate(fitness_fn);
        result.best_genome = pop_.best();
        result.best_fitness = result.best_genome.fitness;

        auto end = std::chrono::high_resolution_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    GAResult run_batch(std::function<void(Genome&)> fitness_fn, std::function<void(int, const GAResult&)> callback = nullptr) {
        GAResult result;
        result.fitness_history.reserve(config_.max_generations);
        result.avg_history.reserve(config_.max_generations);
        result.diversity_history.reserve(config_.max_generations);

        auto start = std::chrono::high_resolution_clock::now();
        stop_flag_ = false;

        for (int gen = 0; gen < config_.max_generations && !stop_flag_; ++gen) {
            pop_.evaluate_batch(fitness_fn);

            float best_fit = -1e30f;
            size_t best_idx = 0;
            float sum = 0.0f;
            for (size_t i = 0; i < pop_.individuals_.size(); ++i) {
                float f = pop_.individuals_[i].fitness;
                sum += f;
                if (f > best_fit) { best_fit = f; best_idx = i; }
            }
            float avg = sum / static_cast<float>(pop_.individuals_.size());
            float div = pop_.diversity();

            result.fitness_history.push_back(best_fit);
            result.avg_history.push_back(avg);
            result.diversity_history.push_back(div);
            result.generations_run = gen + 1;
            result.best_fitness = best_fit;
            result.avg_fitness = avg;

            pop_.prev_best_ = best_fit;
            if (best_fit <= pop_.best_fitness_ + 1e-6f) {
                pop_.stagnation_++;
                if (pop_.stagnation_ > 50 && config_.adaptive_mutation) {
                    auto* adapt = dynamic_cast<AdaptiveMutation*>(mutation_.get());
                    if (adapt) adapt->decay();
                    inject_random(0.02f);
                }
            } else {
                pop_.stagnation_ = 0;
                pop_.best_fitness_ = best_fit;
            }

            pop_.copy_to_buffer();
            create_next_generation();
            pop_.swap_buffers();

            auto now = std::chrono::high_resolution_clock::now();
            result.elapsed_ms = std::chrono::duration<double, std::milli>(now - start).count();
            if (callback) callback(gen + 1, result);
        }

        pop_.evaluate_batch(fitness_fn);
        result.best_genome = pop_.best();
        result.best_fitness = result.best_genome.fitness;

        auto end = std::chrono::high_resolution_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

private:
    void create_next_generation() {
        size_t pop_size = pop_.size();
        size_t elite_count = std::max<size_t>(1, static_cast<size_t>(pop_size * config_.elitism_ratio));

        pop_.sort_by_fitness();
        for (size_t i = 0; i < elite_count; ++i) {
            pop_.buffer_[i] = pop_.individuals_[i];
        }

        for (size_t i = elite_count; i < pop_size; i += 2) {
            size_t p1 = selection_->select(pop_.individuals_, rng_);
            size_t p2 = selection_->select(pop_.individuals_, rng_);

            pop_.buffer_[i].fitness = 0.0f;
            pop_.buffer_[i + 1 < pop_size ? i + 1 : i].fitness = 0.0f;

            crossover_->crossover(pop_.individuals_[p1], pop_.individuals_[p2],
                                   pop_.buffer_[i], pop_.buffer_[i + 1 < pop_size ? i + 1 : i], rng_);

            mutation_->mutate(pop_.buffer_[i], rng_);
            if (i + 1 < pop_size) mutation_->mutate(pop_.buffer_[i + 1], rng_);

            if (!config_.symmetry_pairs.empty()) {
                repair_symmetry(pop_.buffer_[i]);
                if (i + 1 < pop_size) repair_symmetry(pop_.buffer_[i + 1]);
            }
        }
    }

    void repair_symmetry(Genome& g) const {
        for (auto [a, b] : config_.symmetry_pairs) {
            if (a < static_cast<int>(g.genes.size()) && b < static_cast<int>(g.genes.size()))
                g.genes[b] = g.genes[a];
        }
    }

    void inject_random(float ratio) {
        size_t count = static_cast<size_t>(pop_.size() * ratio);
        for (size_t i = 0; i < count; ++i) {
            size_t idx = std::uniform_int_distribution<size_t>(0, pop_.size() - 1)(rng_);
            pop_.buffer_[idx].randomize(config_.min_gene, config_.max_gene);
            pop_.buffer_[idx].fitness = 0.0f;
            if (!config_.symmetry_pairs.empty()) repair_symmetry(pop_.buffer_[idx]);
        }
    }

    void he_init(int input_size, int hidden_size) {
        std::normal_distribution<float> hid_dist(0.0f, std::sqrt(2.0f / static_cast<float>(input_size)));
        std::normal_distribution<float> out_dist(0.0f, std::sqrt(2.0f / static_cast<float>(hidden_size)));
        for (auto& g : pop_.individuals_) {
            for (int h = 0; h < hidden_size; ++h)
                for (int i = 0; i < input_size; ++i)
                    g.genes[h * input_size + i] = hid_dist(rng_);
            int ho = input_size * hidden_size + hidden_size;
            int out_size = static_cast<int>((g.genes.size() - ho - hidden_size) / hidden_size);
            if (out_size < 1) out_size = 1;
            for (int o = 0; o < out_size; ++o)
                for (int h = 0; h < hidden_size; ++h)
                    g.genes[ho + o * hidden_size + h] = out_dist(rng_);
        }
    }
};

} // namespace ga
