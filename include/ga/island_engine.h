#pragma once

#include "ga_engine.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

namespace ga {

struct IslandConfig {
    size_t num_islands = 4;
    size_t pop_per_island = 100;
    int migration_interval = 20;
    size_t migration_count = 5;
    GAConfig base_config;
};

class IslandGAEngine {
    std::vector<GAEngine> islands_;
    IslandConfig config_;
    std::mt19937 rng_;

public:
    explicit IslandGAEngine(const IslandConfig& cfg)
        : config_(cfg), rng_(std::random_device{}()) {
        islands_.reserve(cfg.num_islands);
        for (size_t i = 0; i < cfg.num_islands; ++i) {
            GAConfig island_cfg = cfg.base_config;
            island_cfg.population_size = cfg.pop_per_island;
            islands_.emplace_back(island_cfg);
        }
    }

    GAResult run(std::function<float(const Genome&)> fitness_fn,
                 std::function<void(int, const GAResult&)> callback = nullptr) {
        GAResult best_result;
        best_result.fitness_history.reserve(config_.base_config.max_generations);
        best_result.avg_history.reserve(config_.base_config.max_generations);
        best_result.diversity_history.reserve(config_.base_config.max_generations);

        auto start = std::chrono::high_resolution_clock::now();

        for (int gen = 0; gen < config_.base_config.max_generations; ++gen) {
            for (auto& island : islands_) {
                island.population().evaluate(fitness_fn);
            }

            GAResult combined;
            float best_fit = -1e30f;
            float total_avg = 0.0f;
            size_t total_pop = 0;
            const Genome* best_genome = nullptr;

            for (auto& island : islands_) {
                const auto& ib = island.population().best();
                if (ib.fitness > best_fit) {
                    best_fit = ib.fitness;
                    best_genome = &ib;
                }
                float iavg = island.population().avg_fitness();
                size_t isize = island.population().size();
                total_avg += iavg * isize;
                total_pop += isize;
            }

            total_avg /= static_cast<float>(total_pop);

            best_result.fitness_history.push_back(best_fit);
            best_result.avg_history.push_back(total_avg);
            best_result.generations_run = gen + 1;
            best_result.best_fitness = best_fit;
            best_result.avg_fitness = total_avg;

            if (gen % config_.migration_interval == 0 && gen > 0) {
                migrate();
            }

            for (auto& island : islands_) {
                island.population().update_stagnation();
            }

            for (auto& island : islands_) {
                island.population().copy_to_buffer();
            }

            for (auto& island : islands_) {
                island.create_next_generation_mutable();
                island.population().swap_buffers();
            }

            if (callback) callback(gen + 1, best_result);
        }

        for (auto& island : islands_) {
            island.population().evaluate(fitness_fn);
            const auto& ib = island.population().best();
            if (ib.fitness > best_result.best_fitness) {
                best_result.best_genome = ib;
                best_result.best_fitness = ib.fitness;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        best_result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return best_result;
    }

    GAResult run_batch(std::function<void(Genome&)> fitness_fn,
                       std::function<void(int, const GAResult&)> callback = nullptr) {
        GAResult best_result;
        best_result.fitness_history.reserve(config_.base_config.max_generations);
        best_result.avg_history.reserve(config_.base_config.max_generations);
        best_result.diversity_history.reserve(config_.base_config.max_generations);

        auto start = std::chrono::high_resolution_clock::now();

        for (int gen = 0; gen < config_.base_config.max_generations; ++gen) {
            for (auto& island : islands_) {
                island.population().evaluate_batch(fitness_fn);
            }

            float best_fit = -1e30f;
            float total_avg = 0.0f;
            size_t total_pop = 0;

            for (auto& island : islands_) {
                const auto& ib = island.population().best();
                if (ib.fitness > best_fit) best_fit = ib.fitness;
                float iavg = island.population().avg_fitness();
                size_t isize = island.population().size();
                total_avg += iavg * isize;
                total_pop += isize;
            }

            total_avg /= static_cast<float>(total_pop);

            best_result.fitness_history.push_back(best_fit);
            best_result.avg_history.push_back(total_avg);
            best_result.generations_run = gen + 1;
            best_result.best_fitness = best_fit;
            best_result.avg_fitness = total_avg;

            if (gen % config_.migration_interval == 0 && gen > 0) {
                migrate();
            }

            for (auto& island : islands_) {
                island.population().copy_to_buffer();
                island.create_next_generation_mutable();
                island.population().swap_buffers();
            }

            if (callback) callback(gen + 1, best_result);
        }

        for (auto& island : islands_) {
            island.population().evaluate_batch(fitness_fn);
            const auto& ib = island.population().best();
            if (ib.fitness > best_result.best_fitness) {
                best_result.best_genome = ib;
                best_result.best_fitness = ib.fitness;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        best_result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return best_result;
    }

    std::vector<GAEngine>& islands() { return islands_; }
    const IslandConfig& config() const { return config_; }

private:
    void migrate() {
        size_t n = islands_.size();
        if (n < 2) return;

        std::vector<std::vector<Genome>> migrants(n);
        for (size_t i = 0; i < n; ++i) {
            islands_[i].population().sort_by_fitness();
            migrants[i].resize(config_.migration_count);
            for (size_t j = 0; j < config_.migration_count; ++j) {
                migrants[i][j] = islands_[i].population().individuals()[j];
            }
        }

        for (size_t i = 0; i < n; ++i) {
            size_t target = (i + 1) % n;
            auto& pop = islands_[target].population();
            pop.sort_by_fitness();
            for (size_t j = 0; j < config_.migration_count; ++j) {
                size_t idx = pop.size() - 1 - j;
                if (idx < pop.size()) {
                    pop.individuals()[idx] = migrants[i][j];
                    pop.individuals()[idx].fitness = 0.0f;
                }
            }
        }
    }
};

} // namespace ga
