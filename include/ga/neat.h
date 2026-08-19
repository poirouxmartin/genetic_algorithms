#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <functional>
#include <map>

namespace neat {

struct Innovation {
    int from_node;
    int to_node;
    int innovation_num;
};

struct ConnectionGene {
    int from_node;
    int to_node;
    float weight;
    bool enabled;
    int innovation;
};

struct NodeGene {
    int id;
    float x;
    float y;
};

struct NEATGenome {
    std::vector<NodeGene> nodes;
    std::vector<ConnectionGene> connections;
    float fitness = 0.0f;
    float adjusted_fitness = 0.0f;
    int species_id = -1;

    float compatibility_distance(const NEATGenome& other, float c1 = 1.0f, float c2 = 1.0f, float c3 = 0.4f) const {
        int max_inn = 0;
        for (auto& c : connections) max_inn = std::max(max_inn, c.innovation);
        for (auto& c : other.connections) max_inn = std::max(max_inn, c.innovation);

        int matching = 0, disjoint = 0, excess = 0;
        float weight_diff = 0.0f;
        int max_conn = std::max(connections.size(), other.connections.size());

        std::vector<bool> in_a(max_inn + 1, false), in_b(max_inn + 1, false);
        for (auto& c : connections) in_a[c.innovation] = true;
        for (auto& c : other.connections) in_b[c.innovation] = true;

        std::vector<const ConnectionGene*> sorted_a = sorted_by_innovation(connections);
        std::vector<const ConnectionGene*> sorted_b = sorted_by_innovation(other.connections);

        size_t ia = 0, ib = 0;
        while (ia < sorted_a.size() || ib < sorted_b.size()) {
            if (ia < sorted_a.size() && ib < sorted_b.size()) {
                if (sorted_a[ia]->innovation == sorted_b[ib]->innovation) {
                    matching++;
                    weight_diff += std::abs(sorted_a[ia]->weight - sorted_b[ib]->weight);
                    ia++; ib++;
                } else if (sorted_a[ia]->innovation < sorted_b[ib]->innovation) {
                    disjoint++;
                    ia++;
                } else {
                    disjoint++;
                    ib++;
                }
            } else if (ia < sorted_a.size()) {
                excess++;
                ia++;
            } else {
                excess++;
                ib++;
            }
        }

        int n = std::max(connections.size(), other.connections.size());
        if (n == 0) n = 1;
        return c1 * excess / n + c2 * disjoint / n + c3 * weight_diff / std::max(matching, 1);
    }

    static std::vector<const ConnectionGene*> sorted_by_innovation(const std::vector<ConnectionGene>& conns) {
        std::vector<const ConnectionGene*> result;
        result.reserve(conns.size());
        for (auto& c : conns) result.push_back(&c);
        std::sort(result.begin(), result.end(),
            [](const ConnectionGene* a, const ConnectionGene* b) { return a->innovation < b->innovation; });
        return result;
    }
};

struct NEATConfig {
    int population_size = 150;
    int max_generations = 300;
    float survival_ratio = 0.2f;
    float crossover_rate = 0.75f;
    float weight_mutation_rate = 0.8f;
    float weight_perturb_rate = 0.9f;
    float weight_replace_rate = 0.1f;
    float weight_max_sigma = 2.5f;
    float add_connection_rate = 0.05f;
    float add_node_rate = 0.03f;
    float disable_rate = 0.05f;
    float compatibility_threshold = 3.0f;
    float c1 = 1.0f;
    float c2 = 1.0f;
    float c3 = 0.4f;
    float interspecies_mating_rate = 0.01f;
    float target_fitness = 1e30f;
};

struct NEATResult {
    NEATGenome best_genome;
    float best_fitness = 0.0f;
    float avg_fitness = 0.0f;
    int generations_run = 0;
    double elapsed_ms = 0.0f;
    int num_species = 0;
    std::vector<float> fitness_history;
    std::vector<float> avg_history;
    std::vector<float> diversity_history;
    std::vector<int> species_counts;
};

class InnovationDB {
    int next_innovation_ = 0;
    int next_node_ = 0;
    std::vector<Innovation> history_;

public:
    InnovationDB(int input_count, int output_count) {
        for (int i = 0; i < input_count; ++i) {
            next_node_++;
        }
        for (int i = 0; i < output_count; ++i) {
            next_node_++;
        }
        for (int i = 0; i < input_count; ++i) {
            for (int o = 0; o < output_count; ++o) {
                history_.push_back({i, input_count + o, next_innovation_++});
            }
        }
    }

    int new_node() { return next_node_++; }

    int get_innovation(int from, int to) {
        for (auto& inv : history_) {
            if (inv.from_node == from && inv.to_node == to) return inv.innovation_num;
        }
        history_.push_back({from, to, next_innovation_++});
        return history_.back().innovation_num;
    }

    int max_innovation() const { return next_innovation_; }
    int max_node() const { return next_node_; }
};

class NEATNetwork {
    std::vector<NodeGene> nodes_;
    std::vector<ConnectionGene> connections_;
    int input_count_;
    int output_count_;

public:
    NEATNetwork(const NEATGenome& genome, int input_count, int output_count)
        : nodes_(genome.nodes), connections_(genome.connections),
          input_count_(input_count), output_count_(output_count) {
        std::sort(nodes_.begin(), nodes_.end(),
            [](const NodeGene& a, const NodeGene& b) { return a.id < b.id; });
    }

    std::vector<float> evaluate(const float* inputs) {
        int total = static_cast<int>(nodes_.size());
        std::vector<float> values(total, 0.0f);
        std::vector<bool> computed(total, false);

        for (int i = 0; i < input_count_ && i < total; ++i) {
            values[i] = inputs[i];
            computed[i] = true;
        }

        std::vector<bool> in_stack(total, false);

        auto compute = [&](auto& self, int node_idx) -> float {
            if (computed[node_idx]) return values[node_idx];
            if (in_stack[node_idx]) return 0.0f;
            in_stack[node_idx] = true;
            float sum = 0.0f;
            for (auto& conn : connections_) {
                if (!conn.enabled) continue;
                int to_idx = node_index(conn.to_node);
                int from_idx = node_index(conn.from_node);
                if (to_idx == node_idx && from_idx >= 0) {
                    sum += conn.weight * self(self, from_idx);
                }
            }
            values[node_idx] = activation(sum);
            computed[node_idx] = true;
            in_stack[node_idx] = false;
            return values[node_idx];
        };

        for (int i = input_count_; i < total; ++i) {
            compute(compute, i);
        }

        std::vector<float> output;
        for (int i = 0; i < output_count_; ++i) {
            int idx = node_index(input_count_ + i);
            output.push_back(idx >= 0 ? values[idx] : 0.0f);
        }
        return output;
    }

    int decide(const float* inputs) {
        auto outputs = evaluate(inputs);
        int best = 0;
        for (size_t i = 1; i < outputs.size(); ++i) {
            if (outputs[i] > outputs[best]) best = static_cast<int>(i);
        }
        return best;
    }

    float score(const float* inputs) {
        auto outputs = evaluate(inputs);
        return outputs.empty() ? 0.0f : outputs[0];
    }

    static float activation(float x) {
        return std::tanh(x);
    }

private:
    int node_index(int id) const {
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i].id == id) return static_cast<int>(i);
        }
        return -1;
    }
};

class Species {
public:
    std::vector<size_t> members;
    float best_fitness = -1e30f;
    int stagnation_ = 0;

    void clear() { members.clear(); }
    void add(size_t idx) { members.push_back(idx); }
    size_t size() const { return members.size(); }
};

class NEATEngine {
    NEATConfig config_;
    InnovationDB innov_db_;
    std::vector<NEATGenome> population_;
    std::vector<Species> species_;
    std::mt19937 rng_;
    int next_species_id_ = 0;
    int input_count_ = 0;
    int output_count_ = 0;

public:
    NEATEngine(int input_count, int output_count, const NEATConfig& cfg = NEATConfig{})
        : config_(cfg), innov_db_(input_count, output_count), rng_(std::random_device{}()),
          input_count_(input_count), output_count_(output_count) {
        population_.resize(cfg.population_size);
        for (auto& g : population_) {
            init_genome(g, input_count, output_count);
        }
    }

    NEATResult run(int input_count, int output_count,
                   std::function<float(NEATNetwork&)> evaluate_fn,
                   std::function<void(int, const NEATResult&)> callback = nullptr) {
        NEATResult result;
        result.fitness_history.reserve(config_.max_generations);
        result.avg_history.reserve(config_.max_generations);
        result.diversity_history.reserve(config_.max_generations);

        auto start = std::chrono::high_resolution_clock::now();

        for (int gen = 0; gen < config_.max_generations; ++gen) {
            evaluate_population(evaluate_fn);

            speciate();

            float best_fit = -1e30f;
            float total_fit = 0.0f;
            size_t best_idx = 0;
            for (size_t i = 0; i < population_.size(); ++i) {
                total_fit += population_[i].fitness;
                if (population_[i].fitness > best_fit) {
                    best_fit = population_[i].fitness;
                    best_idx = i;
                }
            }

            result.fitness_history.push_back(best_fit);
            result.avg_history.push_back(total_fit / population_.size());
            result.generations_run = gen + 1;
            result.best_fitness = best_fit;
            result.avg_fitness = total_fit / population_.size();
            result.num_species = static_cast<int>(species_.size());
            result.species_counts.clear();
            for (auto& sp : species_) result.species_counts.push_back(static_cast<int>(sp.members.size()));

            if (best_fit >= config_.target_fitness) break;

            adjust_fitnesses();
            create_next_generation(input_count, output_count);

            if (callback) callback(gen + 1, result);
        }

        evaluate_population(evaluate_fn);
        result.best_genome = *std::max_element(population_.begin(), population_.end(),
            [](const NEATGenome& a, const NEATGenome& b) { return a.fitness > b.fitness; });
        result.best_fitness = result.best_genome.fitness;

        auto end = std::chrono::high_resolution_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return result;
    }

    const std::vector<NEATGenome>& population() const { return population_; }

private:
    void init_genome(NEATGenome& g, int input_count, int output_count) {
        g.nodes.clear();
        g.connections.clear();

        float layer_in = 0.0f;
        float layer_out = 1.0f;

        for (int i = 0; i < input_count; ++i) {
            g.nodes.push_back({i, layer_in, static_cast<float>(i) / input_count});
        }
        for (int o = 0; o < output_count; ++o) {
            g.nodes.push_back({input_count + o, layer_out, static_cast<float>(o) / output_count});
        }

        std::uniform_real_distribution<float> wd(-1.0f, 1.0f);
        for (int i = 0; i < input_count; ++i) {
            for (int o = 0; o < output_count; ++o) {
                g.connections.push_back({i, input_count + o, wd(rng_), true,
                    innov_db_.get_innovation(i, input_count + o)});
            }
        }
    }

    void evaluate_population(const std::function<float(NEATNetwork&)>& eval_fn) {
        for (auto& g : population_) {
            NEATNetwork net(g, input_count_, output_count_);
            g.fitness = eval_fn(net);
        }
    }

    void speciate() {
        for (auto& sp : species_) sp.clear();

        for (size_t i = 0; i < population_.size(); ++i) {
            bool found = false;
            for (size_t s = 0; s < species_.size(); ++s) {
                if (species_[s].members.empty()) continue;
                float dist = population_[i].compatibility_distance(
                    population_[species_[s].members[0]], config_.c1, config_.c2, config_.c3);
                if (dist < config_.compatibility_threshold) {
                    species_[s].add(static_cast<int>(i));
                    population_[i].species_id = static_cast<int>(s);
                    found = true;
                    break;
                }
            }
            if (!found) {
                Species new_sp;
                new_sp.add(static_cast<int>(i));
                population_[i].species_id = next_species_id_++;
                species_.push_back(std::move(new_sp));
            }
        }

        for (auto& sp : species_) {
            if (sp.members.empty()) continue;
            float best = -1e30f;
            for (size_t idx : sp.members) {
                if (population_[idx].fitness > best) best = population_[idx].fitness;
            }
            if (best > sp.best_fitness) {
                sp.best_fitness = best;
                sp.stagnation_ = 0;
            } else {
                sp.stagnation_++;
            }
        }

        species_.erase(
            std::remove_if(species_.begin(), species_.end(),
                [](const Species& s) { return s.members.empty(); }),
            species_.end());
    }

    void adjust_fitnesses() {
        for (auto& sp : species_) {
            float sp_size = static_cast<float>(sp.members.size());
            for (size_t idx : sp.members) {
                population_[idx].adjusted_fitness = population_[idx].fitness / sp_size;
            }
        }
    }

    void create_next_generation(int input_count, int output_count) {
        std::vector<NEATGenome> new_pop;
        new_pop.reserve(population_.size());

        for (auto& sp : species_) {
            if (sp.members.size() < 2) continue;

            std::sort(sp.members.begin(), sp.members.end(), [&](size_t a, size_t b) {
                return population_[a].adjusted_fitness > population_[b].adjusted_fitness;
            });

            size_t survive = std::max<size_t>(2, static_cast<size_t>(sp.members.size() * config_.survival_ratio));

            std::uniform_real_distribution<float> prob(0.0f, 1.0f);

            for (size_t i = 0; i < survive && new_pop.size() < population_.size(); ++i) {
                NEATGenome child = population_[sp.members[i]];

                if (prob(rng_) < config_.crossover_rate && survive >= 2) {
                    size_t p2_idx = std::uniform_int_distribution<size_t>(0, sp.members.size() - 1)(rng_);
                    child = crossover(population_[sp.members[i]], population_[sp.members[p2_idx]]);
                }

                mutate(child, input_count, output_count);
                child.fitness = 0.0f;
                child.species_id = sp.members[i] < population_.size() ? population_[sp.members[i]].species_id : -1;
                new_pop.push_back(std::move(child));
            }
        }

        while (new_pop.size() < population_.size()) {
            NEATGenome child = population_[std::uniform_int_distribution<size_t>(0, population_.size() - 1)(rng_)];
            mutate(child, input_count, output_count);
            child.fitness = 0.0f;
            new_pop.push_back(std::move(child));
        }

        if (!new_pop.empty()) {
            size_t best_idx = 0;
            for (size_t i = 1; i < population_.size(); ++i) {
                if (population_[i].fitness > population_[best_idx].fitness) best_idx = i;
            }
            new_pop[0] = population_[best_idx];
        }

        population_ = std::move(new_pop);
    }

    NEATGenome crossover(const NEATGenome& p1, const NEATGenome& p2) {
        const NEATGenome& fit_parent = p1.fitness >= p2.fitness ? p1 : p2;
        const NEATGenome& other = p1.fitness >= p2.fitness ? p2 : p1;

        NEATGenome child;
        child.nodes = fit_parent.nodes;

        std::map<int, const ConnectionGene*> other_map;
        for (auto& c : other.connections) other_map[c.innovation] = &c;

        for (auto& c : fit_parent.connections) {
            auto it = other_map.find(c.innovation);
            if (it != other_map.end()) {
                std::uniform_real_distribution<float> prob(0.0f, 1.0f);
                child.connections.push_back(prob(rng_) < 0.5f ? c : *it->second);
            } else {
                child.connections.push_back(c);
            }
        }

        return child;
    }

    void mutate(NEATGenome& g, int input_count, int output_count) {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        if (prob(rng_) < config_.weight_mutation_rate) {
            for (auto& c : g.connections) {
                if (prob(rng_) < config_.weight_perturb_rate) {
                    c.weight += std::normal_distribution<float>(0.0f, config_.weight_max_sigma)(rng_);
                    c.weight = std::clamp(c.weight, -30.0f, 30.0f);
                } else {
                    c.weight = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng_);
                }
            }
        }

        for (auto& c : g.connections) {
            if (prob(rng_) < config_.disable_rate) {
                c.enabled = !c.enabled;
            }
        }

        if (prob(rng_) < config_.add_connection_rate) {
            add_random_connection(g);
        }

        if (prob(rng_) < config_.add_node_rate) {
            add_random_node(g);
        }
    }

    void add_random_connection(NEATGenome& g) {
        for (int attempt = 0; attempt < 20; ++attempt) {
            if (g.nodes.size() < 2) return;
            size_t a = std::uniform_int_distribution<size_t>(0, g.nodes.size() - 1)(rng_);
            size_t b = std::uniform_int_distribution<size_t>(0, g.nodes.size() - 1)(rng_);
            if (a == b) continue;

            int from = g.nodes[a].id;
            int to = g.nodes[b].id;

            bool exists = false;
            for (auto& c : g.connections) {
                if (c.from_node == from && c.to_node == to) { exists = true; break; }
            }
            if (exists) continue;

            g.connections.push_back({from, to,
                std::normal_distribution<float>(0.0f, 1.0f)(rng_),
                true, innov_db_.get_innovation(from, to)});
            return;
        }
    }

    void add_random_node(NEATGenome& g) {
        if (g.connections.empty()) return;

        size_t idx = std::uniform_int_distribution<size_t>(0, g.connections.size() - 1)(rng_);
        ConnectionGene& old_conn = g.connections[idx];
        if (!old_conn.enabled) return;

        int new_id = innov_db_.new_node();
        float x = 0.5f * (node_x(g, old_conn.from_node) + node_x(g, old_conn.to_node));
        float y = 0.5f * (node_y(g, old_conn.from_node) + node_y(g, old_conn.to_node));
        g.nodes.push_back({new_id, x, y});

        old_conn.enabled = false;
        g.connections.push_back({old_conn.from_node, new_id, 1.0f, true,
            innov_db_.get_innovation(old_conn.from_node, new_id)});
        g.connections.push_back({new_id, old_conn.to_node, old_conn.weight, true,
            innov_db_.get_innovation(new_id, old_conn.to_node)});
    }

    float node_x(const NEATGenome& g, int id) const {
        for (auto& n : g.nodes) if (n.id == id) return n.x;
        return 0.5f;
    }

    float node_y(const NEATGenome& g, int id) const {
        for (auto& n : g.nodes) if (n.id == id) return n.y;
        return 0.5f;
    }
};

} // namespace neat
