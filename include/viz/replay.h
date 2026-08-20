#pragma once

#include "../ga/ga_engine.h"
#include "../games/game_interface.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace viz {

struct ReplayFrame {
    std::string state;
    int action;
    float fitness;
    int step;
};

struct ReplayData {
    std::string game_name;
    float final_fitness;
    int total_steps;
    std::vector<ReplayFrame> frames;
};

inline ReplayData capture_replay(
    const ga::Genome& genome,
    const std::function<std::unique_ptr<games::GameInterface>()>& factory,
    const std::string& game_name,
    size_t input_size, const std::vector<int>& hidden_layers,
    int max_steps)
{
    ReplayData replay;
    replay.game_name = game_name;
    replay.final_fitness = 0;

    auto game = factory();
    game->reset();
    bool scoring = game->uses_scoring();
    std::vector<float> state(input_size);
    int steps = 0;

    if (scoring) {
        games::ScoringNetwork sn(input_size, hidden_layers);
        while (!game->is_done() && steps < max_steps) {
            ReplayFrame frame;
            game->get_state(state.data());
            frame.state = game->render_state();
            frame.step = steps;

            auto valid = game->get_valid_moves();
            if (valid.empty()) break;

            int best_action = valid[0];
            float best_score = -1e30f;
            for (int a : valid) {
                auto sim = game->clone();
                sim->step(a);
                std::vector<float> ns(input_size);
                sim->get_state(ns.data());
                float s = sn.score(genome.genes.data(), ns.data());
                if (s > best_score) { best_score = s; best_action = a; }
            }

            frame.action = best_action;
            game->step(best_action);
            frame.fitness = game->fitness();
            replay.frames.push_back(std::move(frame));
            steps++;
        }
    } else {
        games::PolicyNetwork pn(input_size, hidden_layers, game->action_space());
        while (!game->is_done() && steps < max_steps) {
            ReplayFrame frame;
            game->get_state(state.data());
            frame.state = game->render_state();
            frame.step = steps;

            int action = pn.decide(genome.genes.data(), state.data());
            frame.action = action;

            game->step(action);
            frame.fitness = game->fitness();
            replay.frames.push_back(std::move(frame));
            steps++;
        }
    }

    replay.final_fitness = game->fitness();
    replay.total_steps = steps;
    return replay;
}

} // namespace viz
