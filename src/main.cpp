#include "ga/ga_engine.h"
#include "games/snake.h"
#include "games/tic_tac_toe.h"
#include "games/flappy_bird.h"
#include "games/pong.h"
#include "games/mastermind.h"
#include "games/maze.h"
#include "games/othello.h"
#include "games/chess.h"
#include "viz/tui.h"
#include "viz/gui.h"
#include "viz/replay.h"
#ifdef HAS_SDL2
#include <SDL.h>
#endif

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <iomanip>

struct GameConfig {
    std::string name;
    size_t input_size;
    size_t hidden_size;
    int max_steps;
    std::function<std::unique_ptr<games::GameInterface>()> factory;
};

std::vector<GameConfig> get_game_configs() {
    return {
        {"Snake",       8,  16, 200,  []{ return std::make_unique<games::Snake>(); }},
        {"TicTacToe",   9,  12, 9,    []{ return std::make_unique<games::TicTacToe>(); }},
        {"FlappyBird",  6,  10, 500,  []{ return std::make_unique<games::FlappyBird>(); }},
        {"Pong",        8,  12, 500,  []{ return std::make_unique<games::Pong>(); }},
        {"Mastermind",  4,  8,  40,   []{ return std::make_unique<games::Mastermind>(); }},
        {"Maze",        12, 12, 200,  []{ return std::make_unique<games::Maze>(); }},
        {"Othello",     64, 16, 60,   []{ return std::make_unique<games::Othello>(); }},
        {"Chess",       25, 16, 100,  []{ return std::make_unique<games::SimpleChess>(); }},
    };
}

ga::GAConfig make_config(size_t genome_size, size_t pop = 300, int gens = 300) {
    ga::GAConfig cfg;
    cfg.population_size = pop;
    cfg.gene_count = genome_size;
    cfg.min_gene = -1.0f;
    cfg.max_gene = 1.0f;
    cfg.crossover_rate = 0.85f;
    cfg.mutation_rate = 0.1f;
    cfg.mutation_sigma = 0.2f;
    cfg.elitism_ratio = 0.05f;
    cfg.tournament_size = 3;
    cfg.max_generations = gens;
    cfg.adaptive_mutation = true;
    cfg.parallel_eval = true;
    return cfg;
}

ga::GAResult run_game(const GameConfig& gc, bool show_tui) {
    auto game = gc.factory();
    bool scoring = game->uses_scoring();

    size_t genome_size;
    if (scoring) {
        genome_size = games::ScoringNetwork::weight_count(gc.input_size, gc.hidden_size);
    } else {
        genome_size = games::PolicyNetwork::weight_count(gc.input_size, gc.hidden_size, game->action_space());
    }

    auto cfg = make_config(genome_size, 300, 300);

    viz::TUI tui(80, 30);
    tui.set_enabled(show_tui);

    ga::GAEngine engine(cfg);

    std::cout << "Running " << gc.name << " (genome=" << genome_size
              << (scoring ? " scoring" : " policy") << ")..." << std::endl;

    auto result = engine.run_batch([&](ga::Genome& g) {
        auto tls_game = gc.factory();
        tls_game->reset();
        std::vector<float> state(gc.input_size);
        int steps = 0;

        if (scoring) {
            games::ScoringNetwork sn(gc.input_size, gc.hidden_size);
            while (!tls_game->is_done() && steps < gc.max_steps) {
                auto valid = tls_game->get_valid_moves();
                if (valid.empty()) break;

                tls_game->get_state(state.data());
                int best_action = valid[0];
                float best_score = -1e30f;
                for (int a : valid) {
                    auto sim = tls_game->clone();
                    sim->step(a);
                    std::vector<float> new_state(gc.input_size);
                    sim->get_state(new_state.data());
                    float s = sn.score(g.genes.data(), new_state.data());
                    if (s > best_score) { best_score = s; best_action = a; }
                }
                tls_game->step(best_action);
                steps++;
            }
        } else {
            games::PolicyNetwork pn(gc.input_size, gc.hidden_size, tls_game->action_space());
            while (!tls_game->is_done() && steps < gc.max_steps) {
                tls_game->get_state(state.data());
                int action = pn.decide(g.genes.data(), state.data());
                tls_game->step(action);
                steps++;
            }
        }
        g.fitness = tls_game->fitness();
    });

    std::cout << "  Best: " << result.best_fitness
              << "  Time: " << std::fixed << std::setprecision(1) << result.elapsed_ms << " ms"
              << "  Gen/s: " << std::setprecision(0) << (result.generations_run / (result.elapsed_ms / 1000.0 + 1e-6))
              << "\n" << std::flush;

    return result;
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "   _____     _             __  __           _       _   \n";
    std::cout << "  / ____|   | |           |  \\/  |         | |     | |  \n";
    std::cout << " | (___   __| | ___  ___  | \\  / | __ _ _ __| | __ _| |_ \n";
    std::cout << "  \\___ \\ / _` |/ _ \\/ __| | |\\/| |/ _` | '__| |/ _` | __|\n";
    std::cout << "  ____) | (_| |  __/ (__  | |  | | (_| | |  | | (_| | |_ \n";
    std::cout << " |_____/ \\__,_|\\___|\\___| |_|  |_|\\__,_|_|  |_|\\__,_|\\__|\n";
    std::cout << "\n";

    std::string mode = "all";
    if (argc > 1) mode = argv[1];

    auto configs = get_game_configs();

    if (mode == "bench" || mode == "benchmark") {
        std::vector<std::pair<std::string, ga::GAResult>> results;

        for (const auto& gc : configs) {
            auto result = run_game(gc, false);
            results.emplace_back(gc.name, std::move(result));
        }

        std::cout << "\n";
        std::cout << " ================================================================\n";
        std::cout << "  BENCHMARK RESULTS\n";
        std::cout << " ================================================================\n\n";
        std::cout << " Game           | Best      | Gens | Time(ms) | Gen/s\n";
        std::cout << " ---------------+-----------+------+----------+--------\n";

        for (const auto& [name, result] : results) {
            double gps = result.generations_run / (result.elapsed_ms / 1000.0 + 1e-6);
            std::cout << " " << std::setw(15) << std::left << name
                      << "| " << std::setw(9) << std::right << std::fixed << std::setprecision(2) << result.best_fitness
                      << " | " << std::setw(4) << result.generations_run
                      << " | " << std::setw(8) << std::setprecision(1) << result.elapsed_ms
                      << " | " << std::setw(6) << std::setprecision(0) << gps << "\n";
        }
        std::cout << "\n ================================================================\n";

    } else if (mode == "gui") {
        viz::GUI gui;
        if (!gui.init()) {
            std::cerr << "Failed to initialize SDL2\n";
            return 1;
        }

        std::string game_name = (argc > 2) ? argv[2] : "Snake";
        bool found = false;
        for (const auto& gc : configs) {
            if (gc.name == game_name) found = true;
        }
        if (!found) { std::cerr << "Unknown game: " << game_name << "\n"; return 1; }

        for (const auto& gc : configs) {
            if (gc.name != game_name) continue;

            auto game = gc.factory();
            bool scoring = game->uses_scoring();
            size_t genome_size;
            if (scoring)
                genome_size = games::ScoringNetwork::weight_count(gc.input_size, gc.hidden_size);
            else
                genome_size = games::PolicyNetwork::weight_count(gc.input_size, gc.hidden_size, game->action_space());

            auto cfg = make_config(genome_size, 200, 500);
            ga::GAEngine engine(cfg);

            engine.run_batch([&](ga::Genome& g) {
                auto tls_game = gc.factory();
                tls_game->reset();
                std::vector<float> state(gc.input_size);
                int steps = 0;

                if (scoring) {
                    games::ScoringNetwork sn(gc.input_size, gc.hidden_size);
                    while (!tls_game->is_done() && steps < gc.max_steps) {
                        auto valid = tls_game->get_valid_moves();
                        if (valid.empty()) break;
                        tls_game->get_state(state.data());
                        int best_action = valid[0];
                        float best_score = -1e30f;
                        for (int a : valid) {
                            auto sim = tls_game->clone();
                            sim->step(a);
                            std::vector<float> ns(gc.input_size);
                            sim->get_state(ns.data());
                            float s = sn.score(g.genes.data(), ns.data());
                            if (s > best_score) { best_score = s; best_action = a; }
                        }
                        tls_game->step(best_action);
                        steps++;
                    }
                } else {
                    games::PolicyNetwork pn(gc.input_size, gc.hidden_size, tls_game->action_space());
                    while (!tls_game->is_done() && steps < gc.max_steps) {
                        tls_game->get_state(state.data());
                        int action = pn.decide(g.genes.data(), state.data());
                        tls_game->step(action);
                        steps++;
                    }
                }
                g.fitness = tls_game->fitness();
            }, [&](int gen, const ga::GAResult& r) {
                gui.render(r, gc.name, gen);
                if (gui.should_close()) engine.request_stop();
            });
        }

        while (!gui.should_close()) {
#ifdef HAS_SDL2
            SDL_Delay(100);
#endif
        }

    } else if (mode == "single" && argc > 2) {
        std::string game_name = argv[2];
        bool found = false;
        for (const auto& gc : configs) {
            if (gc.name == game_name) {
                run_game(gc, true);
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "Unknown game: " << game_name << "\n";
            std::cout << "Available: ";
            for (const auto& gc : configs) std::cout << gc.name << " ";
            std::cout << "\n";
            return 1;
        }
    } else {
        for (const auto& gc : configs) {
            run_game(gc, true);
        }
    }

    return 0;
}
