#include "ga/ga_engine.h"
#include "ga/island_engine.h"
#include "ga/neat.h"
#include "ga/game_registry.h"
#include "games/snake.h"
#include "games/tic_tac_toe.h"
#include "games/flappy_bird.h"
#include "games/pong.h"
#include "games/mastermind.h"
#include "games/maze.h"
#include "games/othello.h"
#include "games/chess.h"
#include "games/breakout.h"
#include "games/space_invaders.h"
#include "games/tetris.h"
#include "games/car_race.h"
#include "viz/tui.h"
#include "viz/replay.h"
#include "viz/gui.h"
#include <algorithm>
#include <random>
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

ga::GAConfig make_config(size_t genome_size, size_t pop = 300, int gens = 300,
                         int input_size = 0, int hidden_size = 0) {
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
    cfg.input_size = input_size;
    cfg.hidden_size = hidden_size;
    return cfg;
}

bool matches_game(const GameConfig& gc, const std::string& name) {
    return gc.name == name || gc.display_name == name;
}

void apply_symmetry(ga::GAConfig& cfg, const GameConfig& gc, bool scoring) {
    if (!gc.input_symmetry.empty() && !scoring) {
        size_t in = gc.input_size, hid = gc.hidden_size;
        for (auto [i, j] : gc.input_symmetry) {
            for (size_t h = 0; h < hid; ++h)
                cfg.symmetry_pairs.emplace_back(
                    static_cast<int>(h * in + i),
                    static_cast<int>(h * in + j));
        }
    }
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

    auto cfg = make_config(genome_size, 300, 300,
                           static_cast<int>(gc.input_size), static_cast<int>(gc.hidden_size));
    apply_symmetry(cfg, gc, scoring);

    viz::TUI tui(80, 30);
    tui.set_enabled(show_tui);

    ga::GAEngine engine(cfg);

    std::cout << "Running " << gc.display_name << " (genome=" << genome_size
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
                if (valid.empty()) { tls_game->step(0); steps++; continue; }

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

    auto& configs = GameRegistry::games();

    if (mode == "bench" || mode == "benchmark") {
        std::vector<std::pair<std::string, ga::GAResult>> results;

        for (const auto& gc : configs) {
            auto result = run_game(gc, false);
            results.emplace_back(gc.display_name, std::move(result));
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

    } else if (mode == "config") {
        viz::GUI gui;
        if (!gui.init()) {
            std::cerr << "Failed to initialize SDL2\n";
            return 1;
        }

        int game_idx = 0;
        std::string initial_game = (argc > 2) ? argv[2] : "";
        if (!initial_game.empty()) {
            for (int i = 0; i < static_cast<int>(configs.size()); ++i) {
                if (matches_game(configs[i], initial_game)) { game_idx = i; break; }
            }
        }

        float pop_size = 300.0f;
        float gens = 300.0f;
        float mut_rate = 0.1f;
        float mut_sigma = 0.2f;
        float cross_rate = 0.85f;
        float tourney = 3.0f;
        float elitism = 0.05f;

        std::vector<viz::GUI::Slider> sliders = {
            {"Population",    &pop_size,    50, 2000, 50},
            {"Generations",   &gens,        50, 2000, 50},
            {"Mutation Rate", &mut_rate,    0.01f, 1.0f, 0.01f},
            {"Mutation Sigma",&mut_sigma,   0.01f, 2.0f, 0.01f},
            {"Crossover Rate",&cross_rate,  0.1f, 1.0f, 0.05f},
            {"Tournament",    &tourney,     2, 10, 1},
            {"Elitism",       &elitism,     0.0f, 0.2f, 0.01f},
        };

        bool run_clicked = false;
        while (gui.render_config(game_idx, configs, sliders, run_clicked)) {}

        if (run_clicked) {
            const auto& gc = configs[game_idx];
            {
                    auto game = gc.factory();
                    bool scoring = game->uses_scoring();
                    size_t genome_size;
                    if (scoring) {
                        genome_size = games::ScoringNetwork::weight_count(gc.input_size, gc.hidden_size);
                    } else {
                        genome_size = games::PolicyNetwork::weight_count(gc.input_size, gc.hidden_size, game->action_space());
                    }

                    auto cfg = make_config(genome_size,
                                           static_cast<size_t>(pop_size),
                                           static_cast<int>(gens),
                                           static_cast<int>(gc.input_size),
                                           static_cast<int>(gc.hidden_size));
                    cfg.mutation_rate = mut_rate;
                    cfg.mutation_sigma = mut_sigma;
                    cfg.crossover_rate = cross_rate;
                    cfg.tournament_size = static_cast<int>(tourney);
                    cfg.elitism_ratio = elitism;
                    apply_symmetry(cfg, gc, scoring);

                    ga::GAEngine engine(cfg);
                    auto result = engine.run_batch([&](ga::Genome& g) {
                        auto tls_game = gc.factory();
                        tls_game->reset();
                        std::vector<float> state(gc.input_size);
                        int steps = 0;
                        if (scoring) {
                            games::ScoringNetwork sn(gc.input_size, gc.hidden_size);
                            while (!tls_game->is_done() && steps < gc.max_steps) {
                                auto valid = tls_game->get_valid_moves();
                            if (valid.empty()) { tls_game->step(0); steps++; continue; }
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
                    },
                    [&](int gen, const ga::GAResult& r) {
                        const auto& best = engine.population().best();
                        const float* nn = best.genes.empty() ? nullptr : best.genes.data();
                        size_t os = scoring ? 0 : static_cast<size_t>(game->action_space());
                        gui.render_sim(r, gc.display_name, gen, cfg.max_generations, true,
                                       nn, gc.input_size, gc.hidden_size, os);
                        if (gui.should_close()) { engine.request_stop(); }
                    });

                    std::cout << "Final best: " << result.best_fitness << "\n";
                }
        }

    } else if (mode == "gui") {
        viz::GUI gui;
        if (!gui.init()) {
            std::cerr << "Failed to initialize SDL2\n";
            return 1;
        }

        int game_idx = 0;
        std::string initial_game = (argc > 2) ? argv[2] : "";
        if (!initial_game.empty()) {
            for (int i = 0; i < static_cast<int>(configs.size()); ++i) {
                if (matches_game(configs[i], initial_game)) { game_idx = i; break; }
            }
        }

        float pop_size = 300.0f, gens = 1000.0f;
        float mut_rate = 0.1f, mut_sigma = 0.2f;
        float cross_rate = 0.85f, tourney = 3.0f, elitism = 0.05f;
        float l2_lambda = 0.0f;
        int selected_game = game_idx;

        std::vector<viz::GUI::Slider> sliders = {
            {"Population",     &pop_size,     50, 2000, 50},
            {"Generations",    &gens,         50, 5000, 50},
            {"Mutation Rate",  &mut_rate,     0.01f, 1.0f, 0.01f},
            {"Mutation Sigma", &mut_sigma,    0.01f, 2.0f, 0.01f},
            {"Crossover Rate", &cross_rate,   0.1f, 1.0f, 0.05f},
            {"Tournament",     &tourney,      2, 10, 1},
            {"Elitism",        &elitism,      0.0f, 0.2f, 0.01f},
            {"L2 Weight Decay",&l2_lambda,    0.0f, 0.01f, 0.0001f},
        };

        while (!gui.should_close()) {
            bool run_clicked = false;
            bool config_open = true;
            while (config_open && !gui.should_close()) {
                SDL_Delay(16);
                config_open = gui.render_config(selected_game, configs, sliders, run_clicked);
            }
            if (gui.should_close()) break;
            if (!run_clicked) continue;

            const auto& gc = configs[selected_game];

            auto game = gc.factory();
            bool scoring = game->uses_scoring();
            size_t genome_size;
            if (scoring)
                genome_size = games::ScoringNetwork::weight_count(gc.input_size, gc.hidden_size);
            else
                genome_size = games::PolicyNetwork::weight_count(gc.input_size, gc.hidden_size, game->action_space());

            ga::GAConfig cfg;
            cfg.population_size = static_cast<size_t>(pop_size);
            cfg.gene_count = genome_size;
            cfg.min_gene = -1.0f;
            cfg.max_gene = 1.0f;
            cfg.crossover_rate = cross_rate;
            cfg.mutation_rate = mut_rate;
            cfg.mutation_sigma = mut_sigma;
            cfg.elitism_ratio = elitism;
            cfg.tournament_size = static_cast<size_t>(tourney);
            cfg.max_generations = static_cast<int>(gens);
            cfg.adaptive_mutation = true;
            cfg.parallel_eval = true;
            cfg.input_size = static_cast<int>(gc.input_size);
            cfg.hidden_size = static_cast<int>(gc.hidden_size);
            cfg.l2_lambda = l2_lambda;
            apply_symmetry(cfg, gc, scoring);

            ga::GAEngine engine(cfg);
            int action_space_size = game->action_space();
            ga::GAResult result;
            bool sim_running = false;
            std::string game_state;
            std::unique_ptr<games::GameInterface> watch_game;
            std::unique_ptr<games::GameInterface> display_game;
            int watch_step = 0;
            int display_step = 0;

            auto eval_fn = [&](ga::Genome& g) {
                auto tls_game = gc.factory();
                tls_game->reset();
                std::vector<float> state(gc.input_size);
                int steps = 0;
                if (scoring) {
                    games::ScoringNetwork sn(gc.input_size, gc.hidden_size);
                    while (!tls_game->is_done() && steps < gc.max_steps) {
                        auto valid = tls_game->get_valid_moves();
                        if (valid.empty()) { tls_game->step(0); steps++; continue; }
                        tls_game->get_state(state.data());
                        int ba = valid[0]; float bs = -1e30f;
                        for (int a : valid) {
                            auto sim = tls_game->clone();
                            sim->step(a);
                            std::vector<float> ns(gc.input_size);
                            sim->get_state(ns.data());
                            float s = sn.score(g.genes.data(), ns.data());
                            if (s > bs) { bs = s; ba = a; }
                        }
                        tls_game->step(ba); steps++;
                    }
                } else {
                    games::PolicyNetwork pn(gc.input_size, gc.hidden_size, tls_game->action_space());
                    while (!tls_game->is_done() && steps < gc.max_steps) {
                        tls_game->get_state(state.data());
                        int action = pn.decide(g.genes.data(), state.data());
                        tls_game->step(action); steps++;
                    }
                }
                g.fitness = tls_game->fitness();
                if (cfg.l2_lambda > 0.0f) {
                    float l2 = 0.0f;
                    for (float w : g.genes) l2 += w * w;
                    g.fitness -= cfg.l2_lambda * l2;
                }
            };

            auto play_best = [&]() -> std::string {
                const auto& best = engine.population().best();
                if (best.genes.empty()) return "";
                auto pg = gc.factory();
                pg->reset();
                std::vector<float> st(gc.input_size);
                if (scoring) {
                    games::ScoringNetwork sn(gc.input_size, gc.hidden_size);
                    int s = 0;
                    while (!pg->is_done() && s < gc.max_steps) {
                        auto valid = pg->get_valid_moves();
                        if (valid.empty()) { pg->step(0); s++; continue; }
                        pg->get_state(st.data());
                        int ba = valid[0]; float bs = -1e30f;
                        for (int a : valid) {
                            auto sim = pg->clone();
                            sim->step(a);
                            std::vector<float> ns(gc.input_size);
                            sim->get_state(ns.data());
                            float sc = sn.score(best.genes.data(), ns.data());
                            if (sc > bs) { bs = sc; ba = a; }
                        }
                        pg->step(ba); s++;
                    }
                } else {
                    games::PolicyNetwork pn(gc.input_size, gc.hidden_size, pg->action_space());
                    int s = 0;
                    while (!pg->is_done() && s < gc.max_steps) {
                        pg->get_state(st.data());
                        int action = pn.decide(best.genes.data(), st.data());
                        pg->step(action); s++;
                    }
                }
                return pg->render_state();
            };

            while (!gui.should_close()) {
                SDL_Delay(16);

                auto action = gui.consume_action();

                if (watch_game) {
                    if (action == viz::GUI::Action::PauseResume || action == viz::GUI::Action::Back) {
                        watch_game.reset();
                        sim_running = false;
                    } else if (action == viz::GUI::Action::Menu) {
                        watch_game.reset();
                        display_game.reset();
                        display_step = 0;
                        sim_running = false;
                        break;
                    } else {
                        if (watch_step < gc.max_steps && !watch_game->is_done()) {
                            const auto& best = result.best_genome;
                            if (!best.genes.empty()) {
                                std::vector<float> st(gc.input_size);
                                watch_game->get_state(st.data());
                                if (scoring) {
                                    games::ScoringNetwork sn(gc.input_size, gc.hidden_size);
                                    auto valid = watch_game->get_valid_moves();
                                    if (valid.empty()) {
                                        watch_game->step(0);
                                    } else {
                                        int ba = valid[0]; float bs = -1e30f;
                                        for (int a : valid) {
                                            auto sim = watch_game->clone();
                                            sim->step(a);
                                            std::vector<float> ns(gc.input_size);
                                            sim->get_state(ns.data());
                                            float sc = sn.score(best.genes.data(), ns.data());
                                            if (sc > bs) { bs = sc; ba = a; }
                                        }
                                        watch_game->step(ba);
                                    }
                                } else {
                                    games::PolicyNetwork pn(gc.input_size, gc.hidden_size, watch_game->action_space());
                                    int act = pn.decide(best.genes.data(), st.data());
                                    watch_game->step(act);
                                }
                                watch_step++;
                            }
                        }
                        gui.render_watch(watch_game->render_state(), gc.display_name,
                                         watch_step, gc.max_steps, watch_game->fitness(),
                                         result.best_fitness, watch_game.get());
                    }
                } else {
                    if (action == viz::GUI::Action::Menu) {
                        display_game.reset();
                        display_step = 0;
                        sim_running = false;
                        break;
                    }
                    if (action == viz::GUI::Action::PauseResume) sim_running = !sim_running;
                    if (action == viz::GUI::Action::Watch) {
                        sim_running = false;
                        watch_game = gc.factory();
                        watch_game->reset();
                        watch_step = 0;
                    }

                    if (sim_running && result.generations_run < cfg.max_generations)
                        engine.step(eval_fn, result);

                    if (result.generations_run > 0) {
                        const float* nn_data = nullptr;
                        size_t in_s = 0, hid_s = 0, out_s = 0;
                        if (!result.best_genome.genes.empty()) {
                            nn_data = result.best_genome.genes.data();
                            in_s = gc.input_size; hid_s = gc.hidden_size;
                            if (!scoring) out_s = action_space_size;
                        }

                        if (!display_game) { display_game = gc.factory(); display_game->reset(); display_step = 0; }
                        const auto& best2 = result.best_genome;
                        if (!best2.genes.empty()) {
                            if (display_game->is_done() || display_step >= gc.max_steps) {
                                display_game->reset();
                                display_step = 0;
                            }
                            if (!display_game->is_done() && display_step < gc.max_steps) {
                                std::vector<float> st2(gc.input_size);
                                if (scoring) {
                                    games::ScoringNetwork sn2(gc.input_size, gc.hidden_size);
                                    auto valid2 = display_game->get_valid_moves();
                                    if (valid2.empty()) {
                                        display_game->step(0);
                                    } else {
                                        display_game->get_state(st2.data());
                                        int ba2 = valid2[0]; float bs2 = -1e30f;
                                        for (int a : valid2) {
                                            auto sim2 = display_game->clone();
                                            sim2->step(a);
                                            std::vector<float> ns2(gc.input_size);
                                            sim2->get_state(ns2.data());
                                            float sc2 = sn2.score(best2.genes.data(), ns2.data());
                                            if (sc2 > bs2) { bs2 = sc2; ba2 = a; }
                                        }
                                        display_game->step(ba2);
                                    }
                                } else {
                                    games::PolicyNetwork pn2(gc.input_size, gc.hidden_size, display_game->action_space());
                                    display_game->get_state(st2.data());
                                    int act2 = pn2.decide(best2.genes.data(), st2.data());
                                    display_game->step(act2);
                                }
                                display_step++;
                            }
                        }
                        game_state = display_game->render_state();

                        gui.render_sim(result, gc.display_name, result.generations_run, cfg.max_generations,
                                       sim_running, nn_data, in_s, hid_s, out_s, game_state,
                                       display_game.get());
                    }
                }
            }
        }

    } else if (mode == "replay") {
        viz::GUI gui;
        if (!gui.init()) {
            std::cerr << "Failed to initialize SDL2\n";
            return 1;
        }

        std::string game_name = (argc > 2) ? argv[2] : "Snake";
        bool found = false;
        for (const auto& gc : configs) {
            if (matches_game(gc, game_name)) found = true;
        }
        if (!found) { std::cerr << "Unknown game: " << game_name << "\n"; return 1; }

        for (const auto& gc : configs) {
            if (!matches_game(gc, game_name)) continue;

            std::cout << "Running GA for " << gc.display_name << "...\n";
            auto game = gc.factory();
            bool scoring = game->uses_scoring();
            size_t genome_size;
            if (scoring)
                genome_size = games::ScoringNetwork::weight_count(gc.input_size, gc.hidden_size);
            else
                genome_size = games::PolicyNetwork::weight_count(gc.input_size, gc.hidden_size, game->action_space());

            auto cfg = make_config(genome_size, 200, 300,
                                   static_cast<int>(gc.input_size), static_cast<int>(gc.hidden_size));
            apply_symmetry(cfg, gc, scoring);
            ga::GAEngine engine(cfg);

            auto result = engine.run_batch([&](ga::Genome& g) {
                auto tls_game = gc.factory();
                tls_game->reset();
                std::vector<float> state(gc.input_size);
                int steps = 0;

                if (scoring) {
                    games::ScoringNetwork sn(gc.input_size, gc.hidden_size);
                    while (!tls_game->is_done() && steps < gc.max_steps) {
                        auto valid = tls_game->get_valid_moves();
                        if (valid.empty()) { tls_game->step(0); steps++; continue; }
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
            });

            std::cout << "GA done. Best fitness: " << result.best_fitness
                      << " (" << result.generations_run << " gens, "
                      << result.elapsed_ms << " ms)\n";
            std::cout << "Capturing replay...\n";

            viz::ReplayData replay = viz::capture_replay(
                result.best_genome, gc.factory, gc.display_name,
                gc.input_size, gc.hidden_size, gc.max_steps);

            std::cout << "Replaying " << replay.frames.size() << " frames. Press ESC to exit.\n";

            {
                int frame = 0;
                float speed = 1.0f;
                bool playing = true;
                Uint32 last_tick = SDL_GetTicks();
                bool done = false;

                while (!gui.should_close() && !done) {
                    Uint32 now = SDL_GetTicks();
                    float dt = (now - last_tick) / 1000.0f;
                    last_tick = now;

                    SDL_Event e;
                    while (SDL_PollEvent(&e)) {
                        if (e.type == SDL_QUIT) { done = true; break; }
                        if (e.type == SDL_KEYDOWN) {
                            int max_frame = static_cast<int>(replay.frames.size()) - 1;
                            switch (e.key.keysym.sym) {
                                case SDLK_ESCAPE: done = true; break;
                                case SDLK_SPACE: playing = !playing; break;
                                case SDLK_RIGHT: if (frame < max_frame) frame++; break;
                                case SDLK_LEFT: if (frame > 0) frame--; break;
                                case SDLK_UP: speed += 0.5f; if (speed > 10.0f) speed = 10.0f; break;
                                case SDLK_DOWN: speed -= 0.5f; if (speed < 0.5f) speed = 0.5f; break;
                                case SDLK_r: frame = 0; break;
                                default: break;
                            }
                        }
                    }

                    if (playing) {
                        frame += static_cast<int>(dt * speed * 30.0f);
                        if (frame >= static_cast<int>(replay.frames.size())) {
                            frame = 0;
                        }
                    }

                    gui.render_replay(replay, frame, speed);
                }
            }
        }

    } else if (mode == "dashboard") {
        viz::GUI gui;
        if (!gui.init()) {
            std::cerr << "Failed to initialize SDL2\n";
            return 1;
        }

        std::vector<std::pair<std::string, ga::GAResult>> results;
        for (const auto& gc : configs) {
            std::cout << "Running " << gc.display_name << "...\n";
            auto result = run_game(gc, false);
            results.emplace_back(gc.display_name, std::move(result));
            gui.render_dashboard(results);
        }

        while (!gui.should_close()) {
            SDL_Delay(100);
        }

    } else if (mode == "island" && argc > 2) {
        std::string game_name = argv[2];
        bool found = false;
        for (const auto& gc : configs) {
            if (matches_game(gc, game_name)) found = true;
        }
        if (!found) { std::cerr << "Unknown game: " << game_name << "\n"; return 1; }

        for (const auto& gc : configs) {
            if (!matches_game(gc, game_name)) continue;

            auto game = gc.factory();
            bool scoring = game->uses_scoring();
            size_t genome_size;
            if (scoring)
                genome_size = games::ScoringNetwork::weight_count(gc.input_size, gc.hidden_size);
            else
                genome_size = games::PolicyNetwork::weight_count(gc.input_size, gc.hidden_size, game->action_space());

            ga::IslandConfig icfg;
            icfg.num_islands = 4;
            icfg.pop_per_island = 75;
            icfg.migration_interval = 20;
            icfg.migration_count = 5;
            icfg.base_config = make_config(genome_size, 75, 300,
                                           static_cast<int>(gc.input_size), static_cast<int>(gc.hidden_size));
            icfg.base_config.adaptive_mutation = true;

            ga::IslandGAEngine engine(icfg);

            auto result = engine.run_batch([&](ga::Genome& g) {
                auto tls_game = gc.factory();
                tls_game->reset();
                std::vector<float> state(gc.input_size);
                int steps = 0;

                if (scoring) {
                    games::ScoringNetwork sn(gc.input_size, gc.hidden_size);
                    while (!tls_game->is_done() && steps < gc.max_steps) {
                        auto valid = tls_game->get_valid_moves();
                        if (valid.empty()) { tls_game->step(0); steps++; continue; }
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
            });

            double gps = result.generations_run / (result.elapsed_ms / 1000.0 + 1e-6);
            std::cout << "  Best: " << result.best_fitness
                      << "  Time: " << std::fixed << std::setprecision(1) << result.elapsed_ms << " ms"
                      << "  Gen/s: " << std::setprecision(0) << gps << "\n" << std::flush;
        }

    } else if (mode == "neat" && argc > 2) {
        std::string game_name = argv[2];
        bool found = false;
        for (const auto& gc : configs) {
            if (matches_game(gc, game_name)) found = true;
        }
        if (!found) { std::cerr << "Unknown game: " << game_name << "\n"; return 1; }

        for (const auto& gc : configs) {
            if (!matches_game(gc, game_name)) continue;

            auto game = gc.factory();
            int action_space_size = game->action_space();

            neat::NEATConfig ncfg;
            ncfg.population_size = 150;
            ncfg.max_generations = 300;

            neat::NEATEngine engine(gc.input_size, action_space_size, ncfg);

            auto result = engine.run(gc.input_size, action_space_size,
                [&](neat::NEATNetwork& net) -> float {
                    auto tls_game = gc.factory();
                    tls_game->reset();
                    std::vector<float> state(gc.input_size);
                    int steps = 0;
                    while (!tls_game->is_done() && steps < gc.max_steps) {
                        tls_game->get_state(state.data());
                        int action = net.decide(state.data());
                        tls_game->step(action);
                        steps++;
                    }
                    return tls_game->fitness();
                });

            double gps = result.generations_run / (result.elapsed_ms / 1000.0 + 1e-6);
            std::cout << "  Best: " << result.best_fitness
                      << "  Species: " << result.num_species
                      << "  Time: " << std::fixed << std::setprecision(1) << result.elapsed_ms << " ms"
                      << "  Gen/s: " << std::setprecision(0) << gps << "\n" << std::flush;
        }

    } else if (mode == "coevolve" && argc > 2) {
        std::string game_name = argv[2];
        bool found = false;
        for (const auto& gc : configs) {
            if (matches_game(gc, game_name)) {
                found = true;

                int n_matches = 3;
                if (argc > 3) n_matches = std::atoi(argv[3]);

                auto game = gc.factory();
                if (!game->uses_scoring()) {
                    std::cerr << "Coevolution only supports scoring games (Othello, Chess)\n";
                    return 1;
                }

                size_t genome_size = games::ScoringNetwork::weight_count(gc.input_size, gc.hidden_size);
    auto cfg = make_config(genome_size, 300, 300,
                           static_cast<int>(gc.input_size), static_cast<int>(gc.hidden_size));
                apply_symmetry(cfg, gc, true);
                ga::GAEngine engine(cfg);

                std::cout << "Running coevolution " << gc.display_name
                          << " (" << n_matches << " matches/genome)...\n" << std::flush;

                std::mt19937 match_rng(42);

                auto result = engine.run_batch([&](ga::Genome& g) {
                    auto& pop = engine.population().individuals();
                    float total_score = 0.0f;

                    for (int m = 0; m < n_matches; ++m) {
                        size_t opp_idx = std::uniform_int_distribution<size_t>(0, pop.size() - 1)(match_rng);
                        if (opp_idx == &g - pop.data()) {
                            opp_idx = (opp_idx + 1) % pop.size();
                        }

                        auto match_game = gc.factory();
                        match_game->reset();
                        games::ScoringNetwork sn(gc.input_size, gc.hidden_size);

                        std::vector<float> state(gc.input_size);
                        int steps = 0;

                        while (!match_game->is_done() && steps < gc.max_steps) {
                            auto valid = match_game->get_valid_moves();
                            if (valid.empty()) { match_game->step(0); steps++; continue; }

                            match_game->get_state(state.data());
                            int best_action = valid[0];
                            float best_score = -1e30f;
                            for (int a : valid) {
                                auto sim = match_game->clone();
                                sim->step(a);
                                std::vector<float> new_state(gc.input_size);
                                sim->get_state(new_state.data());

                                const ga::Genome& current = (steps % 2 == 0) ? g : pop[opp_idx];
                                float s = sn.score(current.genes.data(), new_state.data());
                                if (s > best_score) { best_score = s; best_action = a; }
                            }
                            match_game->step(best_action);
                            steps++;
                        }
                        total_score += match_game->fitness();
                    }
                    g.fitness = total_score / static_cast<float>(n_matches);
                },
                [&](int gen, const ga::GAResult& r) {
                    double gps = r.generations_run / (r.elapsed_ms / 1000.0 + 1e-6);
                    std::cout << "  Gen " << std::setw(4) << gen
                              << "  Best: " << std::setw(7) << std::fixed << std::setprecision(2) << r.best_fitness
                              << "  Avg: " << std::setw(7) << std::setprecision(2) << r.avg_fitness
                              << "  Gen/s: " << std::setw(6) << std::setprecision(0) << gps << "\n" << std::flush;
                });

                double gps = result.generations_run / (result.elapsed_ms / 1000.0 + 1e-6);
                std::cout << "  Final Best: " << result.best_fitness
                          << "  Time: " << std::fixed << std::setprecision(1) << result.elapsed_ms << " ms"
                          << "  Gen/s: " << std::setprecision(0) << gps << "\n" << std::flush;
                break;
            }
        }
        if (!found) {
            std::cerr << "Unknown game: " << game_name << "\n";
            std::cout << "Available: ";
            for (const auto& gc : configs) std::cout << gc.display_name << " ";
            std::cout << "\n";
            return 1;
        }

    } else if (mode == "single" && argc > 2) {
        std::string game_name = argv[2];
        bool found = false;
        for (const auto& gc : configs) {
            if (matches_game(gc, game_name)) {
                run_game(gc, true);
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "Unknown game: " << game_name << "\n";
            std::cout << "Available: ";
            for (const auto& gc : configs) std::cout << gc.display_name << " ";
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
