#pragma once

#include "game_interface.h"
#include <array>
#include <vector>
#include <algorithm>

namespace games {

class Mastermind : public GameInterface {
    static constexpr int CODE_LEN = 4;
    static constexpr int NUM_COLORS = 6;
    static constexpr int MAX_GUESSES = 10;

    std::array<int, CODE_LEN> secret_;
    std::array<int, CODE_LEN> current_guess_;
    int guesses_made_;
    bool done_;
    int exact_matches_;

public:
    Mastermind() { reset(); }

    std::string name() const override { return "Mastermind"; }
    int action_space() const override { return CODE_LEN * NUM_COLORS; }

    void reset() override {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, NUM_COLORS - 1);
        for (int& c : secret_) c = dist(rng);
        current_guess_.fill(0);
        guesses_made_ = 0;
        done_ = false;
        exact_matches_ = 0;
    }

    void step(int action) override {
        if (done_) return;

        int color = action % NUM_COLORS;
        int pos = action / NUM_COLORS;
        if (pos >= CODE_LEN) pos = CODE_LEN - 1;

        current_guess_[pos] = color;

        bool all_set = true;
        for (int i = 0; i < CODE_LEN; ++i)
            if (current_guess_[i] != secret_[i]) { all_set = false; break; }

        if (all_set || guesses_made_ >= MAX_GUESSES - 1) {
            done_ = true;
            return;
        }

        guesses_made_++;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        int exact = 0, color_match = 0;
        std::array<bool, CODE_LEN> s_used{}, g_used{};

        for (int i = 0; i < CODE_LEN; ++i) {
            if (current_guess_[i] == secret_[i]) {
                exact++;
                s_used[i] = g_used[i] = true;
            }
        }

        for (int i = 0; i < CODE_LEN; ++i) {
            if (g_used[i]) continue;
            for (int j = 0; j < CODE_LEN; ++j) {
                if (!s_used[j] && current_guess_[i] == secret_[j]) {
                    color_match++;
                    s_used[j] = true;
                    break;
                }
            }
        }

        return static_cast<float>(exact) * 10.0f + static_cast<float>(color_match) * 2.0f;
    }

    int action_count() const override { return guesses_made_; }

    std::string render_state() const override {
        std::string result;
        for (int i = 0; i < CODE_LEN; ++i) {
            if (done_)
                result += std::to_string(secret_[i]);
            else
                result += std::to_string(current_guess_[i]);
            if (i < CODE_LEN - 1) result += ' ';
        }
        return result;
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(Mastermind, 4, 8, 40)
