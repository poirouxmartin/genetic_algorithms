#pragma once

#include "game_interface.h"
#include <array>
#include <algorithm>
#include <cmath>

namespace games {

class SpaceInvaders : public GameInterface {
    static constexpr int GRID_W = 20;
    static constexpr int GRID_H = 15;
    static constexpr int NUM_INVADERS = 8;
    static constexpr int MAX_FRAMES = 500;

    int ship_x_;
    float invader_x_[NUM_INVADERS];
    int invader_y_[NUM_INVADERS];
    bool invader_alive_[NUM_INVADERS];
    int invader_dir_;
    float bullet_x_, bullet_y_;
    bool bullet_active_;
    int score_;
    int frames_;
    bool done_;
    int action_count_;
    int invaders_killed_;

public:
    SpaceInvaders() { reset(); }

    std::string name() const override { return "SpaceInvaders"; }
    int action_space() const override { return 3; }

    void reset() override {
        ship_x_ = GRID_W / 2;
        for (int i = 0; i < NUM_INVADERS; ++i) {
            invader_x_[i] = 2.0f + (i % 4) * 4.0f;
            invader_y_[i] = 1 + i / 4;
            invader_alive_[i] = true;
        }
        invader_dir_ = 1;
        bullet_x_ = 0;
        bullet_y_ = 0;
        bullet_active_ = false;
        score_ = 0;
        frames_ = 0;
        done_ = false;
        action_count_ = 0;
        invaders_killed_ = 0;
    }

    void step(int action) override {
        if (done_) return;
        frames_++;

        if (action == 1 && ship_x_ > 0) ship_x_--;
        else if (action == 2 && ship_x_ < GRID_W - 1) ship_x_++;
        else if (action == 2) {} // stay
        action_count_++;

        if (action == 0 && !bullet_active_) {
            bullet_x_ = ship_x_;
            bullet_y_ = GRID_H - 2;
            bullet_active_ = true;
        }

        if (bullet_active_) {
            bullet_y_ -= 1.0f;
            if (bullet_y_ < 0) bullet_active_ = false;

            if (bullet_active_) {
                int bx = std::clamp(static_cast<int>(bullet_x_), 0, GRID_W - 1);
                int by = std::clamp(static_cast<int>(bullet_y_), 0, GRID_H - 1);
                for (int i = 0; i < NUM_INVADERS; ++i) {
                    if (invader_alive_[i] &&
                        bx == static_cast<int>(invader_x_[i]) &&
                        by == invader_y_[i]) {
                        invader_alive_[i] = false;
                        bullet_active_ = false;
                        score_ += 10;
                        invaders_killed_++;
                        break;
                    }
                }
            }
        }

        bool move_down = false;
        for (int i = 0; i < NUM_INVADERS; ++i) {
            if (!invader_alive_[i]) continue;
            invader_x_[i] += invader_dir_ * 0.3f;
            if (invader_x_[i] < 0 || invader_x_[i] >= GRID_W) move_down = true;
        }
        if (move_down) {
            invader_dir_ = -invader_dir_;
            for (int i = 0; i < NUM_INVADERS; ++i) {
                if (!invader_alive_[i]) continue;
                invader_x_[i] = std::clamp(invader_x_[i], 0.0f, static_cast<float>(GRID_W - 1));
                invader_y_[i]++;
            }
        }

        for (int i = 0; i < NUM_INVADERS; ++i) {
            if (invader_alive_[i] && invader_y_[i] >= GRID_H - 2) {
                done_ = true;
                return;
            }
        }

        if (invaders_killed_ == NUM_INVADERS) {
            done_ = true;
            return;
        }

        if (frames_ >= MAX_FRAMES) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return static_cast<float>(score_) * 1.0f
             + static_cast<float>(frames_) * 0.01f
             + (invaders_killed_ == NUM_INVADERS ? 100.0f : 0.0f);
    }

    int action_count() const override { return frames_; }

    void get_state(float* state) const override {
        state[0] = static_cast<float>(ship_x_) / GRID_W;
        state[1] = bullet_active_ ? bullet_y_ / GRID_H : 0.0f;
        state[2] = bullet_active_ ? bullet_x_ / GRID_W : 0.0f;
        float nearest_x = 0.5f, nearest_y = 0.5f;
        float min_dist = 1e9f;
        for (int i = 0; i < NUM_INVADERS; ++i) {
            if (!invader_alive_[i]) continue;
            float dx = invader_x_[i] - ship_x_;
            float dy = static_cast<float>(invader_y_[i]) - (GRID_H - 2);
            float dist = dx * dx + dy * dy;
            if (dist < min_dist) { min_dist = dist; nearest_x = invader_x_[i] / GRID_W; nearest_y = static_cast<float>(invader_y_[i]) / GRID_H; }
        }
        state[3] = nearest_x;
        state[4] = nearest_y;
    }

    std::string render_state() const override {
        std::string result;
        for (int r = 0; r < GRID_H; ++r) {
            for (int c = 0; c < GRID_W; ++c) {
                bool drawn = false;
                if (r == GRID_H - 1 && c == ship_x_) { result += 'A'; drawn = true; }
                if (!drawn && bullet_active_ && c == static_cast<int>(bullet_x_) && r == static_cast<int>(bullet_y_)) { result += '|'; drawn = true; }
                if (!drawn) {
                    bool found = false;
                    for (int i = 0; i < NUM_INVADERS; ++i) {
                        if (invader_alive_[i] && c == static_cast<int>(invader_x_[i]) && r == invader_y_[i]) {
                            result += 'V';
                            found = true;
                            break;
                        }
                    }
                    if (!found) result += ' ';
                }
            }
            result += '\n';
        }
        return result;
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(SpaceInvaders, 5, {10}, 500)
