#pragma once

#include "game_interface.h"
#include <array>
#include <algorithm>
#include <cmath>

namespace games {

class Breakout : public GameInterface {
    static constexpr int GRID_W = 20;
    static constexpr int GRID_H = 25;
    static constexpr int PADDLE_W = 4;
    static constexpr int BALL_SPEED = 1;
    static constexpr int NUM_BRICK_ROWS = 5;
    static constexpr int MAX_FRAMES = 500;

    int paddle_x_;
    float ball_x_, ball_y_;
    float ball_vx_, ball_vy_;
    std::array<bool, GRID_W * GRID_H> bricks_{};
    int score_;
    int frames_;
    bool done_;
    int action_count_;

public:
    Breakout() { reset(); }

    std::string name() const override { return "Breakout"; }
    int action_space() const override { return 3; }

    void reset() override {
        paddle_x_ = GRID_W / 2 - PADDLE_W / 2;
        ball_x_ = GRID_W / 2.0f;
        ball_y_ = GRID_H - 3.0f;
        ball_vx_ = 0.7f;
        ball_vy_ = -1.0f;
        bricks_.fill(false);
        for (int r = 0; r < NUM_BRICK_ROWS; ++r)
            for (int c = 0; c < GRID_W; ++c)
                bricks_[r * GRID_W + c] = true;
        score_ = 0;
        frames_ = 0;
        done_ = false;
        action_count_ = 0;
    }

    void step(int action) override {
        if (done_) return;
        frames_++;

        if (action == 1 && paddle_x_ > 0) paddle_x_--;
        else if (action == 2 && paddle_x_ + PADDLE_W < GRID_W) paddle_x_++;

        ball_x_ += ball_vx_ * BALL_SPEED;
        ball_y_ += ball_vy_ * BALL_SPEED;

        if (ball_x_ < 0) { ball_x_ = 0; ball_vx_ = std::abs(ball_vx_); }
        if (ball_x_ >= GRID_W) { ball_x_ = GRID_W - 1.0f; ball_vx_ = -std::abs(ball_vx_); }
        if (ball_y_ < 0) { ball_y_ = 0; ball_vy_ = std::abs(ball_vy_); }

        int bx = std::clamp(static_cast<int>(ball_x_), 0, GRID_W - 1);
        int by = std::clamp(static_cast<int>(ball_y_), 0, GRID_H - 1);
        if (by >= 0 && by < GRID_H && bricks_[by * GRID_W + bx]) {
            bricks_[by * GRID_W + bx] = false;
            score_++;
            ball_vy_ = -ball_vy_;
        }

        int px = std::clamp(static_cast<int>(ball_x_), 0, GRID_W - 1);
        int py = GRID_H - 2;
        if (ball_y_ >= py && px >= paddle_x_ && px < paddle_x_ + PADDLE_W && ball_vy_ > 0) {
            ball_vy_ = -std::abs(ball_vy_);
            float center = paddle_x_ + PADDLE_W / 2.0f;
            ball_vx_ = (ball_x_ - center) / (PADDLE_W / 2.0f) * 0.8f;
        }

        if (ball_y_ >= GRID_H) done_ = true;

        bool all_clear = true;
        for (bool b : bricks_) { if (b) { all_clear = false; break; } }
        if (all_clear) done_ = true;

        if (frames_ >= MAX_FRAMES) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return static_cast<float>(score_) * 10.0f
             + static_cast<float>(frames_) * 0.01f;
    }

    int action_count() const override { return frames_; }

    void get_state(float* state) const override {
        state[0] = static_cast<float>(paddle_x_) / GRID_W;
        state[1] = ball_x_ / GRID_W;
        state[2] = ball_y_ / GRID_H;
        state[3] = ball_vx_;
        state[4] = ball_vy_;
        int bx = std::clamp(static_cast<int>(ball_x_), 0, GRID_W - 1);
        int by = std::clamp(static_cast<int>(ball_y_) + 1, 0, GRID_H - 1);
        state[5] = bricks_[by * GRID_W + bx] ? 1.0f : 0.0f;
    }

    std::string render_state() const override {
        std::string result;
        for (int r = 0; r < GRID_H; ++r) {
            for (int c = 0; c < GRID_W; ++c) {
                if (r < NUM_BRICK_ROWS && bricks_[r * GRID_W + c]) {
                    result += '#';
                } else if (r == GRID_H - 2 && c >= paddle_x_ && c < paddle_x_ + PADDLE_W) {
                    result += '=';
                } else if (c == static_cast<int>(ball_x_) && r == static_cast<int>(ball_y_)) {
                    result += 'o';
                } else {
                    result += ' ';
                }
            }
            result += '\n';
        }
        return result;
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(Breakout, 6, 10, 500)
