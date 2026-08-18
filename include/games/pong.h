#pragma once

#include "game_interface.h"
#include <cmath>
#include <algorithm>

namespace games {

class Pong : public GameInterface {
    static constexpr int W = 80;
    static constexpr int H = 40;
    static constexpr int PADDLE_H = 6;
    static constexpr int MAX_ROUNDS = 500;

    float ball_x_, ball_y_, ball_vx_, ball_vy_;
    float paddle_y_;
    float ai_paddle_y_;
    int score_player_, score_ai_;
    int rounds_;
    bool done_;

public:
    Pong() { reset(); }

    std::string name() const override { return "Pong"; }
    int action_space() const override { return 3; }

    void reset() override {
        ball_x_ = W / 2.0f;
        ball_y_ = H / 2.0f;
        ball_vx_ = 1.0f;
        ball_vy_ = 0.5f;
        paddle_y_ = H / 2.0f - PADDLE_H / 2.0f;
        ai_paddle_y_ = H / 2.0f - PADDLE_H / 2.0f;
        score_player_ = 0;
        score_ai_ = 0;
        rounds_ = 0;
        done_ = false;
    }

    void step(int action) override {
        if (done_) return;

        if (action == 0) paddle_y_ -= 1.0f;
        else if (action == 2) paddle_y_ += 1.0f;
        paddle_y_ = std::clamp(paddle_y_, 0.0f, static_cast<float>(H - PADDLE_H));

        ai_paddle_y_ = std::clamp(ball_y_ - PADDLE_H / 2.0f, 0.0f, static_cast<float>(H - PADDLE_H));

        ball_x_ += ball_vx_;
        ball_y_ += ball_vy_;

        if (ball_y_ <= 0 || ball_y_ >= H - 1) {
            ball_vy_ = -ball_vy_;
            ball_y_ = std::clamp(ball_y_, 1.0f, static_cast<float>(H - 2));
        }

        if (ball_x_ <= 2 && ball_vx_ < 0) {
            if (ball_y_ >= paddle_y_ && ball_y_ <= paddle_y_ + PADDLE_H) {
                ball_vx_ = -ball_vx_ * 1.05f;
                ball_vy_ += (ball_y_ - (paddle_y_ + PADDLE_H / 2.0f)) * 0.3f;
            } else {
                score_ai_++;
                rounds_++;
                reset_ball();
            }
        }

        if (ball_x_ >= W - 3 && ball_vx_ > 0) {
            if (ball_y_ >= ai_paddle_y_ && ball_y_ <= ai_paddle_y_ + PADDLE_H) {
                ball_vx_ = -ball_vx_;
            } else {
                score_player_++;
                rounds_++;
                reset_ball();
            }
        }

        if (rounds_ >= MAX_ROUNDS) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return static_cast<float>(score_player_) * 10.0f
             - static_cast<float>(score_ai_) * 10.0f
             + static_cast<float>(rounds_);
    }

    int action_count() const override { return rounds_; }

    void get_state(float* state) const {
        state[0] = ball_x_ / W;
        state[1] = ball_y_ / H;
        state[2] = ball_vx_ / 2.0f;
        state[3] = ball_vy_ / 2.0f;
        state[4] = paddle_y_ / H;
        state[5] = ai_paddle_y_ / H;
        state[6] = static_cast<float>(score_player_) / 10.0f;
        state[7] = static_cast<float>(score_ai_) / 10.0f;
    }

    std::string render_state() const override {
        std::string grid(H * (W + 1), ' ');
        for (int y = 0; y < H; ++y) grid[y * (W + 1) + W] = '\n';

        auto set = [&](int x, int y, char c) {
            if (x >= 0 && x < W && y >= 0 && y < H)
                grid[y * (W + 1) + x] = c;
        };

        int bx = static_cast<int>(ball_x_), by = static_cast<int>(ball_y_);
        set(bx, by, 'O');

        for (int i = 0; i < PADDLE_H; ++i) {
            set(1, static_cast<int>(paddle_y_) + i, '|');
            set(W - 2, static_cast<int>(ai_paddle_y_) + i, '|');
        }

        return grid;
    }

private:
    void reset_ball() {
        ball_x_ = W / 2.0f;
        ball_y_ = H / 2.0f;
        ball_vx_ = (rounds_ % 2 == 0) ? 1.0f : -1.0f;
        ball_vy_ = (static_cast<float>(std::rand() % 100) / 100.0f - 0.5f) * 2.0f;
    }
};

} // namespace games
