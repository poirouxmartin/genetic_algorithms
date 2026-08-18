#pragma once

#include "game_interface.h"
#include <cmath>
#include <algorithm>

namespace games {

class FlappyBird : public GameInterface {
    static constexpr float GRAVITY = 0.5f;
    static constexpr float JUMP_VEL = -8.0f;
    static constexpr float PIPE_SPEED = 3.0f;
    static constexpr int PIPE_GAP = 150;
    static constexpr int SCREEN_W = 400;
    static constexpr int SCREEN_H = 600;
    static constexpr int PIPE_W = 60;
    static constexpr int BIRD_SIZE = 20;
    static constexpr int MAX_FRAMES = 1000;

    float bird_y_, bird_vy_;
    float pipe_x_;
    float pipe_gap_y_;
    int score_;
    int frames_;
    bool done_;
    int action_count_;

public:
    FlappyBird() { reset(); }

    std::string name() const override { return "FlappyBird"; }
    int action_space() const override { return 2; }

    void reset() override {
        bird_y_ = SCREEN_H / 2.0f;
        bird_vy_ = 0.0f;
        pipe_x_ = SCREEN_W + 100.0f;
        pipe_gap_y_ = 150.0f + static_cast<float>(std::rand() % 300);
        score_ = 0;
        frames_ = 0;
        done_ = false;
        action_count_ = 0;
    }

    void step(int action) override {
        if (done_) return;
        frames_++;

        if (action == 1) {
            bird_vy_ = JUMP_VEL;
            action_count_++;
        }

        bird_vy_ += GRAVITY;
        bird_y_ += bird_vy_;

        pipe_x_ -= PIPE_SPEED;

        if (pipe_x_ < -PIPE_W) {
            pipe_x_ = SCREEN_W + 100.0f + static_cast<float>(std::rand() % 200);
            pipe_gap_y_ = 100.0f + static_cast<float>(std::rand() % 300);
            score_++;
        }

        if (bird_y_ < 0 || bird_y_ > SCREEN_H - BIRD_SIZE) {
            done_ = true;
            return;
        }

        float bird_x = 80.0f;
        if (pipe_x_ < bird_x + BIRD_SIZE && pipe_x_ + PIPE_W > bird_x) {
            if (bird_y_ < pipe_gap_y_ || bird_y_ + BIRD_SIZE > pipe_gap_y_ + PIPE_GAP) {
                done_ = true;
                return;
            }
        }

        if (frames_ >= MAX_FRAMES) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return static_cast<float>(score_) * 10.0f
             + static_cast<float>(frames_) * 0.01f
             + (bird_y_ > pipe_gap_y_ && bird_y_ < pipe_gap_y_ + PIPE_GAP ? 1.0f : 0.0f);
    }

    int action_count() const override { return frames_; }

    void get_state(float* state) const {
        float bird_x = 80.0f;
        float rel_x = (pipe_x_ - bird_x) / static_cast<float>(SCREEN_W);
        float gap_center = pipe_gap_y_ + PIPE_GAP / 2.0f;
        float rel_y = (bird_y_ - gap_center) / static_cast<float>(SCREEN_H);

        state[0] = bird_y_ / SCREEN_H;
        state[1] = bird_vy_ / 10.0f;
        state[2] = rel_x;
        state[3] = rel_y;
        state[4] = (pipe_gap_y_) / SCREEN_H;
        state[5] = (pipe_gap_y_ + PIPE_GAP) / SCREEN_H;
    }

    std::string render_state() const override {
        return "FlappyBird: Score=" + std::to_string(score_) +
               " Y=" + std::to_string((int)bird_y_);
    }
};

} // namespace games
