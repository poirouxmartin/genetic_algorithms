#pragma once

#include "game_interface.h"
#include <array>
#include <deque>
#include <algorithm>
#include <cmath>

namespace games {

class Snake : public GameInterface {
    static constexpr int WIDTH = 10;
    static constexpr int HEIGHT = 10;
    static constexpr int MAX_STEPS = 200;

    std::deque<std::pair<int,int>> snake_;
    std::pair<int,int> food_;
    int dx_, dy_;
    int steps_;
    int score_;
    bool done_;
    int last_action_;

public:
    Snake() { reset(); }

    std::string name() const override { return "Snake"; }
    int action_space() const override { return 4; }

    void reset() override {
        snake_.clear();
        int cx = WIDTH / 2, cy = HEIGHT / 2;
        snake_.push_back({cx, cy});
        snake_.push_back({cx - 1, cy});
        snake_.push_back({cx - 2, cy});
        dx_ = 1; dy_ = 0;
        steps_ = 0;
        score_ = 0;
        done_ = false;
        last_action_ = 0;
        spawn_food();
    }

    void step(int action) override {
        if (done_) return;

        static constexpr int dx[] = {0, 0, -1, 1};
        static constexpr int dy[] = {-1, 1, 0, 0};

        if (action >= 0 && action < 4) {
            if (dx[action] + dx_ != 0 || dy[action] + dy_ != 0) {
                dx_ = dx[action];
                dy_ = dy[action];
            }
            last_action_ = action;
        }

        auto head = snake_.front();
        head.first += dx_;
        head.second += dy_;
        steps_++;

        if (head.first < 0 || head.first >= WIDTH || head.second < 0 || head.second >= HEIGHT) {
            done_ = true;
            return;
        }

        for (const auto& seg : snake_) {
            if (seg == head) { done_ = true; return; }
        }

        snake_.push_front(head);

        if (head == food_) {
            score_++;
            spawn_food();
        } else {
            snake_.pop_back();
        }

        if (steps_ >= MAX_STEPS) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return static_cast<float>(score_) * 10.0f
             - static_cast<float>(steps_) * 0.01f
             + static_cast<float>(snake_.size());
    }

    int action_count() const override { return steps_; }

    void get_state(float* state) const {
        auto head = snake_.front();
        bool danger_straight = false, danger_right = false, danger_left = false;

        auto check = [&](int ndx, int ndy) -> bool {
            int nx = head.first + ndx, ny = head.second + ndy;
            if (nx < 0 || nx >= WIDTH || ny < 0 || ny >= HEIGHT) return true;
            for (const auto& s : snake_) if (s == std::make_pair(nx, ny)) return true;
            return false;
        };

        danger_straight = check(dx_, dy_);
        int rdx = -dy_, rdy = dx_;
        int ldx = dy_, ldy = -dx_;
        danger_right = check(rdx, rdy);
        danger_left = check(ldx, ldy);

        bool food_left = food_.first < head.first;
        bool food_right = food_.first > head.first;
        bool food_up = food_.second < head.second;
        bool food_down = food_.second > head.second;

        bool dir_up = dy_ == -1, dir_down = dy_ == 1;
        bool dir_left = dx_ == -1, dir_right = dx_ == 1;

        state[0] = danger_straight ? 1.0f : 0.0f;
        state[1] = danger_right ? 1.0f : 0.0f;
        state[2] = danger_left ? 1.0f : 0.0f;
        state[3] = dir_left ? 1.0f : 0.0f;
        state[4] = dir_right ? 1.0f : 0.0f;
        state[5] = dir_up ? 1.0f : 0.0f;
        state[6] = dir_down ? 1.0f : 0.0f;
        state[7] = food_left ? 1.0f : 0.0f;
        state[8] = food_right ? 1.0f : 0.0f;
        state[9] = food_up ? 1.0f : 0.0f;
        state[10] = food_down ? 1.0f : 0.0f;
    }

    std::string render_state() const override {
        std::string grid(WIDTH * HEIGHT, '.');
        for (const auto& s : snake_) {
            if (s.first >= 0 && s.first < WIDTH && s.second >= 0 && s.second < HEIGHT)
                grid[s.second * WIDTH + s.first] = '#';
        }
        if (!snake_.empty()) {
            auto h = snake_.front();
            if (h.first >= 0 && h.first < WIDTH && h.second >= 0 && h.second < HEIGHT)
                grid[h.second * WIDTH + h.first] = '@';
        }
        if (food_.first >= 0 && food_.first < WIDTH && food_.second >= 0 && food_.second < HEIGHT)
            grid[food_.second * WIDTH + food_.first] = '*';

        std::string result;
        for (int y = 0; y < HEIGHT; ++y) {
            result.append(grid.substr(y * WIDTH, WIDTH));
            result += '\n';
        }
        return result;
    }

private:
    void spawn_food() {
        std::vector<std::pair<int,int>> empty;
        for (int y = 0; y < HEIGHT; ++y)
            for (int x = 0; x < WIDTH; ++x) {
                bool occupied = false;
                for (const auto& s : snake_)
                    if (s == std::make_pair(x, y)) { occupied = true; break; }
                if (!occupied) empty.push_back({x, y});
            }
        if (!empty.empty()) {
            thread_local std::mt19937 rng(std::random_device{}());
            food_ = empty[std::uniform_int_distribution<size_t>(0, empty.size() - 1)(rng)];
        }
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(Snake, 11, {16}, 200)
