#pragma once

#include "game_interface.h"
#include <vector>
#include <algorithm>
#include <random>
#include <mutex>

namespace games {

class Maze : public GameInterface {
    static constexpr int WIDTH = 21;
    static constexpr int HEIGHT = 21;
    static constexpr int MAX_STEPS = 500;

    static std::array<char, WIDTH * HEIGHT> shared_grid_;
    static std::once_flag grid_init_flag_;

    std::array<char, WIDTH * HEIGHT> grid_;
    int player_x_, player_y_;
    int goal_x_, goal_y_;
    int steps_;
    bool done_;

public:
    Maze() { reset(); }

    std::string name() const override { return "Maze"; }
    int action_space() const override { return 4; }

    void reset() override {
        std::call_once(grid_init_flag_, [this]() {
            shared_grid_.fill('#');
            auto carve = [&](auto& self, int x, int y) -> void {
                shared_grid_[y * WIDTH + x] = ' ';
                static constexpr int dx[] = {0, 0, -2, 2};
                static constexpr int dy[] = {-2, 2, 0, 0};
                std::array<int, 4> dirs = {0, 1, 2, 3};
                thread_local std::mt19937 rng(std::random_device{}());
                std::shuffle(dirs.begin(), dirs.end(), rng);
                for (int d : dirs) {
                    int nx = x + dx[d], ny = y + dy[d];
                    if (nx > 0 && nx < WIDTH && ny > 0 && ny < HEIGHT && shared_grid_[ny * WIDTH + nx] == '#') {
                        shared_grid_[(y + dy[d] / 2) * WIDTH + (x + dx[d] / 2)] = ' ';
                        self(self, nx, ny);
                    }
                }
            };
            carve(carve, 1, 1);
        });
        grid_ = shared_grid_;
        player_x_ = 1;
        player_y_ = 1;
        goal_x_ = WIDTH - 2;
        goal_y_ = HEIGHT - 2;
        grid_[player_y_ * WIDTH + player_x_] = 'P';
        grid_[goal_y_ * WIDTH + goal_x_] = 'G';
        steps_ = 0;
        done_ = false;
    }

    void step(int action) override {
        if (done_) return;

        static constexpr int dx[] = {0, 0, -1, 1};
        static constexpr int dy[] = {-1, 1, 0, 0};

        int nx = player_x_ + dx[action];
        int ny = player_y_ + dy[action];

        if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT &&
            grid_[ny * WIDTH + nx] != '#') {
            player_x_ = nx;
            player_y_ = ny;
        }

        steps_++;

        if (player_x_ == goal_x_ && player_y_ == goal_y_) {
            done_ = true;
            return;
        }
        if (steps_ >= MAX_STEPS) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        float dist = std::abs(player_x_ - goal_x_) + std::abs(player_y_ - goal_y_);
        float max_dist = WIDTH + HEIGHT;
        float reached = (player_x_ == goal_x_ && player_y_ == goal_y_) ? 100.0f : 0.0f;
        return reached + (1.0f - dist / max_dist) * 10.0f - static_cast<float>(steps_) * 0.01f;
    }

    int action_count() const override { return steps_; }

    void get_state(float* state) const override {
        state[0] = static_cast<float>(player_x_) / WIDTH;
        state[1] = static_cast<float>(player_y_) / HEIGHT;
        state[2] = static_cast<float>(goal_x_) / WIDTH;
        state[3] = static_cast<float>(goal_y_) / HEIGHT;

        state[4] = (player_y_ > 0 && grid_[(player_y_ - 1) * WIDTH + player_x_] != '#') ? 1.0f : 0.0f;
        state[5] = (player_y_ < HEIGHT - 1 && grid_[(player_y_ + 1) * WIDTH + player_x_] != '#') ? 1.0f : 0.0f;
        state[6] = (player_x_ > 0 && grid_[player_y_ * WIDTH + player_x_ - 1] != '#') ? 1.0f : 0.0f;
        state[7] = (player_x_ < WIDTH - 1 && grid_[player_y_ * WIDTH + player_x_ + 1] != '#') ? 1.0f : 0.0f;

        float dx = (goal_x_ - player_x_) / static_cast<float>(WIDTH);
        float dy = (goal_y_ - player_y_) / static_cast<float>(HEIGHT);
        state[8] = dx;
        state[9] = dy;

        state[10] = (goal_y_ < player_y_) ? 1.0f : 0.0f;
        state[11] = (goal_x_ > player_x_) ? 1.0f : 0.0f;
    }

    std::string render_state() const override {
        std::string result;
        for (int y = 0; y < HEIGHT; ++y) {
            result.append(grid_.data() + y * WIDTH, WIDTH);
            result += '\n';
        }
        return result;
    }

};

inline std::array<char, Maze::WIDTH * Maze::HEIGHT> Maze::shared_grid_{};
inline std::once_flag Maze::grid_init_flag_;

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(Maze, 12, {12}, 500)
