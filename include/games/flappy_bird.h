#pragma once

#include "game_interface.h"
#include <cmath>
#include <algorithm>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

namespace games {

class FlappyBird : public GameInterface {
    static constexpr float GRAVITY = 0.5f;
    static constexpr float JUMP_VEL = -8.0f;
    static constexpr float BASE_PIPE_SPEED = 2.0f;
    static constexpr int PIPE_GAP = 150;
    static constexpr int SCREEN_W = 400;
    static constexpr int SCREEN_H = 600;
    static constexpr int PIPE_W = 60;
    static constexpr int BIRD_SIZE = 20;

    float bird_y_, bird_vy_;
    float pipe_x_;
    float pipe_gap_y_;
    float pipe_speed_;
    int score_;
    int frames_;
    bool done_;

public:
    FlappyBird() { reset(); }

    std::string name() const override { return "FlappyBird"; }
    int action_space() const override { return 2; }

    void reset() override {
        bird_y_ = SCREEN_H / 2.0f;
        bird_vy_ = 0.0f;
        pipe_x_ = SCREEN_W + 100.0f;
        pipe_gap_y_ = 150.0f + static_cast<float>(std::rand() % 300);
        pipe_speed_ = BASE_PIPE_SPEED;
        score_ = 0;
        frames_ = 0;
        done_ = false;
    }

    void step(int action) override {
        if (done_) return;
        frames_++;

        if (action == 1) {
            bird_vy_ = JUMP_VEL;
        }

        bird_vy_ += GRAVITY;
        bird_y_ += bird_vy_;

        pipe_speed_ = BASE_PIPE_SPEED + score_ * 0.15f;
        pipe_x_ -= pipe_speed_;

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

        if (frames_ >= 1000) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return static_cast<float>(score_) * 30.0f
             + static_cast<float>(frames_) * 0.1f
             + pipe_speed_ * 5.0f;
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
        std::string grid(SCREEN_W / 10 * (SCREEN_H / 10), ' ');
        int gw = SCREEN_W / 10;
        int gh = SCREEN_H / 10;
        auto set = [&](int x, int y, char c) {
            if (x >= 0 && x < gw && y >= 0 && y < gh) grid[y * gw + x] = c;
        };
        int pipe_x_cell = static_cast<int>(pipe_x_) / 10;
        int gap_top = static_cast<int>(pipe_gap_y_) / 10;
        int gap_bot = gap_top + PIPE_GAP / 10;
        for (int y = 0; y < gh; ++y) {
            set(pipe_x_cell, y, '|');
            set(pipe_x_cell + 1, y, '|');
            set(pipe_x_cell + 2, y, '|');
            if (y < gap_top || y >= gap_bot) {
                set(pipe_x_cell, y, 'X');
                set(pipe_x_cell + 1, y, 'X');
                set(pipe_x_cell + 2, y, 'X');
            }
        }
        int bird_x = 5;
        int bird_y_cell = static_cast<int>(bird_y_) / 10;
        set(bird_x, bird_y_cell, 'O');
        set(bird_x + 1, bird_y_cell, 'O');
        set(bird_x, bird_y_cell + 1, 'O');
        set(bird_x - 1, bird_y_cell, '@');
        std::string result;
        for (int y = 0; y < gh; ++y) {
            result.append(grid.substr(y * gw, gw));
            result += '\n';
        }
        return result;
    }

    bool has_sdl_render() const override { return true; }

    void render_sdl(SDL_Renderer* renderer, int x, int y, int w, int h) const override {
        float sx = static_cast<float>(w) / SCREEN_W;
        float sy = static_cast<float>(h) / SCREEN_H;
        auto to_screen = [&](float px, float py) -> std::pair<int,int> {
            return {static_cast<int>(x + px * sx), static_cast<int>(y + py * sy)};
        };

        SDL_SetRenderDrawColor(renderer, 20, 24, 40, 255);
        SDL_Rect bg = {x, y, w, h};
        SDL_RenderFillRect(renderer, &bg);

        SDL_SetRenderDrawColor(renderer, 70, 75, 95, 255);
        for (int gy = 0; gy < h; gy += static_cast<int>(sy * 20)) {
            SDL_RenderDrawLine(renderer, x, y + gy, x + w, y + gy);
        }

        int pw = static_cast<int>(PIPE_W * sx);
        int gap_top = static_cast<int>(pipe_gap_y_ * sy);
        int gap_bot = static_cast<int>((pipe_gap_y_ + PIPE_GAP) * sy);
        auto [px, _] = to_screen(pipe_x_, 0);

        SDL_SetRenderDrawColor(renderer, 80, 180, 80, 255);
        SDL_Rect top_pipe = {px, y, pw, gap_top};
        SDL_Rect bot_pipe = {px, y + gap_bot, pw, h - gap_bot};
        SDL_RenderFillRect(renderer, &top_pipe);
        SDL_RenderFillRect(renderer, &bot_pipe);

        SDL_SetRenderDrawColor(renderer, 60, 140, 60, 255);
        SDL_Rect top_cap = {px - 4, y + gap_top - 12, pw + 8, 12};
        SDL_Rect bot_cap = {px - 4, y + gap_bot, pw + 8, 12};
        SDL_RenderFillRect(renderer, &top_cap);
        SDL_RenderFillRect(renderer, &bot_cap);

        auto [bx, by] = to_screen(80.0f, bird_y_);
        int br = static_cast<int>(BIRD_SIZE * sx * 0.5f);
        SDL_SetRenderDrawColor(renderer, 230, 200, 60, 255);
        for (int dy = -br; dy <= br; ++dy) {
            int dx_max = static_cast<int>(std::sqrt(static_cast<float>(br * br - dy * dy)));
            SDL_RenderDrawLine(renderer, bx - dx_max, by + dy, bx + dx_max, by + dy);
        }
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_Rect eye = {bx + br / 3, by - br / 3, br / 3, br / 3};
        SDL_RenderFillRect(renderer, &eye);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_Rect pupil = {bx + br / 2, by - br / 4, br / 4, br / 4};
        SDL_RenderFillRect(renderer, &pupil);

        int ground_y = y + h - static_cast<int>(sy * 20);
        SDL_SetRenderDrawColor(renderer, 100, 80, 50, 255);
        SDL_Rect ground = {x, ground_y, w, static_cast<int>(sy * 20)};
        SDL_RenderFillRect(renderer, &ground);
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(FlappyBird, 6, {10}, 1000)
