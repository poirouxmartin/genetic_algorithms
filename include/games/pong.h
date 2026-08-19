#pragma once

#include "game_interface.h"
#include <cmath>
#include <algorithm>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

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

        float ai_target = ball_y_ - PADDLE_H / 2.0f;
        float ai_diff = ai_target - ai_paddle_y_;
        float ai_speed = 1.2f;
        if (ai_diff > ai_speed) ai_paddle_y_ += ai_speed;
        else if (ai_diff < -ai_speed) ai_paddle_y_ -= ai_speed;
        else ai_paddle_y_ = ai_target;
        ai_paddle_y_ = std::clamp(ai_paddle_y_, 0.0f, static_cast<float>(H - PADDLE_H));

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

    bool has_sdl_render() const override { return true; }

    void render_sdl(SDL_Renderer* renderer, int x, int y, int w, int h) const override {
        float sx = static_cast<float>(w) / W;
        float sy = static_cast<float>(h) / H;

        SDL_SetRenderDrawColor(renderer, 20, 24, 40, 255);
        SDL_Rect bg = {x, y, w, h};
        SDL_RenderFillRect(renderer, &bg);

        SDL_SetRenderDrawColor(renderer, 50, 54, 70, 255);
        int center_x = x + w / 2;
        for (int i = 0; i < h; i += static_cast<int>(sy * 2)) {
            SDL_RenderDrawLine(renderer, center_x, y + i, center_x, y + i + static_cast<int>(sy));
        }

        int pw = static_cast<int>(sx * 3);
        auto [bx, by] = to_screen(ball_x_, ball_y_, x, y, sx, sy);
        int br = static_cast<int>(std::min(sx, sy) * 3);
        SDL_SetRenderDrawColor(renderer, 230, 230, 230, 255);
        for (int dy = -br; dy <= br; ++dy) {
            int dx_max = static_cast<int>(std::sqrt(static_cast<float>(br * br - dy * dy)));
            SDL_RenderDrawLine(renderer, bx - dx_max, by + dy, bx + dx_max, by + dy);
        }

        int pl = x + static_cast<int>(sx);
        int pr = x + w - static_cast<int>(sx * 2);
        int ph = static_cast<int>(PADDLE_H * sy);
        auto [_, py] = to_screen(0, paddle_y_, x, y, sx, sy);
        auto [__, ay] = to_screen(0, ai_paddle_y_, x, y, sx, sy);

        SDL_SetRenderDrawColor(renderer, 123, 159, 212, 255);
        SDL_Rect player_paddle = {pl, py, pw, ph};
        SDL_RenderFillRect(renderer, &player_paddle);

        SDL_SetRenderDrawColor(renderer, 212, 122, 122, 255);
        SDL_Rect ai_paddle_r = {pr, ay, pw, ph};
        SDL_RenderFillRect(renderer, &ai_paddle_r);
    }

private:
    static std::pair<int,int> to_screen(float gx, float gy, int x, int y, float sx, float sy) {
        return {static_cast<int>(x + gx * sx), static_cast<int>(y + gy * sy)};
    }

    void reset_ball() {
        ball_x_ = W / 2.0f;
        ball_y_ = H / 2.0f;
        ball_vx_ = (rounds_ % 2 == 0) ? 1.0f : -1.0f;
        ball_vy_ = (static_cast<float>(std::rand() % 100) / 100.0f - 0.5f) * 2.0f;
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(Pong, 8, 12, 500)
