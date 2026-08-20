#pragma once

#include "game_interface.h"
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

namespace games {

class Game2048 : public GameInterface {
    static constexpr int SIZE = 4;
    static constexpr int TOTAL = SIZE * SIZE;

    std::array<int, TOTAL> board_{};
    int moves_ = 0;
    bool done_ = false;
    int max_tile_ = 0;
    int score_ = 0;
    int merges_ = 0;

    int& at(int r, int c) { return board_[r * SIZE + c]; }
    int at(int r, int c) const { return board_[r * SIZE + c]; }

    void slide_row_left(int row) {
        int vals[TOTAL];
        int cnt = 0;
        for (int c = 0; c < SIZE; ++c) {
            int v = at(row, c);
            if (v != 0) vals[cnt++] = v;
        }
        int out = 0;
        for (int i = 0; i < cnt; ) {
            if (i + 1 < cnt && vals[i] == vals[i + 1]) {
                int merged = vals[i] * 2;
                score_ += merged;
                merges_++;
                if (merged > max_tile_) max_tile_ = merged;
                at(row, out++) = merged;
                i += 2;
            } else {
                at(row, out++) = vals[i];
                i++;
            }
        }
        for (int c = out; c < SIZE; ++c) at(row, c) = 0;
    }

    void rotate_cw() {
        std::array<int, TOTAL> tmp = board_;
        for (int r = 0; r < SIZE; ++r)
            for (int c = 0; c < SIZE; ++c)
                board_[c * SIZE + (SIZE - 1 - r)] = tmp[r * SIZE + c];
    }

    bool boards_equal(const std::array<int, TOTAL>& a, const std::array<int, TOTAL>& b) const {
        for (int i = 0; i < TOTAL; ++i)
            if (a[i] != b[i]) return false;
        return true;
    }

    bool slide_left() {
        std::array<int, TOTAL> before = board_;
        for (int r = 0; r < SIZE; ++r) slide_row_left(r);
        return !boards_equal(before, board_);
    }

    void add_random_tile() {
        std::vector<int> empty;
        for (int i = 0; i < TOTAL; ++i)
            if (board_[i] == 0) empty.push_back(i);
        if (empty.empty()) return;
        int idx = empty[std::rand() % static_cast<int>(empty.size())];
        board_[idx] = (std::rand() % 10 == 0) ? 4 : 2;
    }

    bool has_moves() const {
        for (int i = 0; i < TOTAL; ++i) {
            if (board_[i] == 0) return true;
            int r = i / SIZE, c = i % SIZE;
            if (c + 1 < SIZE && board_[i] == board_[r * SIZE + c + 1]) return true;
            if (r + 1 < SIZE && board_[i] == board_[(r + 1) * SIZE + c]) return true;
        }
        return false;
    }

public:
    Game2048() { reset(); }

    std::string name() const override { return "2048"; }
    int action_space() const override { return 4; }
    bool uses_scoring() const override { return false; }

    std::unique_ptr<GameInterface> clone() const override {
        auto c = std::make_unique<Game2048>();
        c->board_ = board_;
        c->moves_ = moves_;
        c->done_ = done_;
        c->max_tile_ = max_tile_;
        c->score_ = score_;
        c->merges_ = merges_;
        return c;
    }

    void reset() override {
        board_.fill(0);
        moves_ = 0;
        done_ = false;
        max_tile_ = 0;
        score_ = 0;
        merges_ = 0;
        add_random_tile();
        add_random_tile();
    }

    void step(int action) override {
        if (done_) return;
        if (action < 0 || action >= 4) return;

        static const int rots[] = {3, 2, 1, 0};
        int rotations = rots[action];
        for (int i = 0; i < rotations; ++i) rotate_cw();
        bool changed = slide_left();
        for (int i = 0; i < (4 - rotations) % 4; ++i) rotate_cw();

        if (changed) {
            moves_++;
            add_random_tile();
        }
        if (!has_moves()) done_ = true;
    }

    bool is_done() const override { return done_; }

    std::vector<int> get_valid_moves() const override {
        if (done_) return {};
        return {0, 1, 2, 3};
    }

    float fitness() const override {
        float log_score = 0.0f;
        if (max_tile_ > 0) log_score = std::log2(static_cast<float>(max_tile_));
        return log_score * 100.0f + static_cast<float>(score_) * 0.01f
             + static_cast<float>(merges_) * 5.0f;
    }

    int action_count() const override { return moves_; }

    void get_state(float* state) const override {
        float max_log = (max_tile_ > 0) ? std::log2(static_cast<float>(max_tile_)) : 1.0f;
        for (int i = 0; i < TOTAL; ++i) {
            if (board_[i] > 0)
                state[i] = std::log2(static_cast<float>(board_[i])) / max_log;
            else
                state[i] = 0.0f;
        }

        int empty = 0;
        int max_r = 0, max_c = 0;
        int max_v = 0;
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                if (board_[r * SIZE + c] == 0) empty++;
                if (board_[r * SIZE + c] > max_v) {
                    max_v = board_[r * SIZE + c];
                    max_r = r; max_c = c;
                }
            }
        }
        state[16] = static_cast<float>(empty) / TOTAL;
        state[17] = static_cast<float>(max_r) / (SIZE - 1);
        state[18] = static_cast<float>(max_c) / (SIZE - 1);
        state[19] = static_cast<float>(moves_) / 1000.0f;
    }

    std::string render_state() const override {
        std::string result;
        for (int r = 0; r < SIZE; ++r) {
            result += "+------+------+------+------+\n";
            for (int c = 0; c < SIZE; ++c) {
                result += "|";
                int v = board_[r * SIZE + c];
                if (v == 0) result += "      ";
                else {
                    std::string s = std::to_string(v);
                    int pad = 6 - static_cast<int>(s.size());
                    result += std::string(pad / 2, ' ') + s + std::string(pad - pad / 2, ' ');
                }
            }
            result += "|\n";
        }
        result += "+------+------+------+------+\n";
        result += "Score:" + std::to_string(score_) + " Moves:" + std::to_string(moves_);
        result += " Max:" + std::to_string(max_tile_);
        return result;
    }

    bool has_sdl_render() const override { return true; }

    static void draw_digit(SDL_Renderer* r, int d, int cx, int cy, int sz, SDL_Color col) {
        static const uint8_t digits[10] = {
            0x7B, 0x42, 0x7D, 0x63, 0x19,
            0x39, 0x79, 0x43, 0x7B, 0x7B
        };
        static const int glyph[10][15] = {
            {1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1},
            {0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1},
            {1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1},
            {1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1},
            {1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1},
            {1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1},
            {1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1},
            {1,1,1, 0,0,1, 0,1,0, 0,1,0, 0,1,0},
            {1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1},
            {1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1}
        };
        int pw = sz / 3, ph = sz / 5;
        SDL_SetRenderDrawColor(r, col.r, col.g, col.b, col.a);
        for (int py = 0; py < 5; ++py)
            for (int px = 0; px < 3; ++px)
                if (glyph[d][py * 3 + px]) {
                    SDL_Rect rect = {cx - sz / 2 + px * pw, cy - sz / 2 + py * ph, pw - 1, ph - 1};
                    SDL_RenderFillRect(r, &rect);
                }
    }

    static void draw_number(SDL_Renderer* r, int val, int cx, int cy, int sz, SDL_Color col) {
        if (val == 0) return;
        char buf[16];
        int len = 0;
        int tmp = val;
        while (tmp > 0) { buf[len++] = '0' + tmp % 10; tmp /= 10; }
        int total_w = len * (sz + 2);
        int start_x = cx - total_w / 2;
        for (int i = len - 1; i >= 0; --i) {
            int d = buf[i] - '0';
            int dx = start_x + (len - 1 - i) * (sz + 2) + sz / 2;
            draw_digit(r, d, dx, cy, sz, col);
        }
    }

    void render_sdl(SDL_Renderer* renderer, int x, int y, int w, int h) const override {
        int cell = std::min(w, h) / SIZE;
        int bw = cell * SIZE;
        int bh = cell * SIZE;
        int ox = x + (w - bw) / 2;
        int oy = y + (h - bh) / 2;

        SDL_SetRenderDrawColor(renderer, 42, 45, 62, 255);
        SDL_Rect bg = {x, y, w, h};
        SDL_RenderFillRect(renderer, &bg);

        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                int sx = ox + c * cell;
                int sy = oy + r * cell;
                int v = board_[r * SIZE + c];

                if (v == 0)
                    SDL_SetRenderDrawColor(renderer, 55, 58, 75, 255);
                else if (v <= 4)
                    SDL_SetRenderDrawColor(renderer, 107, 191, 160, 255);
                else if (v <= 16)
                    SDL_SetRenderDrawColor(renderer, 123, 159, 212, 255);
                else if (v <= 64)
                    SDL_SetRenderDrawColor(renderer, 191, 160, 123, 255);
                else if (v <= 256)
                    SDL_SetRenderDrawColor(renderer, 212, 184, 122, 255);
                else if (v <= 1024)
                    SDL_SetRenderDrawColor(renderer, 212, 140, 122, 255);
                else
                    SDL_SetRenderDrawColor(renderer, 212, 122, 122, 255);

                SDL_Rect sq = {sx + 2, sy + 2, cell - 4, cell - 4};
                SDL_RenderFillRect(renderer, &sq);

                if (v > 0) {
                    SDL_Color col = {226, 228, 234, 255};
                    if (v >= 128) col = {27, 30, 46, 255};
                    draw_number(renderer, v, sx + cell / 2, sy + cell / 2, cell / 3, col);
                }
            }
        }

        SDL_SetRenderDrawColor(renderer, 80, 84, 100, 255);
        SDL_Rect border = {ox - 1, oy - 1, bw + 2, bh + 2};
        SDL_RenderDrawRect(renderer, &border);

        int info_y = oy + bh + 10;
        int info_h = std::max(16, cell / 2);
        int half_w = bw / 2 - 5;

        SDL_SetRenderDrawColor(renderer, 55, 58, 75, 255);
        SDL_Rect score_bg = {ox, info_y, half_w, info_h};
        SDL_RenderFillRect(renderer, &score_bg);
        SDL_SetRenderDrawColor(renderer, 107, 191, 160, 255);
        SDL_Rect score_bar = {ox, info_y, half_w * std::min(1.0f, score_ / 50000.0f), info_h};
        SDL_RenderFillRect(renderer, &score_bar);
        draw_number(renderer, score_, ox + half_w / 2, info_y + info_h / 2, info_h * 2 / 3, {226, 228, 234, 255});

        SDL_SetRenderDrawColor(renderer, 55, 58, 75, 255);
        SDL_Rect max_bg = {ox + half_w + 10, info_y, half_w, info_h};
        SDL_RenderFillRect(renderer, &max_bg);
        SDL_SetRenderDrawColor(renderer, 212, 184, 122, 255);
        SDL_Rect max_bar = {ox + half_w + 10, info_y,
                           static_cast<int>(half_w * std::min(1.0f, std::log2(static_cast<float>(std::max(max_tile_, 1))) / 11.0f)), info_h};
        SDL_RenderFillRect(renderer, &max_bar);
        draw_number(renderer, max_tile_, ox + half_w + 10 + half_w / 2, info_y + info_h / 2, info_h * 2 / 3, {226, 228, 234, 255});
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME_DISPLAY(Game2048, "2048", 20, {16}, 2000)
