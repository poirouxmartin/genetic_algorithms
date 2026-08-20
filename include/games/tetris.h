#pragma once

#include "game_interface.h"
#include <array>
#include <algorithm>

namespace games {

class Tetris : public GameInterface {
    static constexpr int GRID_W = 8;
    static constexpr int GRID_H = 12;
    static constexpr int MAX_FRAMES = 500;
    static constexpr int NUM_PIECES = 4;

    struct Piece { int cells[4][2]; int type; };

    std::array<int, GRID_W * GRID_H> board_{};
    Piece current_;
    int piece_x_, piece_y_;
    int score_;
    int frames_;
    bool done_;
    int lines_cleared_;

    static const int SHAPES[NUM_PIECES][4][2];

    int next_piece_idx_;

    void spawn_piece() {
        next_piece_idx_ = (next_piece_idx_ + 1) % NUM_PIECES;
        current_ = { {{SHAPES[next_piece_idx_][0][0], SHAPES[next_piece_idx_][0][1]},
                       {SHAPES[next_piece_idx_][1][0], SHAPES[next_piece_idx_][1][1]},
                       {SHAPES[next_piece_idx_][2][0], SHAPES[next_piece_idx_][2][1]},
                       {SHAPES[next_piece_idx_][3][0], SHAPES[next_piece_idx_][3][1]}},
                     next_piece_idx_ };
        piece_x_ = GRID_W / 2;
        piece_y_ = 0;

        for (auto& cell : current_.cells) {
            int cx = piece_x_ + cell[0], cy = piece_y_ + cell[1];
            if (cy >= 0 && cy < GRID_H && cx >= 0 && cx < GRID_W) {
                if (board_[cy * GRID_W + cx] != 0) { done_ = true; return; }
            }
        }
    }

    bool check_pos(int dx, int dy) const {
        for (auto& cell : current_.cells) {
            int nx = piece_x_ + cell[0] + dx;
            int ny = piece_y_ + cell[1] + dy;
            if (nx < 0 || nx >= GRID_W || ny >= GRID_H) return false;
            if (ny >= 0 && board_[ny * GRID_W + nx] != 0) return false;
        }
        return true;
    }

    void lock_piece() {
        for (auto& cell : current_.cells) {
            int cx = piece_x_ + cell[0], cy = piece_y_ + cell[1];
            if (cy >= 0 && cy < GRID_H && cx >= 0 && cx < GRID_W)
                board_[cy * GRID_W + cx] = 1;
        }
        clear_lines();
    }

    void clear_lines() {
        int cleared = 0;
        for (int r = GRID_H - 1; r >= 0; --r) {
            bool full = true;
            for (int c = 0; c < GRID_W; ++c) {
                if (board_[r * GRID_W + c] == 0) { full = false; break; }
            }
            if (full) {
                cleared++;
                for (int rr = r; rr > 0; --rr)
                    for (int c = 0; c < GRID_W; ++c)
                        board_[rr * GRID_W + c] = board_[(rr - 1) * GRID_W + c];
                for (int c = 0; c < GRID_W; ++c) board_[c] = 0;
                r++;
            }
        }
        if (cleared > 0) {
            lines_cleared_ += cleared;
            score_ += cleared * cleared * 10;
        }
    }

public:
    Tetris() { next_piece_idx_ = 0; reset(); }

    std::string name() const override { return "Tetris"; }
    int action_space() const override { return 4; }

    void reset() override {
        board_.fill(0);
        score_ = 0;
        frames_ = 0;
        done_ = false;
        lines_cleared_ = 0;
        next_piece_idx_ = 0;
        spawn_piece();
    }

    void step(int action) override {
        if (done_) return;
        frames_++;

        if (action == 0) {
            if (check_pos(-1, 0)) piece_x_--;
        } else if (action == 1) {
            if (check_pos(1, 0)) piece_x_++;
        } else if (action == 2) {
            Piece rotated = current_;
            for (auto& cell : rotated.cells) {
                int tmp = cell[0];
                cell[0] = -cell[1];
                cell[1] = tmp;
            }
            Piece saved = current_;
            current_ = rotated;
            if (!check_pos(0, 0)) current_ = saved;
        }

        if (check_pos(0, 1)) {
            piece_y_++;
        } else {
            lock_piece();
            if (!done_) spawn_piece();
        }

        if (frames_ >= MAX_FRAMES) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        float height_penalty = 0;
        for (int c = 0; c < GRID_W; ++c) {
            for (int r = 0; r < GRID_H; ++r) {
                if (board_[r * GRID_W + c] != 0) {
                    height_penalty += (GRID_H - r);
                    break;
                }
            }
        }
        return static_cast<float>(score_)
             + static_cast<float>(lines_cleared_) * 5.0f
             - height_penalty * 0.1f
             + static_cast<float>(frames_) * 0.01f;
    }

    int action_count() const override { return frames_; }

    void get_state(float* state) const override {
        state[0] = static_cast<float>(score_) / 100.0f;
        state[1] = static_cast<float>(lines_cleared_) / 20.0f;
        state[2] = static_cast<float>(piece_x_) / GRID_W;
        state[3] = static_cast<float>(piece_y_) / GRID_H;

        int max_h = 0;
        for (int c = 0; c < GRID_W; ++c) {
            for (int r = 0; r < GRID_H; ++r) {
                if (board_[r * GRID_W + c] != 0) {
                    int h = GRID_H - r;
                    if (h > max_h) max_h = h;
                    break;
                }
            }
        }
        state[4] = static_cast<float>(max_h) / GRID_H;

        int holes = 0;
        for (int c = 0; c < GRID_W; ++c) {
            bool block_seen = false;
            for (int r = 0; r < GRID_H; ++r) {
                if (board_[r * GRID_W + c] != 0) block_seen = true;
                else if (block_seen) holes++;
            }
        }
        state[5] = static_cast<float>(holes) / 30.0f;
    }

    std::string render_state() const override {
        std::string result;
        for (int r = 0; r < GRID_H; ++r) {
            for (int c = 0; c < GRID_W; ++c) {
                bool is_piece = false;
                for (auto& cell : current_.cells) {
                    if (c == piece_x_ + cell[0] && r == piece_y_ + cell[1]) {
                        is_piece = true;
                        break;
                    }
                }
                if (is_piece) result += 'X';
                else if (board_[r * GRID_W + c] != 0) result += '#';
                else result += '.';
            }
            result += '\n';
        }
        return result;
    }
};

inline const int Tetris::SHAPES[NUM_PIECES][4][2] = {
    {{0,0},{1,0},{0,1},{1,1}},   // square
    {{0,0},{1,0},{2,0},{3,0}},   // line
    {{0,0},{1,0},{2,0},{2,1}},   // L-shape
    {{0,0},{1,0},{1,1},{2,1}},   // S-shape
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(Tetris, 6, {10}, 500)
