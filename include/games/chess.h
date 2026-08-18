#pragma once

#include "game_interface.h"
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

namespace games {

class SimpleChess : public GameInterface {
    static constexpr int SIZE = 5;
    enum Piece : int { NONE=0, PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6 };

    std::array<int, SIZE * SIZE> board_{};
    int current_player_ = 1;
    int moves_ = 0;
    int max_moves_ = 50;
    bool done_ = false;
    bool king_captured_ = false;

public:
    SimpleChess() { reset(); }

    std::string name() const override { return "SimpleChess"; }
    int action_space() const override { return SIZE * SIZE * SIZE * SIZE; }
    bool uses_scoring() const override { return true; }
    std::unique_ptr<GameInterface> clone() const override {
        auto c = std::make_unique<SimpleChess>();
        c->board_ = board_;
        c->current_player_ = current_player_;
        c->moves_ = moves_;
        c->done_ = done_;
        c->king_captured_ = king_captured_;
        return c;
    }

    void reset() override {
        board_.fill(0);
        int back[] = {ROOK, KNIGHT, QUEEN, KING, BISHOP};
        for (int c = 0; c < SIZE; ++c) {
            board_[c] = back[c];
            board_[SIZE + c] = PAWN;
            board_[(SIZE - 1) * SIZE + c] = -back[c];
            board_[(SIZE - 2) * SIZE + c] = -PAWN;
        }
        current_player_ = 1;
        moves_ = 0;
        done_ = false;
        king_captured_ = false;
    }

    void step(int action) override {
        if (done_) return;
        int from = action / (SIZE * SIZE);
        int to = action % (SIZE * SIZE);
        if (from < 0 || from >= SIZE * SIZE || to < 0 || to >= SIZE * SIZE) return;

        int piece = board_[from];
        if (piece == 0) return;
        if ((current_player_ == 1 && piece < 0) || (current_player_ == 2 && piece > 0)) return;

        int fr = from / SIZE, fc = from % SIZE;
        int tr = to / SIZE, tc = to % SIZE;
        int target = board_[to];

        if (target != 0 && ((current_player_ == 1 && target > 0) || (current_player_ == 2 && target < 0))) return;

        int dr = std::abs(tr - fr), dc = std::abs(tc - fc);
        bool valid = false;
        int abs_piece = std::abs(piece);

        if (abs_piece == PAWN) {
            int dir = (piece > 0) ? -1 : 1;
            valid = (tc == fc && dr == 1 && target == 0 && (tr - fr) == dir) ||
                    (tc == fc && dr == 2 && (piece > 0 ? fc == SIZE - 2 : fc == 1) && target == 0 &&
                     board_[(fr + dir) * SIZE + fc] == 0 && (tr - fr) == 2 * dir) ||
                    (dc == 1 && dr == 1 && target != 0 && (tr - fr) == dir);
        } else if (abs_piece == KNIGHT) {
            valid = (dr == 2 && dc == 1) || (dr == 1 && dc == 2);
        } else if (abs_piece == BISHOP) {
            valid = dr == dc && dr > 0 && is_path_clear(fr, fc, tr, tc);
        } else if (abs_piece == ROOK) {
            valid = (dr == 0 || dc == 0) && (dr + dc > 0) && is_path_clear(fr, fc, tr, tc);
        } else if (abs_piece == QUEEN) {
            valid = ((dr == 0 || dc == 0) || (dr == dc)) && (dr + dc > 0) && is_path_clear(fr, fc, tr, tc);
        } else if (abs_piece == KING) {
            valid = dr <= 1 && dc <= 1 && (dr + dc > 0);
        }

        if (!valid) return;
        if (std::abs(target) == KING) { king_captured_ = true; done_ = true; }

        board_[to] = piece;
        board_[from] = 0;
        if (abs_piece == PAWN && (tr == 0 || tr == SIZE - 1))
            board_[to] = (piece > 0) ? QUEEN : -QUEEN;

        moves_++;
        current_player_ = (current_player_ == 1) ? 2 : 1;
        if (moves_ >= max_moves_) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        if (king_captured_) return 100.0f;
        float score = 0.0f;
        for (int v : board_) score += v;
        return score - static_cast<float>(moves_) * 0.1f;
    }

    int action_count() const override { return moves_; }

    void get_state(float* state) const override {
        for (int i = 0; i < SIZE * SIZE; ++i)
            state[i] = static_cast<float>(board_[i]) / 6.0f;
    }

    std::vector<int> get_valid_moves() const override {
        std::vector<int> moves;
        for (int from = 0; from < SIZE * SIZE; ++from) {
            int piece = board_[from];
            if (piece == 0) continue;
            if ((current_player_ == 1 && piece < 0) || (current_player_ == 2 && piece > 0)) continue;
            for (int to = 0; to < SIZE * SIZE; ++to) {
                if (to == from) continue;
                int fr = from / SIZE, fc = from % SIZE;
                int tr = to / SIZE, tc = to % SIZE;
                int target = board_[to];
                if (target != 0 && ((current_player_ == 1 && target > 0) || (current_player_ == 2 && target < 0))) continue;

                int dr = std::abs(tr - fr), dc = std::abs(tc - fc);
                bool valid = false;
                int abs_piece = std::abs(piece);

                if (abs_piece == PAWN) {
                    int dir = (piece > 0) ? -1 : 1;
                    valid = (tc == fc && dr == 1 && target == 0 && (tr - fr) == dir) ||
                            (tc == fc && dr == 2 && (piece > 0 ? fc == SIZE - 2 : fc == 1) && target == 0 &&
                             board_[(fr + dir) * SIZE + fc] == 0 && (tr - fr) == 2 * dir) ||
                            (dc == 1 && dr == 1 && target != 0 && (tr - fr) == dir);
                } else if (abs_piece == KNIGHT) {
                    valid = (dr == 2 && dc == 1) || (dr == 1 && dc == 2);
                } else if (abs_piece == BISHOP) {
                    valid = dr == dc && dr > 0 && is_path_clear(fr, fc, tr, tc);
                } else if (abs_piece == ROOK) {
                    valid = (dr == 0 || dc == 0) && (dr + dc > 0) && is_path_clear(fr, fc, tr, tc);
                } else if (abs_piece == QUEEN) {
                    valid = ((dr == 0 || dc == 0) || (dr == dc)) && (dr + dc > 0) && is_path_clear(fr, fc, tr, tc);
                } else if (abs_piece == KING) {
                    valid = dr <= 1 && dc <= 1 && (dr + dc > 0);
                }

                if (valid) moves.push_back(from * SIZE * SIZE + to);
            }
        }
        return moves;
    }

    std::string render_state() const override {
        std::string result;
        static const char* symbols[] = {".", "P", "N", "B", "R", "Q", "K"};
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                int v = board_[r * SIZE + c];
                if (v == 0) result += '.';
                else if (v > 0) result += symbols[v];
                else result += static_cast<char>(std::tolower(symbols[-v][0]));
            }
            result += '\n';
        }
        return result;
    }

private:
    bool is_path_clear(int fr, int fc, int tr, int tc) const {
        int dr = (tr > fr) ? 1 : (tr < fr) ? -1 : 0;
        int dc = (tc > fc) ? 1 : (tc < fc) ? -1 : 0;
        int r = fr + dr, c = fc + dc;
        while (r != tr || c != tc) {
            if (board_[r * SIZE + c] != 0) return false;
            r += dr; c += dc;
        }
        return true;
    }
};

} // namespace games
