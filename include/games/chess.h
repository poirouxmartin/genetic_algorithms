#pragma once

#include "game_interface.h"
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

namespace games {

class SimpleChess : public GameInterface {
    static constexpr int SIZE = 5;
    enum Piece : int { NONE=0, PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6 };

    std::array<int, SIZE * SIZE> board_{};
    int current_player_ = 1;
    int moves_ = 0;
    int max_moves_ = 60;
    bool done_ = false;
    bool white_won_ = false;
    bool black_won_ = false;

    static constexpr int PIECE_VALUES[] = {0, 1, 3, 3, 5, 9, 100};

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

    bool is_valid_move(int from, int to) const {
        int piece = board_[from];
        if (piece == 0) return false;
        int target = board_[to];
        int fr = from / SIZE, fc = from % SIZE;
        int tr = to / SIZE, tc = to % SIZE;
        int dr = std::abs(tr - fr), dc = std::abs(tc - fc);
        int abs_piece = std::abs(piece);

        if (target != 0 && ((piece > 0 && target > 0) || (piece < 0 && target < 0))) return false;

        if (abs_piece == PAWN) {
            int dir = (piece > 0) ? -1 : 1;
            return (tc == fc && dr == 1 && target == 0 && (tr - fr) == dir) ||
                   (tc == fc && dr == 2 && (piece > 0 ? fr == SIZE - 2 : fr == 1) && target == 0 &&
                    board_[(fr + dir) * SIZE + fc] == 0 && (tr - fr) == 2 * dir) ||
                   (dc == 1 && dr == 1 && target != 0 && (tr - fr) == dir);
        } else if (abs_piece == KNIGHT) {
            return (dr == 2 && dc == 1) || (dr == 1 && dc == 2);
        } else if (abs_piece == BISHOP) {
            return dr == dc && dr > 0 && is_path_clear(fr, fc, tr, tc);
        } else if (abs_piece == ROOK) {
            return (dr == 0 || dc == 0) && (dr + dc > 0) && is_path_clear(fr, fc, tr, tc);
        } else if (abs_piece == QUEEN) {
            return ((dr == 0 || dc == 0) || (dr == dc)) && (dr + dc > 0) && is_path_clear(fr, fc, tr, tc);
        } else if (abs_piece == KING) {
            return dr <= 1 && dc <= 1 && (dr + dc > 0);
        }
        return false;
    }

    std::vector<int> get_all_moves(int player) const {
        std::vector<int> moves;
        for (int from = 0; from < SIZE * SIZE; ++from) {
            int piece = board_[from];
            if (piece == 0) continue;
            if ((player == 1 && piece < 0) || (player == 2 && piece > 0)) continue;
            for (int to = 0; to < SIZE * SIZE; ++to) {
                if (to == from) continue;
                if (is_valid_move(from, to)) moves.push_back(from * SIZE * SIZE + to);
            }
        }
        return moves;
    }

    int pick_greedy_move(int player) const {
        auto moves = get_all_moves(player);
        if (moves.empty()) return -1;
        if (player == 2) {
            return moves[std::rand() % moves.size()];
        }
        int best_move = moves[0];
        int best_score = -99999;
        for (int m : moves) {
            int from = m / (SIZE * SIZE);
            int to = m % (SIZE * SIZE);
            int captured = board_[to];
            int score = PIECE_VALUES[std::abs(captured)] * 10;
            if (player == 2) score = -score;
            if (score > best_score) { best_score = score; best_move = m; }
        }
        return best_move;
    }

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
        c->white_won_ = white_won_;
        c->black_won_ = black_won_;
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
        white_won_ = false;
        black_won_ = false;
    }

    void step(int action) override {
        if (done_) return;

        if (current_player_ == 2) {
            int move = pick_greedy_move(2);
            if (move == -1) { white_won_ = true; done_ = true; return; }
            int from = move / (SIZE * SIZE);
            int to = move % (SIZE * SIZE);
            int piece = board_[from];
            int target = board_[to];
            if (std::abs(target) == KING) { black_won_ = true; done_ = true; return; }
            board_[to] = piece;
            board_[from] = 0;
            if (std::abs(piece) == PAWN && to / SIZE == SIZE - 1) board_[to] = -QUEEN;
            moves_++;
            current_player_ = 1;
            if (moves_ >= max_moves_) done_ = true;
            return;
        }

        int from = action / (SIZE * SIZE);
        int to = action % (SIZE * SIZE);
        if (from < 0 || from >= SIZE * SIZE || to < 0 || to >= SIZE * SIZE) return;
        int piece = board_[from];
        if (piece == 0 || piece < 0) return;
        if (!is_valid_move(from, to)) return;
        int target = board_[to];
        if (std::abs(target) == KING) { white_won_ = true; done_ = true; return; }
        board_[to] = piece;
        board_[from] = 0;
        if (std::abs(piece) == PAWN && to / SIZE == 0) board_[to] = QUEEN;
        moves_++;
        current_player_ = 2;
        if (moves_ >= max_moves_) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        if (white_won_) return 100.0f;
        if (black_won_) return -50.0f;
        float score = 0.0f;
        for (int v : board_) {
            if (v == 0) continue;
            float val = static_cast<float>(PIECE_VALUES[std::abs(v)]);
            float rank_bonus = 0.0f;
            int r = 0;
            for (int i = 0; i < SIZE * SIZE; ++i) {
                if (board_[i] == v) { r = i / SIZE; break; }
            }
            if (v > 0) rank_bonus = static_cast<float>(SIZE - 1 - r) * 0.5f;
            else rank_bonus = static_cast<float>(r) * 0.5f;
            score += (val + rank_bonus) * (v > 0 ? 1.0f : -1.0f);
        }
        return score * 2.0f;
    }

    int action_count() const override { return moves_; }

    void get_state(float* state) const override {
        for (int i = 0; i < SIZE * SIZE; ++i)
            state[i] = static_cast<float>(board_[i]) / 6.0f;
    }

    std::vector<int> get_valid_moves() const override {
        if (current_player_ != 1) return {};
        return get_all_moves(1);
    }

    std::string render_state() const override {
        std::string result;
        static const char* symbols[] = {".", "P", "N", "B", "R", "Q", "K"};
        result += "  abcde\n";
        for (int r = 0; r < SIZE; ++r) {
            result += static_cast<char>('1' + r);
            result += ' ';
            for (int c = 0; c < SIZE; ++c) {
                int v = board_[r * SIZE + c];
                if (v == 0) result += '.';
                else if (v > 0) result += static_cast<char>(std::toupper(symbols[v][0]));
                else result += static_cast<char>(std::tolower(symbols[-v][0]));
            }
            result += '\n';
        }
        result += white_won_ ? "White wins!" : black_won_ ? "Black wins!" : (current_player_ == 1 ? "White to move" : "Black to move");
        return result;
    }

    bool has_sdl_render() const override { return true; }

    void render_sdl(SDL_Renderer* renderer, int x, int y, int w, int h) const override {
        int cell = std::min(w / SIZE, h / SIZE);
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
                bool light = (r + c) % 2 == 0;
                if (light)
                    SDL_SetRenderDrawColor(renderer, 200, 195, 180, 255);
                else
                    SDL_SetRenderDrawColor(renderer, 110, 80, 50, 255);
                SDL_Rect sq = {sx, sy, cell, cell};
                SDL_RenderFillRect(renderer, &sq);

                int v = board_[r * SIZE + c];
                if (v != 0) {
                    bool is_white = v > 0;
                    int piece = std::abs(v);

                    if (is_white)
                        SDL_SetRenderDrawColor(renderer, 255, 255, 245, 255);
                    else
                        SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);

                    int cx2 = sx + cell / 2;
                    int cy2 = sy + cell / 2;
                    int sz = cell / 3;

                    switch (piece) {
                        case PAWN: {
                            int r2 = sz / 2;
                            for (int dy = -r2; dy <= r2; ++dy) {
                                int dx = static_cast<int>(std::sqrt(std::max(0, r2 * r2 - dy * dy)));
                                SDL_RenderDrawLine(renderer, cx2 - dx, cy2 + dy, cx2 + dx, cy2 + dy);
                            }
                            SDL_Rect base = {cx2 - sz / 2, cy2 + r2, sz, r2 / 2};
                            SDL_RenderFillRect(renderer, &base);
                            break;
                        }
                        case KNIGHT: {
                            SDL_Rect body = {cx2 - sz / 3, cy2 - sz / 2, sz * 2 / 3, sz};
                            SDL_RenderFillRect(renderer, &body);
                            SDL_Rect head = {cx2, cy2 - sz, sz / 2, sz / 2};
                            SDL_RenderFillRect(renderer, &head);
                            break;
                        }
                        case BISHOP: {
                            for (int i = 0; i < sz; ++i) {
                                int hw = (sz - i) / 2;
                                SDL_RenderDrawLine(renderer, cx2 - hw, cy2 - sz + i, cx2 + hw, cy2 - sz + i);
                            }
                            SDL_Rect base = {cx2 - sz / 2, cy2, sz, sz / 4};
                            SDL_RenderFillRect(renderer, &base);
                            break;
                        }
                        case ROOK: {
                            SDL_Rect tower = {cx2 - sz / 3, cy2 - sz / 2, sz * 2 / 3, sz};
                            SDL_RenderFillRect(renderer, &tower);
                            SDL_Rect batt1 = {cx2 - sz / 2, cy2 - sz / 2 - sz / 5, sz / 4, sz / 5};
                            SDL_Rect batt2 = {cx2 - sz / 8, cy2 - sz / 2 - sz / 5, sz / 4, sz / 5};
                            SDL_Rect batt3 = {cx2 + sz / 4, cy2 - sz / 2 - sz / 5, sz / 4, sz / 5};
                            SDL_RenderFillRect(renderer, &batt1);
                            SDL_RenderFillRect(renderer, &batt2);
                            SDL_RenderFillRect(renderer, &batt3);
                            break;
                        }
                        case QUEEN: {
                            for (int i = 0; i < sz; ++i) {
                                int hw = (sz - i) * 3 / 4;
                                SDL_RenderDrawLine(renderer, cx2 - hw, cy2 - sz + i, cx2 + hw, cy2 - sz + i);
                            }
                            for (int a = 0; a < 5; ++a) {
                                float ang = static_cast<float>(a) * 3.14159f / 4.0f - 3.14159f / 2.0f;
                                int px2 = cx2 + static_cast<int>(std::cos(ang) * sz / 3);
                                int py2 = cy2 - sz + static_cast<int>(std::sin(ang) * sz / 3);
                                SDL_Rect dot = {px2 - 2, py2 - 2, 4, 4};
                                SDL_RenderFillRect(renderer, &dot);
                            }
                            SDL_Rect base = {cx2 - sz / 2, cy2 + sz / 4, sz, sz / 4};
                            SDL_RenderFillRect(renderer, &base);
                            break;
                        }
                        case KING: {
                            SDL_Rect body = {cx2 - sz / 3, cy2 - sz / 2, sz * 2 / 3, sz};
                            SDL_RenderFillRect(renderer, &body);
                            SDL_SetRenderDrawColor(renderer,
                                is_white ? 255 : 30, is_white ? 255 : 30, is_white ? 245 : 30, 255);
                            SDL_RenderDrawLine(renderer, cx2, cy2 - sz, cx2, cy2 - sz / 2);
                            SDL_RenderDrawLine(renderer, cx2 - sz / 4, cy2 - sz * 3 / 4, cx2 + sz / 4, cy2 - sz * 3 / 4);
                            SDL_Rect base = {cx2 - sz / 2, cy2 + sz / 4, sz, sz / 4};
                            SDL_RenderFillRect(renderer, &base);
                            break;
                        }
                    }
                }
            }
        }

        SDL_SetRenderDrawColor(renderer, 80, 84, 100, 255);
        SDL_Rect border = {ox - 1, oy - 1, bw + 2, bh + 2};
        SDL_RenderDrawRect(renderer, &border);
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME_DISPLAY(SimpleChess, "Chess", 25, 20, 120)
