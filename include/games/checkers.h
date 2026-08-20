#pragma once

#include "game_interface.h"
#include <array>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

namespace games {

class Checkers : public GameInterface {
    static constexpr int SIZE = 8;
    static constexpr int DARK_COUNT = 32;

    enum Piece : int { NONE=0, MAN=1, KING=2 };

    std::array<int, SIZE * SIZE> board_{};
    int current_player_ = 1;
    int moves_ = 0;
    bool done_ = false;
    bool p1_won_ = false;
    bool p2_won_ = false;

    struct Move { int from, to, captured; bool multi; };
    std::vector<Move> jumps_;
    std::vector<Move> slides_;
    bool jumps_mandatory_ = true;

    static int dark_index(int r, int c) {
        if ((r + c) % 2 == 0) return -1;
        return (r * 4) + (c / 2);
    }

    void find_moves_for(int player) {
        jumps_.clear();
        slides_.clear();
        int dir = (player == 1) ? -1 : 1;

        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                int p = board_[r * SIZE + c];
                if (p == 0 || (player == 1 && p < 0) || (player == 2 && p > 0)) continue;

                int dr[] = {dir, dir};
                int dc[] = {-1, 1};
                for (int d = 0; d < 2; ++d) {
                    int nr = r + dr[d], nc = c + dc[d];
                    if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;
                    int target = board_[nr * SIZE + nc];
                    if (target == 0) {
                        slides_.push_back({r * SIZE + c, nr * SIZE + nc, -1, false});
                    } else if ((player == 1 && target < 0) || (player == 2 && target > 0)) {
                        int jr = nr + dr[d], jc = nc + dc[d];
                        if (jr >= 0 && jr < SIZE && jc >= 0 && jc < SIZE && board_[jr * SIZE + jc] == 0) {
                            jumps_.push_back({r * SIZE + c, jr * SIZE + jc, nr * SIZE + nc, false});
                        }
                    }
                }

                if (std::abs(p) == KING) {
                    int dr2[] = {-dir, -dir};
                    for (int d = 0; d < 2; ++d) {
                        int nr = r + dr2[d], nc = c + dc[d];
                        if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;
                        int target = board_[nr * SIZE + nc];
                        if (target == 0) {
                            slides_.push_back({r * SIZE + c, nr * SIZE + nc, -1, false});
                        } else if ((player == 1 && target < 0) || (player == 2 && target > 0)) {
                            int jr = nr + dr2[d], jc = nc + dc[d];
                            if (jr >= 0 && jr < SIZE && jc >= 0 && jc < SIZE && board_[jr * SIZE + jc] == 0) {
                                jumps_.push_back({r * SIZE + c, jr * SIZE + jc, nr * SIZE + nc, false});
                            }
                        }
                    }
                }
            }
        }
    }

    bool has_any_jump(int player) const {
        int dir = (player == 1) ? -1 : 1;
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                int p = board_[r * SIZE + c];
                if (p == 0 || (player == 1 && p < 0) || (player == 2 && p > 0)) continue;

                int dirs_r[] = {dir, dir, -dir, -dir};
                int dirs_c[] = {-1, 1, -1, 1};
                for (int d = 0; d < 4; ++d) {
                    if (std::abs(p) != KING && dirs_r[d] == -dir) continue;
                    int nr = r + dirs_r[d], nc = c + dirs_c[d];
                    if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) continue;
                    int target = board_[nr * SIZE + nc];
                    if (target != 0 && ((player == 1 && target < 0) || (player == 2 && target > 0))) {
                        int jr = nr + dirs_r[d], jc = nc + dirs_c[d];
                        if (jr >= 0 && jr < SIZE && jc >= 0 && jc < SIZE && board_[jr * SIZE + jc] == 0)
                            return true;
                    }
                }
            }
        }
        return false;
    }

    void make_move(int from, int to, int captured) {
        int piece = board_[from];
        board_[from] = 0;
        if (captured >= 0) board_[captured] = 0;
        board_[to] = piece;

        int promo_row = (piece > 0) ? 0 : SIZE - 1;
        if (std::abs(piece) == MAN && to / SIZE == promo_row)
            board_[to] = (piece > 0) ? KING : -KING;

        current_player_ = (current_player_ == 1) ? 2 : 1;
        moves_++;

        int p1 = 0, p2 = 0;
        for (int v : board_) {
            if (v > 0) p1++;
            else if (v < 0) p2++;
        }
        if (p1 == 0) { p2_won_ = true; done_ = true; }
        else if (p2 == 0) { p1_won_ = true; done_ = true; }
        else if (moves_ >= 200) done_ = true;
        else {
            find_moves_for(current_player_);
            if (jumps_.empty() && slides_.empty()) {
                if (current_player_ == 1) p2_won_ = true;
                else p1_won_ = true;
                done_ = true;
            }
        }
    }

    int pick_ai_move() {
        find_moves_for(current_player_);
        if (jumps_mandatory_ && !jumps_.empty()) {
            return std::rand() % static_cast<int>(jumps_.size());
        }
        if (!slides_.empty()) {
            return std::rand() % static_cast<int>(slides_.size());
        }
        if (!jumps_.empty()) {
            return std::rand() % static_cast<int>(jumps_.size());
        }
        return -1;
    }

public:
    Checkers() { reset(); }

    std::string name() const override { return "Checkers"; }
    int action_space() const override { return DARK_COUNT * DARK_COUNT; }
    bool uses_scoring() const override { return true; }

    std::unique_ptr<GameInterface> clone() const override {
        auto c = std::make_unique<Checkers>();
        c->board_ = board_;
        c->current_player_ = current_player_;
        c->moves_ = moves_;
        c->done_ = done_;
        c->p1_won_ = p1_won_;
        c->p2_won_ = p2_won_;
        return c;
    }

    void reset() override {
        board_.fill(0);
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                if ((r + c) % 2 == 0) continue;
                if (r < 3) board_[r * SIZE + c] = -MAN;
                else if (r > 4) board_[r * SIZE + c] = MAN;
            }
        }
        current_player_ = 1;
        moves_ = 0;
        done_ = false;
        p1_won_ = false;
        p2_won_ = false;
    }

    void step(int action) override {
        if (done_) return;

        if (current_player_ == 2) {
            int idx = pick_ai_move();
            if (idx == -1) { p1_won_ = true; done_ = true; return; }
            if (!jumps_mandatory_ || jumps_.empty()) {
                if (idx < static_cast<int>(slides_.size())) {
                    auto& m = slides_[idx];
                    make_move(m.from, m.to, m.captured);
                } else {
                    auto& m = jumps_[idx - static_cast<int>(slides_.size())];
                    make_move(m.from, m.to, m.captured);
                }
            } else {
                auto& m = jumps_[idx];
                make_move(m.from, m.to, m.captured);
            }
            return;
        }

        find_moves_for(1);
        bool must_jump = !jumps_.empty();

        int from = action / DARK_COUNT;
        int to = action % DARK_COUNT;
        if (from == to) return;

        if (must_jump) {
            for (auto& m : jumps_) {
                if (m.from == from && m.to == to) {
                    make_move(m.from, m.to, m.captured);
                    return;
                }
            }
        } else {
            for (auto& m : slides_) {
                if (m.from == from && m.to == to) {
                    make_move(m.from, m.to, -1);
                    return;
                }
            }
        }
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        if (p1_won_) return 100.0f;
        if (p2_won_) return -100.0f;

        float score = 0.0f;
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                int v = board_[r * SIZE + c];
                if (v == 0) continue;
                float val = (std::abs(v) == KING) ? 3.0f : 1.0f;
                float advance = 0.0f;
                if (v > 0) advance = static_cast<float>(SIZE - 1 - r) * 0.15f;
                else advance = static_cast<float>(r) * 0.15f;
                score += (val + advance) * (v > 0 ? 1.0f : -1.0f);
            }
        }
        return score;
    }

    int action_count() const override { return moves_; }

    void get_state(float* state) const override {
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                int v = board_[r * SIZE + c];
                float val = 0.0f;
                if (v == MAN) val = 0.5f;
                else if (v == KING) val = 1.0f;
                else if (v == -MAN) val = -0.5f;
                else if (v == -KING) val = -1.0f;
                state[r * SIZE + c] = val;
            }
        }
        int p1 = 0, p2 = 0, k1 = 0, k2 = 0;
        for (int v : board_) {
            if (v > 0) { p1++; if (v == KING) k1++; }
            else if (v < 0) { p2++; if (v == -KING) k2++; }
        }
        state[64] = static_cast<float>(p1) / 12.0f;
        state[65] = static_cast<float>(p2) / 12.0f;
        state[66] = static_cast<float>(k1) / 12.0f;
        state[67] = static_cast<float>(k2) / 12.0f;
        state[68] = static_cast<float>(current_player_ == 1 ? 1.0f : 0.0f);
    }

    std::vector<int> get_valid_moves() const override {
        if (current_player_ != 1) return {};
        const_cast<Checkers*>(this)->find_moves_for(1);
        std::vector<int> valid;
        bool must_jump = !jumps_.empty();
        if (must_jump) {
            for (auto& m : jumps_)
                valid.push_back(m.from * DARK_COUNT + m.to);
        } else {
            for (auto& m : slides_)
                valid.push_back(m.from * DARK_COUNT + m.to);
        }
        return valid;
    }

    std::string render_state() const override {
        std::string result;
        result += "  a b c d e f g h\n";
        for (int r = 0; r < SIZE; ++r) {
            result += static_cast<char>('1' + r);
            result += ' ';
            for (int c = 0; c < SIZE; ++c) {
                int v = board_[r * SIZE + c];
                if (v == 0) {
                    result += ((r + c) % 2 == 0) ? ' ' : '.';
                } else if (v == MAN) result += 'o';
                else if (v == KING) result += 'O';
                else if (v == -MAN) result += 'x';
                else if (v == -KING) result += 'X';
            }
            result += '\n';
        }
        result += p1_won_ ? "You win!" : p2_won_ ? "You lose!" :
                  (current_player_ == 1 ? "Your turn" : "Opponent's turn");
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
                    SDL_SetRenderDrawColor(renderer, 200, 190, 160, 255);
                else
                    SDL_SetRenderDrawColor(renderer, 90, 60, 35, 255);
                SDL_Rect sq = {sx, sy, cell, cell};
                SDL_RenderFillRect(renderer, &sq);

                int v = board_[r * SIZE + c];
                if (v != 0) {
                    bool is_p1 = v > 0;
                    bool is_king = std::abs(v) == KING;

                    int cx2 = sx + cell / 2;
                    int cy2 = sy + cell / 2;
                    int radius = cell * 3 / 8;

                    if (is_p1)
                        SDL_SetRenderDrawColor(renderer, 220, 220, 210, 255);
                    else
                        SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);

                    for (int dy = -radius; dy <= radius; ++dy) {
                        int dx = static_cast<int>(std::sqrt(std::max(0, radius * radius - dy * dy)));
                        SDL_RenderDrawLine(renderer, cx2 - dx, cy2 + dy, cx2 + dx, cy2 + dy);
                    }

                    if (is_king) {
                        SDL_SetRenderDrawColor(renderer, is_p1 ? 180 : 80,
                                               is_p1 ? 170 : 80,
                                               is_p1 ? 100 : 40, 255);
                        SDL_Rect crown = {cx2 - radius / 2, cy2 - radius / 2, radius, radius / 2};
                        SDL_RenderFillRect(renderer, &crown);
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
REGISTER_GAME(Checkers, 69, 16, 200)
