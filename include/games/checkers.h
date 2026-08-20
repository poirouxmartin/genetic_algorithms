#pragma once

#include "game_interface.h"
#include <array>
#include <vector>
#include <algorithm>
#include <memory>
#include <string>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

namespace games {

class Checkers : public GameInterface {
    static constexpr int SZ = 8;

    enum Piece : int { EMPTY = 0, W_MAN = 1, W_KING = 2, B_MAN = -1, B_KING = -2 };

    std::array<int, SZ * SZ> board_{};
    int current_player_ = 1;
    int moves_ = 0;
    bool done_ = false;
    bool white_won_ = false;
    bool black_won_ = false;
    bool draw_ = false;

    int& at(int r, int c) { return board_[r * SZ + c]; }
    int at(int r, int c) const { return board_[r * SZ + c]; }

    bool in_bounds(int r, int c) const { return r >= 0 && r < SZ && c >= 0 && c < SZ; }

    bool is_player_piece(int p, int player) const {
        if (player == 1) return p == W_MAN || p == W_KING;
        return p == B_MAN || p == B_KING;
    }

    bool is_opponent_piece(int p, int player) const {
        if (player == 1) return p == B_MAN || p == B_KING;
        return p == W_MAN || p == W_KING;
    }

    bool is_king(int p) const { return p == W_KING || p == B_KING; }

    struct Move {
        int from_r, from_c, to_r, to_c;
        int captured_r, captured_c;
        bool is_capture;
    };

    std::vector<Move> find_jumps_from(int r, int c, int player) const {
        std::vector<Move> jumps;
        int piece = at(r, c);
        int fwd = (player == 1) ? -1 : 1;
        int dr[4], dc[4];
        int ndirs = 0;

        dr[ndirs] = fwd; dc[ndirs] = -1; ndirs++;
        dr[ndirs] = fwd; dc[ndirs] = 1; ndirs++;

        if (is_king(piece)) {
            dr[ndirs] = -fwd; dc[ndirs] = -1; ndirs++;
            dr[ndirs] = -fwd; dc[ndirs] = 1; ndirs++;
        }

        for (int d = 0; d < ndirs; ++d) {
            int mr = r + dr[d], mc = c + dc[d];
            int tr = r + 2 * dr[d], tc = c + 2 * dc[d];
            if (!in_bounds(tr, tc)) continue;
            if (is_opponent_piece(at(mr, mc), player) && at(tr, tc) == EMPTY) {
                jumps.push_back({r, c, tr, tc, mr, mc, true});
            }
        }
        return jumps;
    }

    std::vector<Move> find_slides_from(int r, int c, int player) const {
        std::vector<Move> slides;
        int piece = at(r, c);
        int fwd = (player == 1) ? -1 : 1;
        int dr[4], dc[4];
        int ndirs = 0;

        dr[ndirs] = fwd; dc[ndirs] = -1; ndirs++;
        dr[ndirs] = fwd; dc[ndirs] = 1; ndirs++;

        if (is_king(piece)) {
            dr[ndirs] = -fwd; dc[ndirs] = -1; ndirs++;
            dr[ndirs] = -fwd; dc[ndirs] = 1; ndirs++;
        }

        for (int d = 0; d < ndirs; ++d) {
            int tr = r + dr[d], tc = c + dc[d];
            if (!in_bounds(tr, tc)) continue;
            if (at(tr, tc) == EMPTY) {
                slides.push_back({r, c, tr, tc, -1, -1, false});
            }
        }
        return slides;
    }

    std::vector<Move> find_all_jumps(int player) const {
        std::vector<Move> all;
        for (int r = 0; r < SZ; ++r)
            for (int c = 0; c < SZ; ++c)
                if (is_player_piece(at(r, c), player)) {
                    auto jumps = find_jumps_from(r, c, player);
                    all.insert(all.end(), jumps.begin(), jumps.end());
                }
        return all;
    }

    std::vector<Move> find_all_slides(int player) const {
        std::vector<Move> all;
        for (int r = 0; r < SZ; ++r)
            for (int c = 0; c < SZ; ++c)
                if (is_player_piece(at(r, c), player)) {
                    auto slides = find_slides_from(r, c, player);
                    all.insert(all.end(), slides.begin(), slides.end());
                }
        return all;
    }

    bool has_any_move(int player) const {
        for (int r = 0; r < SZ; ++r)
            for (int c = 0; c < SZ; ++c)
                if (is_player_piece(at(r, c), player)) {
                    if (!find_jumps_from(r, c, player).empty()) return true;
                    if (!find_slides_from(r, c, player).empty()) return true;
                }
        return false;
    }

    void apply_move(const Move& m) {
        int piece = at(m.from_r, m.from_c);
        at(m.from_r, m.from_c) = EMPTY;
        at(m.to_r, m.to_c) = piece;

        if (m.is_capture) {
            at(m.captured_r, m.captured_c) = EMPTY;
        }

        int promo_row = (piece == W_MAN) ? 0 : (piece == B_MAN) ? SZ - 1 : -1;
        if (m.to_r == promo_row) {
            at(m.to_r, m.to_c) = (piece == W_MAN) ? W_KING : B_KING;
        }
    }

    bool can_continue_jump(int r, int c, int player) const {
        auto jumps = find_jumps_from(r, c, player);
        return !jumps.empty();
    }

    void count_pieces(int& w, int& b) const {
        w = 0; b = 0;
        for (int p : board_) {
            if (p == W_MAN || p == W_KING) w++;
            else if (p == B_MAN || p == B_KING) b++;
        }
    }

    int pick_opponent_move() {
        auto jumps = find_all_jumps(2);
        if (!jumps.empty()) {
            int best_idx = 0;
            int best_score = -999;
            for (int i = 0; i < static_cast<int>(jumps.size()); ++i) {
                int score = 10;
                int piece = at(jumps[i].from_r, jumps[i].from_c);
                if (is_king(piece)) score += 5;
                if (jumps[i].to_r == 0 || jumps[i].to_r == SZ - 1) score += 3;
                int captured = at(jumps[i].captured_r, jumps[i].captured_c);
                if (captured == W_KING) score += 8;
                else if (captured == W_MAN) score += 4;
                if (score > best_score) { best_score = score; best_idx = i; }
            }
            if (std::rand() % 4 == 0 && jumps.size() > 1)
                return std::rand() % static_cast<int>(jumps.size());
            return best_idx;
        }

        auto slides = find_all_slides(2);
        if (slides.empty()) return -1;

        if (std::rand() % 3 == 0)
            return std::rand() % static_cast<int>(slides.size());

        int best_idx = 0;
        int best_score = -999;
        for (int i = 0; i < static_cast<int>(slides.size()); ++i) {
            int score = 0;
            if (slides[i].to_r == 0 || slides[i].to_r == SZ - 1) score += 5;
            score += (SZ - 1 - slides[i].to_r);
            int center = std::abs(slides[i].to_c - 3) + std::abs(slides[i].to_r - 3);
            score += (6 - center);
            if (score > best_score) { best_score = score; best_idx = i; }
        }
        return best_idx;
    }

public:
    Checkers() { reset(); }

    std::string name() const override { return "Checkers"; }
    int action_space() const override { return SZ * SZ * SZ * SZ; }
    bool uses_scoring() const override { return true; }

    std::unique_ptr<GameInterface> clone() const override {
        auto c = std::make_unique<Checkers>();
        c->board_ = board_;
        c->current_player_ = current_player_;
        c->moves_ = moves_;
        c->done_ = done_;
        c->white_won_ = white_won_;
        c->black_won_ = black_won_;
        c->draw_ = draw_;
        return c;
    }

    void reset() override {
        board_.fill(EMPTY);
        for (int r = 0; r < SZ; ++r) {
            for (int c = 0; c < SZ; ++c) {
                if ((r + c) % 2 != 1) continue;
                if (r < 3) board_[r * SZ + c] = B_MAN;
                else if (r > 4) board_[r * SZ + c] = W_MAN;
            }
        }
        current_player_ = 1;
        moves_ = 0;
        done_ = false;
        white_won_ = false;
        black_won_ = false;
        draw_ = false;
    }

    void step(int action) override {
        if (done_) return;

        if (current_player_ == 2) {
            int idx = pick_opponent_move();
            if (idx == -1) { white_won_ = true; done_ = true; return; }

            auto jumps = find_all_jumps(2);
            if (!jumps.empty()) {
                auto m = jumps[idx % static_cast<int>(jumps.size())];
                apply_move(m);
                int promo_row = SZ - 1;
                if (m.to_r == promo_row && can_continue_jump(m.to_r, m.to_c, 2)) {
                    return;
                }
            } else {
                auto slides = find_all_slides(2);
                if (idx < static_cast<int>(slides.size())) {
                    apply_move(slides[idx]);
                }
            }

            moves_++;
            current_player_ = 1;
        } else {
            auto jumps = find_all_jumps(1);
            bool must_jump = !jumps.empty();

            int from = action / (SZ * SZ);
            int to = action % (SZ * SZ);
            int fr = from / SZ, fc = from % SZ;
            int tr = to / SZ, tc = to % SZ;
            if (from == to) return;

            bool moved = false;
            if (must_jump) {
                for (auto& m : jumps) {
                    if (m.from_r == fr && m.from_c == fc && m.to_r == tr && m.to_c == tc) {
                        apply_move(m);
                        moved = true;
                        int promo_row = 0;
                        if (m.to_r == promo_row && can_continue_jump(m.to_r, m.to_c, 1)) {
                            return;
                        }
                        break;
                    }
                }
            } else {
                auto slides = find_all_slides(1);
                for (auto& m : slides) {
                    if (m.from_r == fr && m.from_c == fc && m.to_r == tr && m.to_c == tc) {
                        apply_move(m);
                        moved = true;
                        break;
                    }
                }
            }
            if (!moved) return;

            moves_++;
            current_player_ = 2;
        }

        int wc = 0, bc = 0;
        count_pieces(wc, bc);
        if (wc == 0) { black_won_ = true; done_ = true; return; }
        if (bc == 0) { white_won_ = true; done_ = true; return; }
        if (!has_any_move(current_player_)) {
            if (current_player_ == 1) { black_won_ = true; done_ = true; }
            else { white_won_ = true; done_ = true; }
        }
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        if (white_won_) return 100.0f;
        if (black_won_) return -50.0f;
        if (draw_) return 0.0f;

        float score = 0.0f;
        for (int r = 0; r < SZ; ++r) {
            for (int c = 0; c < SZ; ++c) {
                int p = board_[r * SZ + c];
                if (p == EMPTY) continue;
                float val = is_king(p) ? 3.0f : 1.0f;
                float advance = 0.0f;
                if (p == W_MAN) advance = static_cast<float>(SZ - 1 - r) * 0.15f;
                else if (p == B_MAN) advance = static_cast<float>(r) * 0.15f;
                float center = (3.5f - std::abs(c - 3.5f)) * 0.1f;
                score += (val + advance + center) * (is_player_piece(p, 1) ? 1.0f : -1.0f);
            }
        }
        return score;
    }

    int action_count() const override { return moves_; }

    void get_state(float* state) const override {
        for (int r = 0; r < SZ; ++r) {
            for (int c = 0; c < SZ; ++c) {
                int p = board_[r * SZ + c];
                float val = 0.0f;
                if (p == W_MAN) val = 0.33f;
                else if (p == W_KING) val = 1.0f;
                else if (p == B_MAN) val = -0.33f;
                else if (p == B_KING) val = -1.0f;
                state[r * SZ + c] = val;
            }
        }
        int wc = 0, bc = 0;
        count_pieces(wc, bc);
        state[64] = static_cast<float>(wc) / 12.0f;
        state[65] = static_cast<float>(bc) / 12.0f;
        state[66] = static_cast<float>(current_player_ == 1 ? 1.0f : 0.0f);
    }

    std::vector<int> get_valid_moves() const override {
        if (current_player_ != 1) return {};
        auto jumps = find_all_jumps(1);
        std::vector<int> valid;
        if (!jumps.empty()) {
            for (auto& m : jumps)
                valid.push_back((m.from_r * SZ + m.from_c) * SZ * SZ + (m.to_r * SZ + m.to_c));
        } else {
            auto slides = find_all_slides(1);
            for (auto& m : slides)
                valid.push_back((m.from_r * SZ + m.from_c) * SZ * SZ + (m.to_r * SZ + m.to_c));
        }
        return valid;
    }

    std::string render_state() const override {
        std::string result;
        result += "  a b c d e f g h\n";
        for (int r = 0; r < SZ; ++r) {
            result += static_cast<char>('8' - r);
            result += ' ';
            for (int c = 0; c < SZ; ++c) {
                int p = board_[r * SZ + c];
                char ch = ((r + c) % 2 == 0) ? ' ' : '.';
                if (p == W_MAN) ch = 'o';
                else if (p == W_KING) ch = 'O';
                else if (p == B_MAN) ch = 'x';
                else if (p == B_KING) ch = 'X';
                result += ch;
            }
            result += '\n';
        }
        result += "  a b c d e f g h\n";
        int wc = 0, bc = 0;
        count_pieces(wc, bc);
        result += "You:" + std::to_string(wc) + " Opp:" + std::to_string(bc);
        if (white_won_) result += " You win!";
        else if (black_won_) result += " You lose!";
        else if (draw_) result += " Draw!";
        else result += (current_player_ == 1 ? " Your turn" : " Opp's turn");
        return result;
    }

    bool has_sdl_render() const override { return true; }

    void render_sdl(SDL_Renderer* renderer, int x, int y, int w, int h) const override {
        int cell = std::min(w, h) / SZ;
        int bw = cell * SZ, bh = cell * SZ;
        int ox = x + (w - bw) / 2;
        int oy = y + (h - bh) / 2;

        SDL_SetRenderDrawColor(renderer, 42, 45, 62, 255);
        SDL_Rect bg = {x, y, w, h};
        SDL_RenderFillRect(renderer, &bg);

        for (int r = 0; r < SZ; ++r) {
            for (int c = 0; c < SZ; ++c) {
                int sx = ox + c * cell;
                int sy = oy + r * cell;
                bool light = (r + c) % 2 == 0;
                SDL_SetRenderDrawColor(renderer, light ? 200 : 90, light ? 190 : 130, light ? 160 : 80, 255);
                SDL_Rect sq = {sx, sy, cell, cell};
                SDL_RenderFillRect(renderer, &sq);

                int p = board_[r * SZ + c];
                if (p != EMPTY) {
                    bool is_white = (p == W_MAN || p == W_KING);
                    bool king = is_king(p);
                    int cx2 = sx + cell / 2;
                    int cy2 = sy + cell / 2;
                    int radius = cell * 3 / 8;

                    SDL_SetRenderDrawColor(renderer, is_white ? 240 : 40, is_white ? 240 : 40, is_white ? 230 : 40, 255);
                    for (int dy = -radius; dy <= radius; ++dy) {
                        int dx = static_cast<int>(std::sqrt(std::max(0, radius * radius - dy * dy)));
                        SDL_RenderDrawLine(renderer, cx2 - dx, cy2 + dy, cx2 + dx, cy2 + dy);
                    }

                    if (king) {
                        SDL_SetRenderDrawColor(renderer, is_white ? 200 : 80, is_white ? 180 : 60, is_white ? 50 : 20, 255);
                        SDL_Rect crown = {cx2 - radius / 2, cy2 - radius / 3, radius, radius / 3};
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
REGISTER_GAME_DISPLAY(Checkers, "Checkers", 67, {16}, 2000)
