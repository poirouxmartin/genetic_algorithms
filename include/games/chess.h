#pragma once

#include "game_interface.h"
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

namespace games {

class Chess : public GameInterface {
    static constexpr int SZ = 8;
    enum Piece : int { NONE=0, PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6 };

    std::array<int, SZ * SZ> board_{};
    int current_player_ = 1;
    int moves_ = 0;
    int max_moves_ = 200;
    bool done_ = false;
    bool white_won_ = false;
    bool black_won_ = false;
    bool draw_ = false;
    int halfmove_clock_ = 0;
    int ep_col_ = -1;
    bool white_king_moved_ = false;
    bool black_king_moved_ = false;
    bool wr_moved_[2] = {false, false};
    bool br_moved_[2] = {false, false};

    static constexpr float PIECE_VAL[] = {0, 1, 3, 3, 5, 9, 100};

    int& at(int r, int c) { return board_[r * SZ + c]; }
    int at(int r, int c) const { return board_[r * SZ + c]; }

    bool in_bounds(int r, int c) const { return r >= 0 && r < SZ && c >= 0 && c < SZ; }

    bool is_path_clear(int fr, int fc, int tr, int tc) const {
        int dr = (tr > fr) ? 1 : (tr < fr) ? -1 : 0;
        int dc = (tc > fc) ? 1 : (tc < fc) ? -1 : 0;
        int r = fr + dr, c = fc + dc;
        while (r != tr || c != tc) {
            if (board_[r * SZ + c] != 0) return false;
            r += dr; c += dc;
        }
        return true;
    }

    bool is_enemy(int piece, int player) const {
        return piece != 0 && ((player == 1 && piece < 0) || (player == 2 && piece > 0));
    }

    bool is_friendly(int piece, int player) const {
        return piece != 0 && ((player == 1 && piece > 0) || (player == 2 && piece < 0));
    }

    bool is_square_attacked(int r, int c, int by_player) const {
        for (int pr = 0; pr < SZ; ++pr) {
            for (int pc = 0; pc < SZ; ++pc) {
                int p = at(pr, pc);
                if (!is_enemy(p, by_player)) continue;
                int ap = std::abs(p);
                int dr = r - pr, dc = c - pc;
                int adr = std::abs(dr), adc = std::abs(dc);

                if (ap == PAWN) {
                    int dir = (p > 0) ? -1 : 1;
                    if (dr == dir && adc == 1) return true;
                } else if (ap == KNIGHT) {
                    if ((adr == 2 && adc == 1) || (adr == 1 && adc == 2)) return true;
                } else if (ap == BISHOP) {
                    if (adr == adc && adr > 0 && is_path_clear(pr, pc, r, c)) return true;
                } else if (ap == ROOK) {
                    if ((adr == 0 || adc == 0) && (adr + adc > 0) && is_path_clear(pr, pc, r, c)) return true;
                } else if (ap == QUEEN) {
                    if (((adr == 0 || adc == 0) || (adr == adc)) && (adr + adc > 0) && is_path_clear(pr, pc, r, c)) return true;
                } else if (ap == KING) {
                    if (adr <= 1 && adc <= 1 && (adr + adc > 0)) return true;
                }
            }
        }
        return false;
    }

    int find_king(int player) const {
        int target = (player == 1) ? KING : -KING;
        for (int i = 0; i < SZ * SZ; ++i)
            if (board_[i] == target) return i;
        return -1;
    }

    bool is_in_check(int player) const {
        int kp = find_king(player);
        if (kp < 0) return false;
        return is_square_attacked(kp / SZ, kp % SZ, player == 1 ? 2 : 1);
    }

    struct RawMove { int from, to; bool en_passant; bool castle_king; bool castle_queen; int promo; };

    std::vector<RawMove> generate_pseudo_legal(int player) const {
        std::vector<RawMove> moves;
        for (int fr = 0; fr < SZ; ++fr) {
            for (int fc = 0; fc < SZ; ++fc) {
                int p = at(fr, fc);
                if (!is_friendly(p, player)) continue;
                int ap = std::abs(p);
                int sign = (player == 1) ? 1 : -1;

                auto add_move = [&](int tr, int tc, bool ep = false, bool ck = false, bool cq = false, int promo = 0) {
                    if (!in_bounds(tr, tc)) return;
                    int target = at(tr, tc);
                    if (is_friendly(target, player)) return;
                    moves.push_back({fr * SZ + fc, tr * SZ + tc, ep, ck, cq, promo});
                };

                if (ap == PAWN) {
                    int dir = (player == 1) ? -1 : 1;
                    int start_row = (player == 1) ? 6 : 1;
                    int promo_row = (player == 1) ? 0 : 7;

                    int nr = fr + dir;
                    if (in_bounds(nr, fc) && at(nr, fc) == 0) {
                        if (nr == promo_row) {
                            for (int promo : {QUEEN, ROOK, BISHOP, KNIGHT})
                                moves.push_back({fr * SZ + fc, nr * SZ + fc, false, false, false, promo});
                        } else {
                            add_move(nr, fc);
                        }
                        if (fr == start_row) {
                            int nr2 = fr + 2 * dir;
                            if (at(nr2, fc) == 0)
                                add_move(nr2, fc);
                        }
                    }
                    for (int dc : {-1, 1}) {
                        int nc = fc + dc;
                        if (!in_bounds(nr, nc)) continue;
                        if (is_enemy(at(nr, nc), player)) {
                            if (nr == promo_row) {
                                for (int promo : {QUEEN, ROOK, BISHOP, KNIGHT})
                                    moves.push_back({fr * SZ + fc, nr * SZ + nc, false, false, false, promo});
                            } else {
                                add_move(nr, nc);
                            }
                        }
                        if (in_bounds(nr, nc) && ep_col_ == nc && fr == (player == 1 ? 3 : 4)) {
                            moves.push_back({fr * SZ + fc, nr * SZ + nc, true, false, false, 0});
                        }
                    }
                } else if (ap == KNIGHT) {
                    int offsets[][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
                    for (auto& o : offsets) add_move(fr + o[0], fc + o[1]);
                } else if (ap == BISHOP) {
                    int dirs[][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
                    for (auto& d : dirs)
                        for (int s = 1; s < SZ; ++s) {
                            int tr = fr + d[0]*s, tc = fc + d[1]*s;
                            if (!in_bounds(tr, tc)) break;
                            if (at(tr, tc) != 0) { if (is_enemy(at(tr, tc), player)) add_move(tr, tc); break; }
                            add_move(tr, tc);
                        }
                } else if (ap == ROOK) {
                    int dirs[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
                    for (auto& d : dirs)
                        for (int s = 1; s < SZ; ++s) {
                            int tr = fr + d[0]*s, tc = fc + d[1]*s;
                            if (!in_bounds(tr, tc)) break;
                            if (at(tr, tc) != 0) { if (is_enemy(at(tr, tc), player)) add_move(tr, tc); break; }
                            add_move(tr, tc);
                        }
                } else if (ap == QUEEN) {
                    int dirs[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
                    for (auto& d : dirs)
                        for (int s = 1; s < SZ; ++s) {
                            int tr = fr + d[0]*s, tc = fc + d[1]*s;
                            if (!in_bounds(tr, tc)) break;
                            if (at(tr, tc) != 0) { if (is_enemy(at(tr, tc), player)) add_move(tr, tc); break; }
                            add_move(tr, tc);
                        }
                } else if (ap == KING) {
                    int dirs[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
                    for (auto& d : dirs) add_move(fr + d[0], fc + d[1]);

                    int row = (player == 1) ? 7 : 0;
                    int kr = (player == 1) ? !white_king_moved_ : !black_king_moved_;
                    bool wrk = (player == 1) ? wr_moved_[1] : br_moved_[1];
                    bool wrq = (player == 1) ? wr_moved_[0] : br_moved_[0];
                    int opp = player == 1 ? 2 : 1;

                    if (fr == row && fc == 4 && !kr && !is_in_check(player)) {
                        if (!wrk && at(row, 5) == 0 && at(row, 6) == 0 && at(row, 7) * sign == ROOK &&
                            !is_square_attacked(row, 5, opp) && !is_square_attacked(row, 6, opp)) {
                            moves.push_back({fr * SZ + fc, row * SZ + 6, false, true, false, 0});
                        }
                        if (!wrq && at(row, 3) == 0 && at(row, 2) == 0 && at(row, 1) == 0 && at(row, 0) * sign == ROOK &&
                            !is_square_attacked(row, 3, opp) && !is_square_attacked(row, 2, opp)) {
                            moves.push_back({fr * SZ + fc, row * SZ + 2, false, false, true, 0});
                        }
                    }
                }
            }
        }
        return moves;
    }

    std::vector<int> generate_legal(int player) {
        std::vector<int> legal;
        auto pseudo = generate_pseudo_legal(player);
        for (auto& m : pseudo) {
            std::array<int, SZ * SZ> saved = board_;
            int saved_ep = ep_col_;
            int from_r = m.from / SZ, from_c = m.from % SZ;
            int to_r = m.to / SZ, to_c = m.to % SZ;

            if (m.castle_king || m.castle_queen) {
                board_[m.to] = board_[m.from];
                board_[m.from] = 0;
                int row = (player == 1) ? 7 : 0;
                if (m.castle_king) {
                    board_[row * SZ + 5] = board_[row * SZ + 7];
                    board_[row * SZ + 7] = 0;
                } else {
                    board_[row * SZ + 3] = board_[row * SZ + 0];
                    board_[row * SZ + 0] = 0;
                }
            } else if (m.en_passant) {
                board_[m.to] = board_[m.from];
                board_[m.from] = 0;
                board_[from_r * SZ + to_c] = 0;
            } else {
                if (m.promo) {
                    board_[m.to] = m.promo * (player == 1 ? 1 : -1);
                } else {
                    board_[m.to] = board_[m.from];
                }
                board_[m.from] = 0;
            }

            bool in_check = is_in_check(player);
            board_ = saved;
            ep_col_ = saved_ep;

            if (!in_check)
                legal.push_back(m.from * SZ * SZ + m.to);
        }
        return legal;
    }

    int pick_opponent_move() {
        auto legal = generate_legal(2);
        if (legal.empty()) return -1;

        int best = legal[0];
        int best_score = -99999;
        for (int m : legal) {
            int to = m % (SZ * SZ);
            int cap = std::abs(board_[to]);
            int score = PIECE_VAL[cap] * 10;
            int tr = to / SZ, tc = to % SZ;
            score += (3 - std::abs(tr - 3.5f) - std::abs(tc - 3.5f));
            if (score > best_score) { best_score = score; best = m; }
        }

        if (std::rand() % 5 == 0 && legal.size() > 1)
            return legal[std::rand() % legal.size()];
        return best;
    }

public:
    Chess() { reset(); }

    std::string name() const override { return "Chess"; }
    int action_space() const override { return SZ * SZ * SZ * SZ; }
    bool uses_scoring() const override { return true; }

    std::unique_ptr<GameInterface> clone() const override {
        auto c = std::make_unique<Chess>();
        c->board_ = board_;
        c->current_player_ = current_player_;
        c->moves_ = moves_;
        c->done_ = done_;
        c->white_won_ = white_won_;
        c->black_won_ = black_won_;
        c->draw_ = draw_;
        c->halfmove_clock_ = halfmove_clock_;
        c->ep_col_ = ep_col_;
        c->white_king_moved_ = white_king_moved_;
        c->black_king_moved_ = black_king_moved_;
        c->wr_moved_[0] = wr_moved_[0]; c->wr_moved_[1] = wr_moved_[1];
        c->br_moved_[0] = br_moved_[0]; c->br_moved_[1] = br_moved_[1];
        return c;
    }

    void reset() override {
        board_.fill(0);
        int back[] = {ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK};
        for (int c = 0; c < SZ; ++c) {
            board_[c] = back[c];
            board_[SZ + c] = PAWN;
            board_[(SZ - 2) * SZ + c] = -PAWN;
            board_[(SZ - 1) * SZ + c] = -back[c];
        }
        current_player_ = 1;
        moves_ = 0;
        done_ = false;
        white_won_ = false;
        black_won_ = false;
        draw_ = false;
        halfmove_clock_ = 0;
        ep_col_ = -1;
        white_king_moved_ = false;
        black_king_moved_ = false;
        wr_moved_[0] = wr_moved_[1] = false;
        br_moved_[0] = br_moved_[1] = false;
    }

    void step(int action) override {
        if (done_) return;

        if (current_player_ == 2) {
            int move = pick_opponent_move();
            if (move == -1) { white_won_ = true; done_ = true; return; }
            int from = move / (SZ * SZ);
            int to = move % (SZ * SZ);
            int piece = board_[from];
            int target = board_[to];
            int fr = from / SZ, fc = from % SZ;
            int tr = to / SZ, tc = to % SZ;

            if (std::abs(piece) == KING) black_king_moved_ = true;
            if (std::abs(piece) == ROOK) {
                if (fr == 0 && fc == 0) br_moved_[0] = true;
                if (fr == 0 && fc == 7) br_moved_[1] = true;
            }

            ep_col_ = -1;
            bool is_capture = (target != 0);
            if (std::abs(piece) == PAWN && std::abs(tr - fr) == 2) ep_col_ = fc;

            board_[to] = piece;
            board_[from] = 0;
            if (std::abs(piece) == PAWN && tr == SZ - 1) board_[to] = -QUEEN;

            halfmove_clock_ = is_capture ? 0 : halfmove_clock_ + 1;
            moves_++;
            current_player_ = 1;

            if (find_king(1) < 0) { black_won_ = true; done_ = true; return; }
            auto legal = generate_legal(1);
            if (legal.empty()) {
                if (is_in_check(1)) { black_won_ = true; done_ = true; }
                else { draw_ = true; done_ = true; }
                return;
            }
            if (halfmove_clock_ >= 100) { draw_ = true; done_ = true; }
            if (moves_ >= max_moves_) { draw_ = true; done_ = true; }
            return;
        }

        int from = action / (SZ * SZ);
        int to = action % (SZ * SZ);
        if (from < 0 || from >= SZ * SZ || to < 0 || to >= SZ * SZ) return;
        int piece = board_[from];
        if (!is_friendly(piece, 1)) return;

        auto legal = generate_legal(1);
        bool found = false;
        int found_from = 0, found_to = 0, found_promo = 0;
        for (int m : legal) {
            int mf = m / (SZ * SZ);
            int mt = m % (SZ * SZ);
            if (mf == from && mt == to) {
                found = true;
                found_from = mf;
                found_to = mt;
                break;
            }
        }
        if (!found) return;

        int fr = from / SZ, fc = from % SZ;
        int tr = to / SZ, tc = to % SZ;
        int target = board_[to];

        if (std::abs(piece) == KING) white_king_moved_ = true;
        if (std::abs(piece) == ROOK) {
            if (fr == 7 && fc == 0) wr_moved_[0] = true;
            if (fr == 7 && fc == 7) wr_moved_[1] = true;
        }

        bool is_capture = (target != 0);
        ep_col_ = -1;
        if (std::abs(piece) == PAWN && std::abs(tr - fr) == 2) ep_col_ = tc;

        auto promo_it = std::find_if(legal.begin(), legal.end(), [&](int m) {
            return m / (SZ * SZ) == from && m % (SZ * SZ) == to;
        });

        board_[to] = piece;
        board_[from] = 0;

        if (std::abs(piece) == PAWN && tr == 0) board_[to] = QUEEN;

        halfmove_clock_ = is_capture ? 0 : halfmove_clock_ + 1;
        moves_++;
        current_player_ = 2;

        if (find_king(2) < 0) { white_won_ = true; done_ = true; return; }
        auto opp_legal = generate_legal(2);
        if (opp_legal.empty()) {
            if (is_in_check(2)) { white_won_ = true; done_ = true; }
            else { draw_ = true; done_ = true; }
            return;
        }
        if (halfmove_clock_ >= 100) { draw_ = true; done_ = true; }
        if (moves_ >= max_moves_) { draw_ = true; done_ = true; }
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        if (white_won_) return 100.0f;
        if (black_won_) return -100.0f;
        if (draw_) return 0.0f;
        float score = 0.0f;
        for (int i = 0; i < SZ * SZ; ++i) {
            int v = board_[i];
            if (v == 0) continue;
            int r = i / SZ;
            float center_bonus = (3.5f - std::abs(r - 3.5f)) * 0.1f;
            score += (PIECE_VAL[std::abs(v)] + center_bonus) * (v > 0 ? 1.0f : -1.0f);
        }
        return score * 2.0f;
    }

    int action_count() const override { return moves_; }

    void get_state(float* state) const override {
        for (int i = 0; i < SZ * SZ; ++i)
            state[i] = static_cast<float>(board_[i]) / 6.0f;

        state[64] = white_king_moved_ ? 0.0f : 1.0f;
        state[65] = white_moved_rook_queen() ? 0.0f : 1.0f;
        state[66] = black_king_moved_ ? 0.0f : 1.0f;
        state[67] = black_moved_rook_queen() ? 0.0f : 1.0f;
        state[68] = ep_col_ >= 0 ? static_cast<float>(ep_col_) / 7.0f : -1.0f;
        state[69] = static_cast<float>(halfmove_clock_) / 100.0f;
    }

    std::vector<int> get_valid_moves() const override {
        if (current_player_ != 1) return {};
        return const_cast<Chess*>(this)->generate_legal(1);
    }

    std::string render_state() const override {
        std::string result;
        static const char* symbols[] = {".", "P", "N", "B", "R", "Q", "K"};
        result += "  abcdefgh\n";
        for (int r = 0; r < SZ; ++r) {
            result += static_cast<char>('8' - r);
            result += ' ';
            for (int c = 0; c < SZ; ++c) {
                int v = board_[r * SZ + c];
                if (v == 0) result += ((r + c) % 2 == 0) ? '.' : ' ';
                else if (v > 0) result += static_cast<char>(std::toupper(symbols[v][0]));
                else result += static_cast<char>(std::tolower(symbols[-v][0]));
            }
            result += '\n';
        }
        result += "  abcdefgh\n";
        if (white_won_) result += "White wins!";
        else if (black_won_) result += "Black wins!";
        else if (draw_) result += "Draw!";
        else result += (current_player_ == 1 ? "White to move" : "Black to move");
        if (is_in_check(current_player_)) result += " (check!)";
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
                SDL_SetRenderDrawColor(renderer, light ? 200 : 90, light ? 195 : 130, light ? 180 : 80, 255);
                SDL_Rect sq = {sx, sy, cell, cell};
                SDL_RenderFillRect(renderer, &sq);

                int v = board_[r * SZ + c];
                if (v != 0) {
                    bool is_white = v > 0;
                    int piece = std::abs(v);
                    int cx2 = sx + cell / 2;
                    int cy2 = sy + cell / 2;
                    int sz = cell / 3;

                    SDL_SetRenderDrawColor(renderer,
                        is_white ? 240 : 40, is_white ? 240 : 40, is_white ? 230 : 40, 255);

                    switch (piece) {
                        case PAWN: {
                            int r2 = sz / 2;
                            for (int dy = -r2; dy <= r2; ++dy) {
                                int dx = static_cast<int>(std::sqrt(std::max(0, r2 * r2 - dy * dy)));
                                SDL_RenderDrawLine(renderer, cx2 - dx, cy2 + dy, cx2 + dx, cy2 + dy);
                            }
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
                            break;
                        }
                        case ROOK: {
                            SDL_Rect tower = {cx2 - sz / 3, cy2 - sz / 2, sz * 2 / 3, sz};
                            SDL_RenderFillRect(renderer, &tower);
                            SDL_Rect top = {cx2 - sz / 2, cy2 - sz / 2 - sz / 5, sz, sz / 5};
                            SDL_RenderFillRect(renderer, &top);
                            break;
                        }
                        case QUEEN: {
                            for (int i = 0; i < sz; ++i) {
                                int hw = (sz - i) * 3 / 4;
                                SDL_RenderDrawLine(renderer, cx2 - hw, cy2 - sz + i, cx2 + hw, cy2 - sz + i);
                            }
                            for (int a = 0; a < 5; ++a) {
                                float ang = static_cast<float>(a) * 3.14159f / 4.0f - 3.14159f / 2.0f;
                                int px = cx2 + static_cast<int>(std::cos(ang) * sz / 3);
                                int py = cy2 - sz + static_cast<int>(std::sin(ang) * sz / 3);
                                SDL_Rect dot = {px - 2, py - 2, 4, 4};
                                SDL_RenderFillRect(renderer, &dot);
                            }
                            break;
                        }
                        case KING: {
                            SDL_Rect body = {cx2 - sz / 3, cy2 - sz / 2, sz * 2 / 3, sz};
                            SDL_RenderFillRect(renderer, &body);
                            SDL_RenderDrawLine(renderer, cx2, cy2 - sz, cx2, cy2 - sz / 2);
                            SDL_RenderDrawLine(renderer, cx2 - sz / 4, cy2 - sz * 3 / 4, cx2 + sz / 4, cy2 - sz * 3 / 4);
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

private:
    bool white_moved_rook_queen() const { return wr_moved_[0]; }
    bool black_moved_rook_queen() const { return br_moved_[0]; }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME_DISPLAY(Chess, "Chess", 70, 20, 200)
