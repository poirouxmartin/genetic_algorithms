#pragma once

#include "game_interface.h"
#include <array>
#include <vector>
#include <algorithm>
#include <memory>

namespace games {

class Othello : public GameInterface {
    static constexpr int SIZE = 8;
    std::array<int, SIZE * SIZE> board_{};
    int current_player_ = 1;
    int moves_ = 0;
    bool done_ = false;
    int score_black_ = 0, score_white_ = 0;

    static constexpr int DX[] = {0,0,1,1,1,-1,-1,-1};
    static constexpr int DY[] = {1,-1,0,1,-1,0,1,-1};

public:
    Othello() { reset(); }

    std::string name() const override { return "Othello"; }
    int action_space() const override { return SIZE * SIZE; }
    bool uses_scoring() const override { return true; }
    std::unique_ptr<GameInterface> clone() const override {
        auto c = std::make_unique<Othello>();
        c->board_ = board_;
        c->current_player_ = current_player_;
        c->moves_ = moves_;
        c->done_ = done_;
        c->score_black_ = score_black_;
        c->score_white_ = score_white_;
        return c;
    }

    void reset() override {
        board_.fill(0);
        int mid = SIZE / 2;
        board_[mid * SIZE + mid] = 2;
        board_[mid * SIZE + mid - 1] = 1;
        board_[(mid - 1) * SIZE + mid] = 1;
        board_[(mid - 1) * SIZE + mid - 1] = 2;
        current_player_ = 1;
        moves_ = 0;
        done_ = false;
        count_pieces();
    }

    void step(int action) override {
        if (done_ || action < 0 || action >= SIZE * SIZE) return;

        int row = action / SIZE, col = action % SIZE;
        if (board_[action] != 0) return;

        int flipped[32];
        int flipped_count = 0;
        for (int d = 0; d < 8; ++d) {
            int r = row + DY[d], c = col + DX[d];
            int line[8];
            int line_count = 0;
            while (r >= 0 && r < SIZE && c >= 0 && c < SIZE && board_[r * SIZE + c] != 0 &&
                   board_[r * SIZE + c] != current_player_) {
                line[line_count++] = r * SIZE + c;
                r += DY[d];
                c += DX[d];
            }
            if (line_count > 0 && r >= 0 && r < SIZE && c >= 0 && c < SIZE &&
                board_[r * SIZE + c] == current_player_) {
                for (int i = 0; i < line_count; ++i) flipped[flipped_count++] = line[i];
            }
        }

        if (flipped_count == 0) return;

        board_[action] = current_player_;
        for (int i = 0; i < flipped_count; ++i) board_[flipped[i]] = current_player_;
        moves_++;

        current_player_ = (current_player_ == 1) ? 2 : 1;

        if (!has_valid_move(current_player_) && !has_valid_move(current_player_ == 1 ? 2 : 1))
            done_ = true;
        if (moves_ >= SIZE * SIZE - 4) done_ = true;

        count_pieces();
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return static_cast<float>(score_black_) - static_cast<float>(score_white_);
    }

    int action_count() const override { return moves_; }

    void get_state(float* state) const override {
        for (int i = 0; i < SIZE * SIZE; ++i) {
            state[i] = static_cast<float>(board_[i]) / 2.0f;
        }
    }

    std::vector<int> get_valid_moves() const override {
        std::vector<int> moves;
        for (int r = 0; r < SIZE; ++r)
            for (int c = 0; c < SIZE; ++c) {
                if (board_[r * SIZE + c] != 0) continue;
                for (int d = 0; d < 8; ++d) {
                    int nr = r + DY[d], nc = c + DX[d];
                    bool found = false;
                    while (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE &&
                           board_[nr * SIZE + nc] != 0 && board_[nr * SIZE + nc] != current_player_) {
                        found = true;
                        nr += DY[d]; nc += DX[d];
                    }
                    if (found && nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE &&
                        board_[nr * SIZE + nc] == current_player_) {
                        moves.push_back(r * SIZE + c);
                        break;
                    }
                }
            }
        return moves;
    }

    std::string render_state() const override {
        std::string result;
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                char ch = '.';
                if (board_[r * SIZE + c] == 1) ch = 'X';
                else if (board_[r * SIZE + c] == 2) ch = 'O';
                result += ch;
            }
            result += '\n';
        }
        return result;
    }

private:
    bool has_valid_move(int player) const {
        for (int r = 0; r < SIZE; ++r)
            for (int c = 0; c < SIZE; ++c) {
                if (board_[r * SIZE + c] != 0) continue;
                for (int d = 0; d < 8; ++d) {
                    int nr = r + DY[d], nc = c + DX[d];
                    bool found = false;
                    while (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE &&
                           board_[nr * SIZE + nc] != 0 && board_[nr * SIZE + nc] != player) {
                        found = true;
                        nr += DY[d]; nc += DX[d];
                    }
                    if (found && nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE &&
                        board_[nr * SIZE + nc] == player) return true;
                }
            }
        return false;
    }

    void count_pieces() {
        score_black_ = 0;
        score_white_ = 0;
        for (int v : board_) {
            if (v == 1) score_black_++;
            else if (v == 2) score_white_++;
        }
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(Othello, 64, {16}, 60)
