#pragma once

#include "game_interface.h"
#include <array>
#include <algorithm>

namespace games {

class TicTacToe : public GameInterface {
    std::array<int, 9> board_{};
    int current_player_ = 1;
    int moves_ = 0;
    int winner_ = 0;
    bool done_ = false;

    static constexpr int WIN_LINES[8][3] = {
        {0,1,2},{3,4,5},{6,7,8},
        {0,3,6},{1,4,7},{2,5,8},
        {0,4,8},{2,4,6}
    };

public:
    TicTacToe() { reset(); }

    std::string name() const override { return "TicTacToe"; }
    int action_space() const override { return 9; }

    void reset() override {
        board_.fill(0);
        current_player_ = 1;
        moves_ = 0;
        winner_ = 0;
        done_ = false;
    }

    void step(int action) override {
        if (done_ || action < 0 || action >= 9 || board_[action] != 0) return;

        board_[action] = current_player_;
        moves_++;

        for (const auto& line : WIN_LINES) {
            if (board_[line[0]] == current_player_ &&
                board_[line[1]] == current_player_ &&
                board_[line[2]] == current_player_) {
                winner_ = current_player_;
                done_ = true;
                return;
            }
        }

        if (moves_ >= 9) { done_ = true; return; }
        current_player_ = (current_player_ == 1) ? 2 : 1;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        if (winner_ == 1) return 10.0f;
        if (winner_ == 2) return -10.0f;
        return 0.0f;
    }

    int action_count() const override { return moves_; }

    std::string render_state() const override {
        std::string result;
        for (int i = 0; i < 9; ++i) {
            char c = board_[i] == 0 ? '.' : (board_[i] == 1 ? 'X' : 'O');
            result += c;
            if (i % 3 == 2) result += '\n';
        }
        return result;
    }
};

} // namespace games
