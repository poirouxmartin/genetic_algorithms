#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cmath>

namespace games {

class GameInterface {
public:
    virtual ~GameInterface() = default;

    virtual std::string name() const = 0;
    virtual void reset() = 0;
    virtual void step(int action) = 0;
    virtual bool is_done() const = 0;
    virtual float fitness() const = 0;
    virtual int action_count() const = 0;
    virtual int action_space() const = 0;

    virtual std::string render_state() const { return ""; }
    virtual std::string render_action(int action) const { return std::to_string(action); }

    virtual void get_state(float*) const {}
    virtual std::vector<int> get_valid_moves() const { return {}; }

    virtual bool uses_scoring() const { return false; }
    virtual std::unique_ptr<GameInterface> clone() const { return nullptr; }
    virtual void apply_move_no_state_change(int action) {}
};

class ScoringNetwork {
    size_t input_size_;
    size_t hidden_size_;

public:
    ScoringNetwork(size_t input, size_t hidden)
        : input_size_(input), hidden_size_(hidden) {}

    float score(const float* genes, const float* state) const {
        std::vector<float> hidden(hidden_size_);

        size_t offset = 0;
        for (size_t h = 0; h < hidden_size_; ++h) {
            float sum = genes[input_size_ * hidden_size_ + h];
            for (size_t i = 0; i < input_size_; ++i) {
                sum += genes[offset + i] * state[i];
            }
            hidden[h] = std::tanh(sum);
            offset += input_size_;
        }

        offset = input_size_ * hidden_size_ + hidden_size_;
        float sum = genes[offset + hidden_size_];
        for (size_t h = 0; h < hidden_size_; ++h) {
            sum += genes[offset + h] * hidden[h];
        }
        return sum;
    }

    static size_t weight_count(size_t input, size_t hidden) {
        return input * hidden + hidden + hidden + 1;
    }
};

class PolicyNetwork {
    size_t input_size_;
    size_t hidden_size_;
    size_t output_size_;

public:
    PolicyNetwork(size_t input, size_t hidden, size_t output)
        : input_size_(input), hidden_size_(hidden), output_size_(output) {}

    int decide(const float* genes, const float* state) const {
        std::vector<float> hidden(hidden_size_);
        std::vector<float> output(output_size_);

        size_t offset = 0;
        for (size_t h = 0; h < hidden_size_; ++h) {
            float sum = genes[input_size_ * hidden_size_ + h];
            for (size_t i = 0; i < input_size_; ++i) {
                sum += genes[offset + i] * state[i];
            }
            hidden[h] = std::tanh(sum);
            offset += input_size_;
        }

        offset = input_size_ * hidden_size_ + hidden_size_;
        int best_action = 0;
        float best_val = -1e30f;
        for (size_t o = 0; o < output_size_; ++o) {
            float sum = genes[offset + hidden_size_ * output_size_ + o];
            for (size_t h = 0; h < hidden_size_; ++h) {
                sum += genes[offset + h] * hidden[h];
            }
            output[o] = sum;
            if (sum > best_val) {
                best_val = sum;
                best_action = static_cast<int>(o);
            }
        }
        return best_action;
    }

    static size_t weight_count(size_t input, size_t hidden, size_t output) {
        return input * hidden + hidden + hidden * output + output;
    }
};

} // namespace games
