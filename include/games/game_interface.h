#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <numeric>

struct SDL_Renderer;

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

    virtual void render_sdl(SDL_Renderer* r, int x, int y, int w, int h) const {}
    virtual bool has_sdl_render() const { return false; }
};

// hidden_layers: sizes of hidden layers, e.g. {64, 32} for two hidden layers.
// Genome layout: for each consecutive pair in [input, h1, h2, ..., hn, output]:
//   weights (fan_in * fan_out) + biases (fan_out)

inline std::vector<int> make_full_layers(int input, const std::vector<int>& hidden_layers, int output) {
    std::vector<int> layers;
    layers.reserve(hidden_layers.size() + 2);
    layers.push_back(input);
    for (int h : hidden_layers) layers.push_back(h);
    layers.push_back(output);
    return layers;
}

inline size_t weight_count_from_layers(const std::vector<int>& layers) {
    size_t total = 0;
    for (size_t i = 0; i + 1 < layers.size(); ++i)
        total += static_cast<size_t>(layers[i]) * static_cast<size_t>(layers[i + 1]) + static_cast<size_t>(layers[i + 1]);
    return total;
}

class ScoringNetwork {
    std::vector<int> layers_;

public:
    ScoringNetwork(int input, const std::vector<int>& hidden_layers)
        : layers_(make_full_layers(input, hidden_layers, 1)) {}

    float score(const float* genes, const float* state) const {
        std::vector<float> cur(state, state + layers_[0]);
        std::vector<float> nxt;
        size_t offset = 0;
        for (size_t l = 0; l + 1 < layers_.size(); ++l) {
            int fan_in = layers_[l];
            int fan_out = layers_[l + 1];
            bool is_output = (l + 2 == layers_.size());
            nxt.resize(fan_out);
            for (int o = 0; o < fan_out; ++o) {
                float sum = genes[offset + fan_in * fan_out + o];
                for (int i = 0; i < fan_in; ++i)
                    sum += genes[offset + o * fan_in + i] * cur[i];
                nxt[o] = is_output ? sum : std::tanh(sum);
            }
            cur.swap(nxt);
            offset += static_cast<size_t>(fan_in) * fan_out + fan_out;
        }
        return cur[0];
    }

    const std::vector<int>& layers() const { return layers_; }

    static size_t weight_count(int input, const std::vector<int>& hidden_layers) {
        return weight_count_from_layers(make_full_layers(input, hidden_layers, 1));
    }
};

class PolicyNetwork {
    std::vector<int> layers_;

public:
    PolicyNetwork(int input, const std::vector<int>& hidden_layers, int output)
        : layers_(make_full_layers(input, hidden_layers, output)) {}

    int decide(const float* genes, const float* state) const {
        std::vector<float> cur(state, state + layers_[0]);
        std::vector<float> nxt;
        size_t offset = 0;
        for (size_t l = 0; l + 1 < layers_.size(); ++l) {
            int fan_in = layers_[l];
            int fan_out = layers_[l + 1];
            bool is_output = (l + 2 == layers_.size());
            nxt.resize(fan_out);
            for (int o = 0; o < fan_out; ++o) {
                float sum = genes[offset + fan_in * fan_out + o];
                for (int i = 0; i < fan_in; ++i)
                    sum += genes[offset + o * fan_in + i] * cur[i];
                nxt[o] = is_output ? sum : std::tanh(sum);
            }
            cur.swap(nxt);
            offset += static_cast<size_t>(fan_in) * fan_out + fan_out;
        }
        int best_action = 0;
        float best_val = -1e30f;
        for (int o = 0; o < static_cast<int>(cur.size()); ++o) {
            if (cur[o] > best_val) { best_val = cur[o]; best_action = o; }
        }
        return best_action;
    }

    const std::vector<int>& layers() const { return layers_; }

    static size_t weight_count(int input, const std::vector<int>& hidden_layers, int output) {
        return weight_count_from_layers(make_full_layers(input, hidden_layers, output));
    }
};

} // namespace games
