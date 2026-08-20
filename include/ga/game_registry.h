#pragma once

#include "games/game_interface.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>

struct GameConfig {
    std::string name;
    std::string display_name;
    size_t input_size;
    std::vector<int> hidden_layers;
    int max_steps;
    std::function<std::unique_ptr<games::GameInterface>()> factory;
    std::vector<std::pair<int,int>> input_symmetry;
};

class GameRegistry {
public:
    static std::vector<GameConfig>& games() {
        static std::vector<GameConfig> instance;
        return instance;
    }
};

struct GameRegistrar {
    GameRegistrar(const GameConfig& gc) {
        GameRegistry::games().push_back(gc);
    }
};

#define REGISTER_GAME(ClassName, InputSz, HiddenLayers, MaxSteps) \
    namespace { \
        static GameRegistrar reg_##ClassName( \
            { #ClassName, #ClassName, InputSz, HiddenLayers, MaxSteps, \
              []{ return std::make_unique<games::ClassName>(); } } \
        ); \
    }

#define REGISTER_GAME_DISPLAY(ClassName, DisplayName, InputSz, HiddenLayers, MaxSteps) \
    namespace { \
        static GameRegistrar reg_##ClassName( \
            { #ClassName, DisplayName, InputSz, HiddenLayers, MaxSteps, \
              []{ return std::make_unique<games::ClassName>(); } } \
        ); \
    }

#define REGISTER_GAME_SYM(ClassName, InputSz, HiddenLayers, MaxSteps, ...) \
    namespace { \
        static GameRegistrar reg_##ClassName( \
            { #ClassName, #ClassName, InputSz, HiddenLayers, MaxSteps, \
              []{ return std::make_unique<games::ClassName>(); }, \
              { __VA_ARGS__ } } \
        ); \
    }
