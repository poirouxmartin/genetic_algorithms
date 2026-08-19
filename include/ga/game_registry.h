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
    size_t hidden_size;
    int max_steps;
    std::function<std::unique_ptr<games::GameInterface>()> factory;
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

#define REGISTER_GAME(ClassName, InputSz, HiddenSz, MaxSteps) \
    namespace { \
        static GameRegistrar reg_##ClassName( \
            { #ClassName, #ClassName, InputSz, HiddenSz, MaxSteps, \
              []{ return std::make_unique<games::ClassName>(); } } \
        ); \
    }

#define REGISTER_GAME_DISPLAY(ClassName, DisplayName, InputSz, HiddenSz, MaxSteps) \
    namespace { \
        static GameRegistrar reg_##ClassName( \
            { #ClassName, DisplayName, InputSz, HiddenSz, MaxSteps, \
              []{ return std::make_unique<games::ClassName>(); } } \
        ); \
    }
