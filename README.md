# Genetic Algorithms

C++17 genetic algorithm framework with multiple game environments (Snake, TicTacToe, FlappyBird, Pong, Mastermind, Maze, Othello, Chess).

## Quick Start
```bash
cmake -B build -S .
cmake --build build --config Release

# Benchmark all games (headless, fast)
./build/genetic_algorithms.exe bench

# Single game with terminal UI
./build/genetic_algorithms.exe single Snake

# GUI mode with SDL2 visualization
./build/genetic_algorithms.exe gui Snake
```

## Games
| Game | Type | Input | Hidden | Max Steps |
|------|------|-------|--------|-----------|
| Snake | Policy | 8 | 16 | 200 |
| TicTacToe | Policy | 9 | 12 | 9 |
| FlappyBird | Policy | 6 | 10 | 500 |
| Pong | Policy | 8 | 12 | 500 |
| Mastermind | Policy | 4 | 8 | 40 |
| Maze | Policy | 12 | 12 | 200 |
| Othello | Scoring | 64 | 16 | 60 |
| Chess | Scoring | 25 | 16 | 100 |

## Architecture
- `include/ga/` - GA engine (population, selection, crossover, mutation)
- `include/games/` - Game interfaces and implementations
- `include/viz/` - TUI, GUI (SDL2), Replay visualization
- `src/main.cpp` - Entry point, game configs, run modes

## Dependencies
- C++17 compiler
- CMake 3.15+
- SDL2 2.30.10 (bundled in `deps/SDL2-2.30.10/`). To get it, download the Visual C++ development package from https://github.com/libsdl-org/SDL/releases and extract it to `deps/SDL2-2.30.10/`. CMake only enables the GUI when `deps/SDL2-2.30.10/include/SDL.h` exists (it links `lib/x64/SDL2.lib`); otherwise the build is TUI only.