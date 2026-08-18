# genetic_algorithms - Project Conventions

## Architecture
- C++17, CMake, SDL2 (bundled in deps/)
- Genetic algorithm engine in `include/ga/`
- Games in `include/games/` (Snake, TicTacToe, FlappyBird, Pong, Mastermind, Maze, Othello, Chess)
- Visualization: TUI (terminal), GUI (SDL2), Replay
- Entry: `src/main.cpp`

## Build & Run
```bash
cmake -B build -S .
cmake --build build --config Release

# Benchmark all games (headless)
./build/genetic_algorithms.exe bench

# Single game with TUI
./build/genetic_algorithms.exe single Snake

# GUI mode (requires SDL2)
./build/genetic_algorithms.exe gui Snake
```

## Code Style
- Modern C++17 (smart pointers, structured bindings, std::function)
- PascalCase for classes, snake_case for functions/variables
- Headers in `include/`, impl in `src/` (minimal, mostly header-only)
- RAII, no raw owning pointers

## Conventions
- Game interface in `include/games/game_interface.h`
- Network types: ScoringNetwork (value-based) vs PolicyNetwork (action-based)
- GA config in `include/ga/ga_engine.h`
- Parallel evaluation enabled by default
- Comments for non-obvious algorithms