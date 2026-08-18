#pragma once

#include "../ga/ga_engine.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace viz {

class TUI {
    int width_ = 80;
    int height_ = 25;
    std::vector<std::string> history_;
    bool enabled_ = true;

public:
    TUI(int w = 80, int h = 25) : width_(w), height_(h) {}

    void set_enabled(bool e) { enabled_ = e; }

    void clear() {
        if (!enabled_) return;
        #ifdef _WIN32
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hConsole != INVALID_HANDLE_VALUE && GetFileType(hConsole) == FILE_TYPE_CHAR) {
            system("cls");
        }
        #else
        std::cout << "\033[2J\033[H";
        #endif
    }

    void render_chart(const std::vector<float>& data, const std::vector<float>& avg_data,
                      const std::string& title, int chart_width = 60, int chart_height = 15) {
        if (!enabled_ || data.empty()) return;

        auto draw_bar_chart = [&](const std::vector<float>& values, const std::string& label, int row_offset) {
            float min_v = *std::min_element(values.begin(), values.end());
            float max_v = *std::max_element(values.begin(), values.end());
            float range = max_v - min_v;
            if (range < 1e-6f) range = 1.0f;

            int n = std::min(static_cast<int>(values.size()), chart_width);
            int start = std::max(0, static_cast<int>(values.size()) - n);

            std::cout << "\n " << label << "\n";
            for (int row = chart_height - 1; row >= 0; --row) {
                float threshold = min_v + range * static_cast<float>(row) / chart_height;
                std::string line = " |";
                for (int i = 0; i < n; ++i) {
                    float v = values[start + i];
                    line += (v >= threshold) ? "\xE2\x96\x88" : " ";
                }
                line += "|";
                if (row == chart_height - 1)
                    line += " " + format_float(max_v);
                else if (row == 0)
                    line += " " + format_float(min_v);
                std::cout << line << "\n";
            }
            std::cout << " +" << std::string(n, '-') << "+\n";
        };

        draw_bar_chart(data, title + " [Best]", 0);
        if (!avg_data.empty())
            draw_bar_chart(avg_data, title + " [Avg]", chart_height + 2);
    }

    void render_summary(const ga::GAResult& result, const std::string& game_name, int gen) {
        if (!enabled_) return;

        clear();
        std::cout << "\n";
        std::cout << " ================================================\n";
        std::cout << "  GENETIC ALGORITHM - " << game_name << "\n";
        std::cout << " ================================================\n\n";

        std::cout << " Generation:  " << gen << "\n";
        std::cout << " Best:        " << std::fixed << std::setprecision(4) << result.best_fitness << "\n";
        std::cout << " Average:     " << std::fixed << std::setprecision(4) << result.avg_fitness << "\n";
        std::cout << " Time:        " << std::fixed << std::setprecision(1) << result.elapsed_ms << " ms\n";

        if (result.generations_run > 0) {
            double gps = result.generations_run / (result.elapsed_ms / 1000.0);
            std::cout << " Gen/sec:    " << std::fixed << std::setprecision(0) << gps << "\n";
        }

        std::cout << "\n";

        render_chart(result.fitness_history, result.avg_history, "Fitness Progression");
    }

    void render_benchmark_table(const std::vector<std::pair<std::string, ga::GAResult>>& results) {
        if (!enabled_) return;

        clear();
        std::cout << "\n";
        std::cout << " ================================================================\n";
        std::cout << "  BENCHMARK RESULTS\n";
        std::cout << " ================================================================\n\n";

        std::cout << " Game           | Best      | Avg       | Gens | Time(ms) | Gen/s\n";
        std::cout << " ---------------+-----------+-----------+------+----------+--------\n";

        for (const auto& [name, result] : results) {
            double gps = result.generations_run / (result.elapsed_ms / 1000.0 + 1e-6);
            std::cout << " " << std::setw(15) << std::left << name
                      << "| " << std::setw(9) << std::right << std::fixed << std::setprecision(2) << result.best_fitness
                      << " | " << std::setw(9) << result.avg_fitness
                      << " | " << std::setw(4) << result.generations_run
                      << " | " << std::setw(8) << std::setprecision(1) << result.elapsed_ms
                      << " | " << std::setw(6) << std::setprecision(0) << gps << "\n";
        }

        std::cout << "\n ================================================================\n";
    }

private:
    std::string format_float(float v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << v;
        return ss.str();
    }
};

} // namespace viz
