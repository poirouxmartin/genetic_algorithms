#pragma once

#ifdef HAS_SDL2

#include "stb/stb_truetype.h"
#include "../ga/ga_engine.h"
#include "../ga/game_registry.h"
#include "../viz/replay.h"
#include <SDL.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>

namespace viz {

static const SDL_Color COL_BG     {27,  30,  46,  255};
static const SDL_Color COL_PANEL  {37,  40,  56,  255};
static const SDL_Color COL_BORDER {55,  58,  75,  255};
static const SDL_Color COL_TEXT   {226, 228, 234, 255};
static const SDL_Color COL_DIM    {138, 141, 158, 255};
static const SDL_Color COL_ACCENT {123, 159, 212, 255};
static const SDL_Color COL_TEAL   {107, 191, 160, 255};
static const SDL_Color COL_GOLD   {212, 184, 122, 255};
static const SDL_Color COL_RED    {212, 122, 122, 255};
static const SDL_Color COL_GRID   {42,  45,  62,  255};

class GUI {
public:
    enum class Action { None, PauseResume, Stop, Watch, Back, Menu };

    struct Slider {
        const char* label;
        float* value;
        float min_val, max_val;
        float step;
    };

private:
    struct Button {
        int x, y, w, h;
        std::string label;
        SDL_Color bg, fg;
    };

    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    int width_ = 1200;
    int height_ = 800;
    bool initialized_ = false;
    bool quit_ = false;
    Action pending_action_ = Action::None;

    stbtt_fontinfo font_info_{};
    unsigned char* ttf_buffer_ = nullptr;
    stbtt_packedchar font_chars_[96]{};
    SDL_Texture* font_texture_ = nullptr;
    int atlas_w_ = 1024;
    int atlas_h_ = 512;
    float font_size_ = 16.0f;
    float char_height_ = 20.0f;

    Button btn_a_{};
    Button btn_b_{};
    Button btn_menu_{};

public:
    GUI(int w = 1200, int h = 800) : width_(w), height_(h) {}
    ~GUI() { cleanup(); }

    bool init() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) return false;
        window_ = SDL_CreateWindow("GA Visualizer",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            width_, height_, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
        if (!window_) return false;
        renderer_ = SDL_CreateRenderer(window_, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer_) return false;
        SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
        initialized_ = true;
        init_font();
        return true;
    }

    void cleanup() {
        if (font_texture_) { SDL_DestroyTexture(font_texture_); font_texture_ = nullptr; }
        delete[] ttf_buffer_; ttf_buffer_ = nullptr;
        if (renderer_) { SDL_DestroyRenderer(renderer_); renderer_ = nullptr; }
        if (window_) { SDL_DestroyWindow(window_); window_ = nullptr; }
        if (initialized_) { SDL_Quit(); initialized_ = false; }
    }

    bool should_close() { return quit_ || !initialized_; }
    Action consume_action() { Action a = pending_action_; pending_action_ = Action::None; return a; }

    bool render_config(int& game_idx, const std::vector<GameConfig>& configs,
                       std::vector<Slider>& sliders, bool& run_clicked) {
        if (!initialized_ || quit_) return false;
        SDL_GetWindowSize(window_, &width_, &height_);
        clear_bg();

        const auto& gc = configs[game_idx];

        fill_rect(15, 8, width_ - 30, 50, COL_PANEL);
        draw_rect_outline(15, 8, width_ - 30, 50, COL_BORDER);
        draw_text("GA Visualizer", 25, 14, COL_DIM);
        draw_text(gc.display_name, 25, 34, COL_TEXT);

        int nav_x = width_ - 200;
        draw_button({nav_x, 14, 40, 28, "<", COL_BORDER, COL_TEXT});
        draw_button({nav_x + 50, 14, 40, 28, ">", COL_BORDER, COL_TEXT});
        draw_text(std::to_string(game_idx + 1) + "/" + std::to_string(configs.size()),
                  nav_x + 96, 20, COL_DIM);

        int y_start = 80;
        int slider_h = 48;

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) { quit_ = true; return false; }
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) { quit_ = true; return false; }
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                int mx = e.button.x, my = e.button.y;

                if (mx >= nav_x && mx < nav_x + 40 && my >= 14 && my <= 42) {
                    game_idx = (game_idx - 1 + static_cast<int>(configs.size())) % static_cast<int>(configs.size());
                }
                if (mx >= nav_x + 50 && mx < nav_x + 90 && my >= 14 && my <= 42) {
                    game_idx = (game_idx + 1) % static_cast<int>(configs.size());
                }

                for (size_t i = 0; i < sliders.size(); ++i) {
                    int sy = y_start + static_cast<int>(i) * slider_h;
                    int sx = 220, sw = width_ - 260;
                    if (mx >= sx && mx <= sx + sw && my >= sy && my <= sy + 28) {
                        float ratio = static_cast<float>(mx - sx) / sw;
                        float nv = sliders[i].min_val + ratio * (sliders[i].max_val - sliders[i].min_val);
                        nv = std::round(nv / sliders[i].step) * sliders[i].step;
                        *sliders[i].value = std::clamp(nv, sliders[i].min_val, sliders[i].max_val);
                    }
                }
                int btn_y = y_start + static_cast<int>(sliders.size()) * slider_h + 20;
                if (mx >= 220 && mx <= 420 && my >= btn_y && my <= btn_y + 44) run_clicked = true;
            }
            if (e.type == SDL_MOUSEWHEEL) {
                int mx = 0, my = 0;
                SDL_GetMouseState(&mx, &my);
                for (size_t i = 0; i < sliders.size(); ++i) {
                    int sy = y_start + static_cast<int>(i) * slider_h;
                    if (my >= sy && my <= sy + 28 && mx >= 220) {
                        *sliders[i].value = std::clamp(
                            *sliders[i].value + e.wheel.y * sliders[i].step,
                            sliders[i].min_val, sliders[i].max_val);
                        break;
                    }
                }
            }
        }

        for (size_t i = 0; i < sliders.size(); ++i) {
            int sy = y_start + static_cast<int>(i) * slider_h;
            int sx = 220, sw = width_ - 260;
            draw_text(sliders[i].label, 20, sy + 4, COL_TEXT);
            fill_rect(sx, sy + 4, sw, 14, COL_PANEL);
            draw_rect_outline(sx, sy + 4, sw, 14, COL_BORDER);
            float ratio = (*sliders[i].value - sliders[i].min_val) / (sliders[i].max_val - sliders[i].min_val);
            int thumb_x = sx + static_cast<int>(ratio * sw);
            fill_rect(thumb_x - 3, sy, 6, 22, COL_ACCENT);
            draw_text(fmt(*sliders[i].value, (sliders[i].step < 1.0f ? 2 : 0)), thumb_x + 10, sy, COL_TEXT);
        }

        int btn_y = y_start + static_cast<int>(sliders.size()) * slider_h + 20;
        draw_button({220, btn_y, 200, 44, "Run", COL_ACCENT, COL_TEXT});

        SDL_RenderPresent(renderer_);
        return !run_clicked;
    }

    void render_sim(const ga::GAResult& result, const std::string& game_name, int gen, int max_gen,
                    bool is_running, const float* nn_genes = nullptr, size_t input_size = 0,
                    size_t hidden_size = 0, size_t output_size = 0, const std::string& game_state = "",
                    const games::GameInterface* game_obj = nullptr) {
        if (!initialized_) return;
        SDL_GetWindowSize(window_, &width_, &height_);
        clear_bg();
        process_events();

        int header_h = 56;
        int chart_h = (height_ - header_h) * 38 / 100;
        int bottom_y = header_h + chart_h + 8;
        int bottom_h = height_ - bottom_y - 8;
        int margin = 10;
        int total_w = width_ - margin * 2;

        draw_header_sim(game_name, gen, result, max_gen, is_running);
        draw_chart(result.fitness_history, result.avg_history,
                   margin, header_h + 4, total_w, chart_h);

        bool has_nn = nn_genes && input_size > 0 && hidden_size > 0 && output_size > 0;
        bool has_game = !game_state.empty();
        int col_w;

        if (has_nn && has_game) {
            col_w = (total_w - margin * 2) / 3;
            draw_stats(result, margin, bottom_y, col_w, bottom_h);
            if (game_obj && game_obj->has_sdl_render()) {
                fill_rect(margin + col_w + margin, bottom_y, col_w, bottom_h, COL_BG);
                draw_rect_outline(margin + col_w + margin, bottom_y, col_w, bottom_h, COL_BORDER);
                draw_text("Game State", margin + col_w + margin + 10, bottom_y + 6, COL_GOLD);
                game_obj->render_sdl(renderer_, margin + col_w + margin + 8,
                                     bottom_y + 8 + ch() + 4, col_w - 16, bottom_h - 16 - ch());
            } else {
                draw_game_panel(game_state, margin + col_w + margin, bottom_y, col_w, bottom_h);
            }
            draw_nn(nn_genes, input_size, hidden_size, output_size,
                    margin + (col_w + margin) * 2, bottom_y, col_w, bottom_h);
        } else if (has_game) {
            col_w = (total_w - margin) / 2;
            draw_stats(result, margin, bottom_y, col_w, bottom_h);
            if (game_obj && game_obj->has_sdl_render()) {
                fill_rect(margin + col_w + margin, bottom_y, col_w, bottom_h, COL_BG);
                draw_rect_outline(margin + col_w + margin, bottom_y, col_w, bottom_h, COL_BORDER);
                draw_text("Game State", margin + col_w + margin + 10, bottom_y + 6, COL_GOLD);
                game_obj->render_sdl(renderer_, margin + col_w + margin + 8,
                                     bottom_y + 8 + ch() + 4, col_w - 16, bottom_h - 16 - ch());
            } else {
                draw_game_panel(game_state, margin + col_w + margin, bottom_y, col_w, bottom_h);
            }
        } else if (has_nn) {
            col_w = (total_w - margin) / 2;
            draw_stats(result, margin, bottom_y, col_w, bottom_h);
            draw_nn(nn_genes, input_size, hidden_size, output_size,
                    margin + col_w + margin, bottom_y, col_w, bottom_h);
        } else {
            draw_stats(result, margin, bottom_y, total_w / 2, bottom_h);
            draw_diversity(result, margin + total_w / 2 + margin, bottom_y, total_w / 2, bottom_h);
        }
        SDL_RenderPresent(renderer_);
    }

    void render_watch(const std::string& game_state, const std::string& game_name,
                      int step, int max_steps, float fitness, float best_fitness,
                      const games::GameInterface* game_obj = nullptr) {
        if (!initialized_) return;
        SDL_GetWindowSize(window_, &width_, &height_);
        clear_bg();
        process_events();

        fill_rect(15, 8, width_ - 30, 50, COL_PANEL);
        draw_rect_outline(15, 8, width_ - 30, 50, COL_BORDER);
        draw_text("Watching: " + game_name, 25, 14, COL_TEXT);
        draw_text("Step: " + std::to_string(step) + " / " + std::to_string(max_steps)
                  + "    Fitness: " + fmt(fitness)
                  + "    Best: " + fmt(best_fitness), 25, 32, COL_TEAL);

        int board_y = 70, board_h = height_ - 70 - 60;
        fill_rect(15, board_y, width_ - 30, board_h, COL_BG);
        draw_rect_outline(15, board_y, width_ - 30, board_h, COL_BORDER);
        if (game_obj && game_obj->has_sdl_render()) {
            draw_text("Game", 25, board_y + 4, COL_GOLD);
            game_obj->render_sdl(renderer_, 25, board_y + 10 + ch(), width_ - 50, board_h - 20 - ch());
        } else {
            draw_game_state(game_state, 25, board_y + 10, width_ - 50, board_h - 20);
        }

        int bar_y = height_ - 52;
        fill_rect(15, bar_y, width_ - 30, 44, COL_PANEL);
        draw_rect_outline(15, bar_y, width_ - 30, 44, COL_BORDER);
        btn_a_ = {25, bar_y + 7, 100, 30, "Back", COL_DIM, COL_TEXT};
        draw_button(btn_a_);
        btn_menu_ = {135, bar_y + 7, 100, 30, "Menu", COL_RED, COL_TEXT};
        draw_button(btn_menu_);
        SDL_RenderPresent(renderer_);
    }

    void render_benchmark(const std::vector<std::pair<std::string, ga::GAResult>>& results) {
        if (!initialized_) return;
        SDL_GetWindowSize(window_, &width_, &height_);
        clear_bg();
        process_events();
        draw_benchmark_bars(results);
        SDL_RenderPresent(renderer_);
    }

    void render_replay(const ReplayData& replay, int frame_idx, float speed) {
        if (!initialized_ || replay.frames.empty()) return;
        SDL_GetWindowSize(window_, &width_, &height_);
        clear_bg();
        process_events();

        int fi = std::clamp(frame_idx, 0, static_cast<int>(replay.frames.size()) - 1);
        const auto& frame = replay.frames[fi];

        fill_rect(15, 8, width_ - 30, 54, COL_PANEL);
        draw_rect_outline(15, 8, width_ - 30, 54, COL_BORDER);
        draw_text("Replay: " + replay.game_name, 25, 14, COL_TEXT);
        draw_text("Step: " + std::to_string(frame.step) + "/" + std::to_string(replay.total_steps)
                  + "  Fitness: " + fmt(frame.fitness)
                  + "  Speed: " + fmt(speed, 1) + "x",
                  25, 14 + ch() + 4, COL_GOLD);

        int bw = width_ - 30, bh = height_ / 2 - 50;
        fill_rect(15, 70, bw, bh, COL_BG);
        draw_rect_outline(15, 70, bw, bh, COL_BORDER);
        draw_game_state(frame.state, 25, 80, bw - 20, bh - 20);

        int prog_y = 70 + bh + 8;
        fill_rect(15, prog_y, bw, 18, COL_PANEL);
        int pw2 = static_cast<int>((float)fi / std::max(1, static_cast<int>(replay.frames.size()) - 1) * bw);
        fill_rect(15, prog_y, pw2, 18, COL_ACCENT);
        draw_rect_outline(15, prog_y, bw, 18, COL_BORDER);

        int iy = prog_y + 26, pw3 = (bw - 10) / 2;
        fill_rect(15, iy, pw3, height_ - iy - 10, COL_PANEL);
        draw_rect_outline(15, iy, pw3, height_ - iy - 10, COL_BORDER);
        draw_text("Frame Info", 25, iy + 6, COL_GOLD);
        draw_text("Action: " + std::to_string(frame.action), 25, iy + 6 + ch() + 8, COL_TEXT);
        draw_text("Step:   " + std::to_string(frame.step), 25, iy + 6 + 2*(ch() + 8), COL_TEXT);
        draw_text("Fitness: " + fmt(frame.fitness), 25, iy + 6 + 3*(ch() + 8), COL_TEAL);

        int hx = 25 + pw3;
        fill_rect(hx, iy, pw3, height_ - iy - 10, COL_PANEL);
        draw_rect_outline(hx, iy, pw3, height_ - iy - 10, COL_BORDER);
        draw_text("Fitness Over Time", hx + 10, iy + 6, COL_GOLD);
        draw_replay_curve(replay, fi, hx + 10, iy + 6 + ch() + 8, pw3 - 20, height_ - iy - 20 - ch());
        SDL_RenderPresent(renderer_);
    }

    void render_dashboard(const std::vector<std::pair<std::string, ga::GAResult>>& results) {
        if (!initialized_ || results.empty()) return;
        SDL_GetWindowSize(window_, &width_, &height_);
        clear_bg();
        process_events();
        draw_text("Dashboard", 20, 10, COL_TEXT);

        int n = static_cast<int>(results.size());
        int cols = (n <= 4) ? 2 : 3;
        int rows = (n + cols - 1) / cols;
        int pad = 10;
        int hdr_h = 10 + ch() + 10;
        int cw = (width_ - pad * (cols + 1)) / cols;
        int cell_h = (height_ - hdr_h - pad * (rows + 1)) / rows;

        SDL_Color pal[] = {{107,191,160,255},{123,159,212,255},{212,184,122,255},{212,122,122,255},
                           {170,140,210,255},{100,190,190,255},{210,160,100,255},{140,200,140,255}};

        for (int i = 0; i < n; ++i) {
            int col2 = i % cols, row2 = i / cols;
            int x = pad + col2 * (cw + pad);
            int y = hdr_h + pad + row2 * (cell_h + pad);
            const auto& [name, r] = results[i];
            fill_rect(x, y, cw, cell_h, COL_PANEL);
            draw_rect_outline(x, y, cw, cell_h, COL_BORDER);
            draw_text(name + " Best:" + fmt(r.best_fitness), x + 8, y + 6, pal[i % 8]);
            if (r.fitness_history.empty()) continue;
            float fmin = *std::min_element(r.fitness_history.begin(), r.fitness_history.end());
            float fmax = *std::max_element(r.fitness_history.begin(), r.fitness_history.end());
            float fr = std::max(fmax - fmin, 1.0f);
            int ml = 8, mt = 8 + ch() + 4;
            int px2 = x + ml, py2 = y + mt, pw2 = cw - ml * 2, ph2 = cell_h - mt - 8;
            int nh = static_cast<int>(r.fitness_history.size());
            int ns = std::min(nh, pw2);
            int start = std::max(0, nh - ns);
            SDL_SetRenderDrawColor(renderer_, pal[i % 8].r, pal[i % 8].g, pal[i % 8].b, 255);
            for (int j = 1; j < ns; ++j) {
                int x0 = px2 + (j-1) * pw2 / std::max(ns-1, 1), x1 = px2 + j * pw2 / std::max(ns-1, 1);
                int y0 = py2 + ph2 - static_cast<int>((r.fitness_history[start+j-1] - fmin) / fr * ph2);
                int y1 = py2 + ph2 - static_cast<int>((r.fitness_history[start+j] - fmin) / fr * ph2);
                SDL_RenderDrawLine(renderer_, x0, y0, x1, y1);
            }
        }
        SDL_RenderPresent(renderer_);
    }

    void process_events() {
        pending_action_ = Action::None;
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) { quit_ = true; return; }
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) {
                pending_action_ = Action::Menu;
                return;
            }
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                int mx = e.button.x, my = e.button.y;
                if (check_btn(btn_menu_, mx, my)) pending_action_ = Action::Menu;
                else if (check_btn(btn_a_, mx, my)) pending_action_ = Action::PauseResume;
                else if (check_btn(btn_b_, mx, my)) pending_action_ = Action::Watch;
            }
        }
    }

private:
    void clear_bg() {
        SDL_SetRenderDrawColor(renderer_, COL_BG.r, COL_BG.g, COL_BG.b, 255);
        SDL_RenderClear(renderer_);
    }

    bool init_font() {
        const char* paths[] = {
            "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\tahoma.ttf",
        };
        for (const char* path : paths) {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f.is_open()) continue;
            auto sz = f.tellg();
            f.seekg(0, std::ios::beg);
            ttf_buffer_ = new unsigned char[static_cast<size_t>(sz)];
            f.read(reinterpret_cast<char*>(ttf_buffer_), sz);
            if (stbtt_InitFont(&font_info_, ttf_buffer_,
                                stbtt_GetFontOffsetForIndex(ttf_buffer_, 0)))
                break;
            delete[] ttf_buffer_;
            ttf_buffer_ = nullptr;
        }
        if (!ttf_buffer_) return false;

        auto* bitmap = new unsigned char[atlas_w_ * atlas_h_];
        std::memset(font_chars_, 0, sizeof(font_chars_));
        stbtt_pack_context pc;
        stbtt_PackBegin(&pc, bitmap, atlas_w_, atlas_h_, 0, 1, nullptr);
        stbtt_PackSetOversampling(&pc, 2, 2);
        stbtt_PackFontRange(&pc, ttf_buffer_, 0, font_size_, 32, 96, font_chars_);
        stbtt_PackEnd(&pc);

        std::vector<uint32_t> pixels(atlas_w_ * atlas_h_);
        for (int i = 0; i < atlas_w_ * atlas_h_; i++)
            pixels[i] = (static_cast<uint32_t>(bitmap[i]) << 24) | 0x00FFFFFF;
        delete[] bitmap;

        font_texture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888,
                                           SDL_TEXTUREACCESS_STATIC, atlas_w_, atlas_h_);
        if (!font_texture_) return false;
        SDL_UpdateTexture(font_texture_, nullptr, pixels.data(), atlas_w_ * 4);
        SDL_SetTextureBlendMode(font_texture_, SDL_BLENDMODE_BLEND);

        int ascent, descent, lg;
        stbtt_GetFontVMetrics(&font_info_, &ascent, &descent, &lg);
        float scale = stbtt_ScaleForPixelHeight(&font_info_, font_size_);
        char_height_ = (ascent - descent + lg) * scale;
        return true;
    }

    int ch() const { return static_cast<int>(char_height_); }
    int tw(const std::string& t) const {
        float x = 0;
        for (unsigned char c : t)
            if (c >= 32 && c < 128) x += font_chars_[c - 32].xadvance;
        return static_cast<int>(x);
    }

    void draw_text(const std::string& text, int x, int y, SDL_Color color) {
        if (!font_texture_) return;
        SDL_SetTextureColorMod(font_texture_, color.r, color.g, color.b);
        SDL_SetTextureAlphaMod(font_texture_, color.a);
        float fx = static_cast<float>(x), fy = static_cast<float>(y);
        for (unsigned char c : text) {
            if (c < 32 || c >= 128) continue;
            stbtt_aligned_quad q;
            stbtt_GetPackedQuad(font_chars_, atlas_w_, atlas_h_, c - 32, &fx, &fy, &q, 1);
            SDL_Rect src = {static_cast<int>(q.s0 * atlas_w_), static_cast<int>(q.t0 * atlas_h_),
                            static_cast<int>((q.s1 - q.s0) * atlas_w_), static_cast<int>((q.t1 - q.t0) * atlas_h_)};
            SDL_Rect dst = {static_cast<int>(q.x0), static_cast<int>(q.y0),
                            std::max(1, (int)(q.x1 - q.x0)), std::max(1, (int)(q.y1 - q.y0))};
            SDL_RenderCopy(renderer_, font_texture_, &src, &dst);
        }
    }

    void fill_rect(int x, int y, int w, int h, SDL_Color c) {
        SDL_SetRenderDrawColor(renderer_, c.r, c.g, c.b, c.a);
        SDL_Rect r = {x, y, w, h};
        SDL_RenderFillRect(renderer_, &r);
    }

    void draw_rect_outline(int x, int y, int w, int h, SDL_Color c) {
        SDL_SetRenderDrawColor(renderer_, c.r, c.g, c.b, c.a);
        SDL_Rect r = {x, y, w, h};
        SDL_RenderDrawRect(renderer_, &r);
    }

    void draw_button(const Button& b) {
        fill_rect(b.x, b.y, b.w, b.h, b.bg);
        draw_rect_outline(b.x, b.y, b.w, b.h, COL_BORDER);
        int t = tw(b.label);
        draw_text(b.label, b.x + (b.w - t) / 2, b.y + (b.h - ch()) / 2, b.fg);
    }

    bool check_btn(const Button& b, int mx, int my) {
        return b.w > 0 && mx >= b.x && mx < b.x + b.w && my >= b.y && my < b.y + b.h;
    }

    void draw_header_sim(const std::string& game_name, int gen, const ga::GAResult& result,
                         int max_gen, bool is_running) {
        fill_rect(15, 8, width_ - 30, 50, COL_PANEL);
        draw_rect_outline(15, 8, width_ - 30, 50, COL_BORDER);

        draw_text(game_name, 25, 14, COL_TEXT);
        draw_text("Gen: " + std::to_string(gen) + (max_gen > 0 ? " / " + std::to_string(max_gen) : ""),
                  25, 32, COL_GOLD);

        draw_text("Best: " + fmt(result.best_fitness), 250, 14, COL_TEAL);
        draw_text("Avg: " + fmt(result.avg_fitness), 250, 32, COL_ACCENT);

        double gps = result.generations_run / (result.elapsed_ms / 1000.0 + 1e-6);
        draw_text("Gen/s: " + fmt((float)gps, 0), 450, 14, COL_DIM);

        if (max_gen > 0) {
            int bx = 450, by = 38, bw = width_ - 560, bh = 16;
            if (bw > 40) {
                fill_rect(bx, by, bw, bh, COL_PANEL);
                draw_rect_outline(bx, by, bw, bh, COL_BORDER);
                int fill = std::max(1, (int)((float)gen / max_gen * (bw - 2)));
                fill_rect(bx + 1, by + 1, fill, bh - 2, COL_ACCENT);
                draw_text(fmt((float)gen / max_gen * 100.0f, 0) + "%", bx + bw / 2 - 12, by + 1, COL_TEXT);
            }
        }

        int rpx = width_ - 340;
        if (is_running) {
            btn_a_ = {rpx, 16, 70, 28, "Pause", COL_ACCENT, COL_TEXT};
            btn_b_ = {rpx + 80, 16, 70, 28, "Watch", COL_TEAL, COL_TEXT};
        } else {
            btn_a_ = {rpx, 16, 70, 28, "Resume", COL_ACCENT, COL_TEXT};
            btn_b_ = {rpx + 80, 16, 70, 28, "Watch", COL_TEAL, COL_TEXT};
        }
        draw_button(btn_a_);
        draw_button(btn_b_);

        btn_menu_ = {rpx + 160, 16, 70, 28, "Menu", COL_RED, COL_TEXT};
        draw_button(btn_menu_);
    }

    void draw_chart(const std::vector<float>& best, const std::vector<float>& avg,
                    int x, int y, int w, int h) {
        if (best.empty()) return;
        fill_rect(x, y, w, h, COL_PANEL);
        draw_rect_outline(x, y, w, h, COL_BORDER);
        draw_text("Fitness", x + 10, y + 4, COL_DIM);

        float all_min = 1e30f, all_max = -1e30f;
        for (float v : best) { all_min = std::min(all_min, v); all_max = std::max(all_max, v); }
        for (float v : avg)  { all_min = std::min(all_min, v); all_max = std::max(all_max, v); }
        float range = std::max(all_max - all_min, 1e-6f);

        int ml = 70, mr = 20, mt = 8 + ch() + 2, mb = 35;
        int px2 = x + ml, py2 = y + mt, pw2 = w - ml - mr, ph2 = h - mt - mb;

        for (int i = 0; i <= 6; ++i) {
            int gy = py2 + ph2 - i * ph2 / 6;
            SDL_SetRenderDrawColor(renderer_, COL_GRID.r, COL_GRID.g, COL_GRID.b, 255);
            SDL_RenderDrawLine(renderer_, px2, gy, px2 + pw2, gy);
            draw_text(fmt(all_min + range * i / 6.0f, 1), x + 2, gy - ch() / 2, COL_DIM);
        }

        int n = std::min(static_cast<int>(best.size()), pw2);
        int start = std::max(0, static_cast<int>(best.size()) - n);
        auto mx2 = [&](int i) { return px2 + i * pw2 / std::max(n - 1, 1); };
        auto my2 = [&](float v) { return py2 + ph2 - static_cast<int>((v - all_min) / range * ph2); };

        auto thick_line = [&](int x0, int y0, int x1, int y1, SDL_Color c) {
            SDL_SetRenderDrawColor(renderer_, c.r, c.g, c.b, c.a);
            SDL_RenderDrawLine(renderer_, x0, y0, x1, y1);
            SDL_RenderDrawLine(renderer_, x0, y0 - 1, x1, y1 - 1);
            SDL_RenderDrawLine(renderer_, x0, y0 + 1, x1, y1 + 1);
        };

        if (avg.size() > 1)
            for (int i = 1; i < n; ++i)
                thick_line(mx2(i-1), my2(avg[start+i-1]), mx2(i), my2(avg[start+i]), COL_DIM);

        for (int i = 1; i < n; ++i)
            thick_line(mx2(i-1), my2(best[start+i-1]), mx2(i), my2(best[start+i]), COL_ACCENT);

        fill_rect(mx2(n-1) - 4, my2(best.back()) - 4, 8, 8, COL_TEXT);
        draw_text(fmt(best.back(), 1), mx2(n-1) + 8, my2(best.back()) - ch() / 2, COL_TEXT);

        int ly = py2 + ph2 + 4;
        fill_rect(px2, ly + 4, 14, 4, COL_ACCENT);
        draw_text("Best", px2 + 18, ly, COL_ACCENT);
        if (!avg.empty()) {
            fill_rect(px2 + 80, ly + 4, 14, 4, COL_DIM);
            draw_text("Avg", px2 + 98, ly, COL_DIM);
        }

        if (n > 10)
            for (int i = 0; i < 5; ++i) {
                int xi = start + i * (n - 1) / 4;
                if (xi >= 0 && xi < static_cast<int>(best.size()))
                    draw_text(std::to_string(xi + 1), mx2(xi - start) - 8, py2 + ph2 + 8, COL_DIM);
            }
    }

    void draw_stats(const ga::GAResult& result, int sx, int sy, int sw, int sh) {
        fill_rect(sx, sy, sw, sh, COL_PANEL);
        draw_rect_outline(sx, sy, sw, sh, COL_BORDER);
        draw_text("Statistics", sx + 12, sy + 8, COL_GOLD);

        double gps = result.generations_run / (result.elapsed_ms / 1000.0 + 1e-6);
        int lh = ch() + 4, lx = sx + 12, ly = sy + 8 + ch() + 8, vx = sx + sw / 2;

        draw_text("Generations:", lx, ly, COL_DIM);
        draw_text(std::to_string(result.generations_run), vx, ly, COL_TEXT);
        draw_text("Elapsed:", lx, ly + lh, COL_DIM);
        draw_text(fmt(result.elapsed_ms / 1000.0f, 1) + "s", vx, ly + lh, COL_TEXT);
        draw_text("Gen/sec:", lx, ly + 2*lh, COL_DIM);
        draw_text(fmt((float)gps, 0), vx, ly + 2*lh, COL_ACCENT);
        draw_text("Best:", lx, ly + 3*lh, COL_DIM);
        draw_text(fmt(result.best_fitness), vx, ly + 3*lh, COL_TEAL);
        draw_text("Avg:", lx, ly + 4*lh, COL_DIM);
        draw_text(fmt(result.avg_fitness), vx, ly + 4*lh, COL_ACCENT);

        if (result.generations_run > 1 && !result.fitness_history.empty()) {
            float delta = result.fitness_history.back() - result.fitness_history.front();
            draw_text("Delta:", lx, ly + 5*lh, COL_DIM);
            draw_text(fmt(delta), vx, ly + 5*lh, delta >= 0 ? COL_TEAL : COL_RED);
        }
        if (!result.diversity_history.empty()) {
            draw_text("Diversity:", lx, ly + 6*lh, COL_DIM);
            draw_text(fmt(result.diversity_history.back()), vx, ly + 6*lh, COL_GOLD);
        }
    }

    void draw_diversity(const ga::GAResult& result, int sx, int sy, int sw, int sh) {
        if (result.diversity_history.empty()) return;
        fill_rect(sx, sy, sw, sh, COL_PANEL);
        draw_rect_outline(sx, sy, sw, sh, COL_BORDER);
        draw_text("Diversity", sx + 10, sy + 6, COL_GOLD);

        float dmin = *std::min_element(result.diversity_history.begin(), result.diversity_history.end());
        float dmax = *std::max_element(result.diversity_history.begin(), result.diversity_history.end());
        float dr = std::max(dmax - dmin, 1.0f);

        int ml = 50, mt = 8 + ch() + 4, mb = 10, mr = 10;
        int px2 = sx + ml, py2 = sy + mt, pw2 = sw - ml - mr, ph2 = sh - mt - mb;

        for (int i = 0; i <= 4; ++i) {
            int gy = py2 + ph2 - i * ph2 / 4;
            SDL_SetRenderDrawColor(renderer_, COL_GRID.r, COL_GRID.g, COL_GRID.b, 255);
            SDL_RenderDrawLine(renderer_, px2, gy, px2 + pw2, gy);
            draw_text(fmt(dmin + dr * i / 4.0f, 1), sx + 2, gy - ch() / 2, COL_DIM);
        }

        int n = std::min(static_cast<int>(result.diversity_history.size()), pw2);
        int start = std::max(0, static_cast<int>(result.diversity_history.size()) - n);
        for (int i = 1; i < n; ++i) {
            int x0 = px2 + (i-1) * pw2 / std::max(n-1, 1), x1 = px2 + i * pw2 / std::max(n-1, 1);
            int y0 = py2 + ph2 - static_cast<int>((result.diversity_history[start+i-1] - dmin) / dr * ph2);
            int y1 = py2 + ph2 - static_cast<int>((result.diversity_history[start+i] - dmin) / dr * ph2);
            SDL_SetRenderDrawColor(renderer_, COL_GOLD.r, COL_GOLD.g, COL_GOLD.b, 220);
            SDL_RenderDrawLine(renderer_, x0, y0, x1, y1);
            SDL_RenderDrawLine(renderer_, x0, y0 - 1, x1, y1 - 1);
        }
    }

    void draw_nn(const float* genes, size_t in, size_t hid, size_t out,
                 int sx, int sy, int sw, int sh) {
        fill_rect(sx, sy, sw, sh, COL_PANEL);
        draw_rect_outline(sx, sy, sw, sh, COL_BORDER);
        draw_text("Neural Network", sx + 10, sy + 6, COL_GOLD);

        int mx2 = sx + 40, my2 = sy + 8 + ch() + 8;
        int mw = sw - 80, mh = sh - 16 - ch() - 8;
        auto node_y = [&](int ls, int idx) { return my2 + mh / 2 + (idx - ls / 2) * 22; };
        int in_x = mx2, hid_x = mx2 + mw / 2, out_x = mx2 + mw;

        for (size_t h = 0; h < hid; ++h)
            for (size_t i = 0; i < in; ++i) {
                float w = genes[h * in + i];
                Uint8 r = w > 0 ? (Uint8)std::min(255.0f, std::abs(w) * 180.0f) : 0;
                Uint8 g = w < 0 ? (Uint8)std::min(255.0f, std::abs(w) * 180.0f) : 0;
                Uint8 a = (Uint8)std::min(255.0f, 20.0f + std::abs(w) * 80.0f);
                SDL_SetRenderDrawColor(renderer_, r, g, 0, a);
                SDL_RenderDrawLine(renderer_, in_x + 4, node_y((int)in, (int)i), hid_x - 4, node_y((int)hid, (int)h));
            }

        size_t ho = in * hid + hid;
        for (size_t o = 0; o < out; ++o)
            for (size_t h = 0; h < hid; ++h) {
                float w = genes[ho + o * hid + h];
                Uint8 r = w > 0 ? (Uint8)std::min(255.0f, std::abs(w) * 180.0f) : 0;
                Uint8 g = w < 0 ? (Uint8)std::min(255.0f, std::abs(w) * 180.0f) : 0;
                Uint8 a = (Uint8)std::min(255.0f, 20.0f + std::abs(w) * 80.0f);
                SDL_SetRenderDrawColor(renderer_, r, g, 0, a);
                SDL_RenderDrawLine(renderer_, hid_x + 4, node_y((int)hid, (int)h), out_x - 4, node_y((int)out, (int)o));
            }

        auto draw_node = [&](int nx, int ny, float bias, const char* label) {
            Uint8 r = (Uint8)(bias > 0 ? 80 : 140);
            Uint8 b2 = (Uint8)(bias < 0 ? 180 : std::min(255.0f, 80.0f + std::abs(bias) * 80.0f));
            fill_rect(nx - 5, ny - 5, 10, 10, {r, 130, b2, 255});
            draw_text(label, nx + 8, ny - ch() / 2, COL_DIM);
        };

        size_t hb = in * hid;
        for (size_t i = 0; i < in; ++i)
            draw_node(in_x, node_y((int)in, (int)i), 0, ("i" + std::to_string(i)).c_str());
        for (size_t h = 0; h < hid; ++h)
            draw_node(hid_x, node_y((int)hid, (int)h), genes[hb + h], ("h" + std::to_string(h)).c_str());
        size_t ob = in * hid + hid + hid * out;
        for (size_t o = 0; o < out; ++o)
            draw_node(out_x, node_y((int)out, (int)o), genes[ob + o], ("o" + std::to_string(o)).c_str());

        draw_text("IN", in_x - 5, my2 - 15, COL_TEAL);
        draw_text("HIDDEN", hid_x - 15, my2 - 15, COL_GOLD);
        draw_text("OUT", out_x - 5, my2 - 15, COL_RED);
    }

    void draw_game_state(const std::string& state, int x, int y, int w, int h) {
        if (state.empty()) return;
        int lines = 0, max_len = 0, cur_len = 0;
        for (char c : state) {
            if (c == '\n') { lines++; max_len = std::max(max_len, cur_len); cur_len = 0; }
            else cur_len++;
        }
        if (cur_len > 0) { lines++; max_len = std::max(max_len, cur_len); }
        if (!lines || !max_len) return;

        int cell_w = std::min(16, w / max_len);
        int cell_h = std::min(18, h / lines);
        int ox = x + (w - max_len * cell_w) / 2;
        int oy = y + (h - lines * cell_h) / 2;
        int cx = ox, cy = oy;
        for (char c : state) {
            if (c == '\n') { cx = ox; cy += cell_h; continue; }
            fill_rect(cx, cy, cell_w - 1, cell_h - 1, get_color(c));
            cx += cell_w;
        }
    }

    void draw_game_panel(const std::string& state, int x, int y, int w, int h) {
        fill_rect(x, y, w, h, COL_BG);
        draw_rect_outline(x, y, w, h, COL_BORDER);
        draw_text("Game State", x + 10, y + 6, COL_GOLD);
        draw_game_state(state, x + 8, y + 8 + ch() + 4, w - 16, h - 16 - ch());
    }

    void draw_benchmark_bars(const std::vector<std::pair<std::string, ga::GAResult>>& results) {
        draw_text("Benchmark Results", 30, 15, COL_TEXT);
        if (results.empty()) return;
        float max_f = -1e30f;
        for (const auto& [n, r] : results) max_f = std::max(max_f, r.best_fitness);
        if (max_f <= 0) max_f = 1.0f;

        int bx = 40, by = 15 + ch() + 15, bw = width_ - 80, bh = height_ - by - 20;
        fill_rect(bx, by, bw, bh, COL_PANEL);

        int n2 = static_cast<int>(results.size());
        int bar_w = std::min(120, (bw - (n2+1) * 15) / n2);
        int gap = (bw - n2 * bar_w) / (n2 + 1);
        SDL_Color pal[] = {{107,191,160,255},{123,159,212,255},{212,184,122,255},{212,122,122,255},
                           {170,140,210,255},{100,190,190,255},{210,160,100,255},{140,200,140,255}};

        for (int i = 0; i < n2; ++i) {
            const auto& [name, result] = results[i];
            float norm = std::max(0.0f, result.best_fitness) / max_f;
            int bh2 = std::max(4, (int)(norm * (bh - 80)));
            int xx = bx + gap + i * (bar_w + gap), yy = by + bh - 40 - bh2;
            fill_rect(xx, yy, bar_w, bh2, pal[i % 8]);
            draw_rect_outline(xx, yy, bar_w, bh2, COL_BORDER);
            draw_text(fmt(result.best_fitness), xx, yy - 8 - ch(), COL_TEXT);
            draw_text(name, xx + bar_w / 2 - tw(name) / 2, by + bh - 30, COL_DIM);
        }
    }

    void draw_replay_curve(const ReplayData& replay, int cf, int x, int y, int w, int h) {
        if (replay.frames.size() < 2) return;
        float fmin = 1e30f, fmax = -1e30f;
        for (const auto& f : replay.frames) { fmin = std::min(fmin, f.fitness); fmax = std::max(fmax, f.fitness); }
        float fr = std::max(fmax - fmin, 1.0f);
        int n2 = static_cast<int>(replay.frames.size());
        for (int i = 1; i < n2; ++i) {
            int x0 = x + (i-1) * w / (n2 - 1), x1 = x + i * w / (n2 - 1);
            int y0 = y + h - (int)((replay.frames[i-1].fitness - fmin) / fr * h);
            int y1 = y + h - (int)((replay.frames[i].fitness - fmin) / fr * h);
            SDL_SetRenderDrawColor(renderer_, i <= cf ? COL_ACCENT.r : 60,
                                   i <= cf ? COL_ACCENT.g : 60, i <= cf ? COL_ACCENT.b : 75, 255);
            SDL_RenderDrawLine(renderer_, x0, y0, x1, y1);
        }
        if (cf >= 0 && cf < n2) {
            int px2 = x + cf * w / (n2 - 1);
            int py2 = y + h - (int)((replay.frames[cf].fitness - fmin) / fr * h);
            fill_rect(px2 - 3, py2 - 3, 6, 6, COL_TEXT);
        }
    }

    SDL_Color get_color(char c) const {
        switch (c) {
            case '#': return {107, 191, 160, 255};
            case ' ': return {27, 30, 46, 255};
            case '.': return {55, 58, 75, 255};
            case 'O': case 'o': return {212, 184, 122, 255};
            case 'X': case 'x': return {212, 122, 122, 255};
            case 'P': case 'p': return {123, 159, 212, 255};
            case 'S': case 's': case '@': case 'V': case 'v': return {107, 191, 160, 255};
            case 'G': case 'g': return {180, 100, 100, 255};
            case '|': case 'A': case 'a': case '=': return {138, 141, 158, 255};
            case 'K': case 'k': return {212, 184, 122, 255};
            case 'Q': case 'q': return {170, 140, 210, 255};
            case 'R': case 'r': return {210, 160, 100, 255};
            case 'B': case 'b': return {123, 159, 212, 255};
            case 'N': case 'n': return {140, 200, 140, 255};
            case 'w': return {100, 190, 190, 255};
            case '*': return {212, 122, 122, 255};
            case 'W': return {200, 200, 210, 255};
            default: return {80, 84, 100, 255};
        }
    }

    static std::string fmt(float v, int p = 2) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(p) << v;
        return ss.str();
    }
};

} // namespace viz

#else

namespace viz {

class GUI {
public:
    enum class Action { None, PauseResume, Stop, Watch, Back, Menu };

    struct Slider { const char* label; float* value; float min_val, max_val, step; };
    bool init() { return false; }
    void cleanup() {}
    bool should_close() { return true; }
    Action consume_action() { return Action::None; }
    bool render_config(int&, const std::vector<GameConfig>&, std::vector<Slider>&, bool&) { return false; }
    void render_sim(const ga::GAResult&, const std::string&, int, int, bool, const float* = nullptr,
                    size_t = 0, size_t = 0, size_t = 0, const std::string& = "",
                    const games::GameInterface* = nullptr) {}
    void render_watch(const std::string&, const std::string&, int, int, float, float,
                      const games::GameInterface* = nullptr) {}
    void render_benchmark(const std::vector<std::pair<std::string, ga::GAResult>>&) {}
    void render_replay(const ReplayData&, int, float) {}
    void render_dashboard(const std::vector<std::pair<std::string, ga::GAResult>>&) {}
};

} // namespace viz

#endif
