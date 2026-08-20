#pragma once

#include "game_interface.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

namespace games {

class TrackMania : public GameInterface {
    static constexpr int MAP_W = 50;
    static constexpr int MAP_H = 50;
    static constexpr float TILE = 8.0f;
    static constexpr float WORLD_W = MAP_W * TILE;
    static constexpr float WORLD_H = MAP_H * TILE;
    static constexpr int NUM_SENSORS = 8;
    static constexpr float SENSOR_RANGE = 45.0f;
    static constexpr float ACCEL = 0.25f;
    static constexpr float BRAKE = 0.85f;
    static constexpr float MAX_SPEED = 4.5f;
    static constexpr float FRICTION = 0.985f;
    static constexpr float STEER_RATE = 0.06f;
    static constexpr float DRIFT_FRICTION = 0.96f;
    static constexpr float DRIFT_STEER_MULT = 1.6f;
    static constexpr int MAX_LAPS = 3;
    static constexpr int NUM_CP = 12;

    enum Tile : uint8_t { GRASS, ROAD, BOOST, WALL, FINISH };

    struct Vec2 {
        float x, y;
        Vec2 operator+(const Vec2& o) const { return {x+o.x, y+o.y}; }
        Vec2 operator-(const Vec2& o) const { return {x-o.x, y-o.y}; }
        Vec2 operator*(float s) const { return {x*s, y*s}; }
        float dot(const Vec2& o) const { return x*o.x + y*o.y; }
        float len() const { return std::sqrt(x*x+y*y); }
        Vec2 norm() const { float l=len(); return l>0?Vec2{x/l,y/l}:Vec2{0,0}; }
    };

    uint8_t map_[MAP_H][MAP_W];
    Vec2 car_pos_, car_vel_;
    float car_angle_;
    float vel_angle_;
    float boost_timer_;
    int steps_;
    int laps_;
    int last_cp_;
    int current_cp_;
    float total_distance_;
    float max_speed_;
    bool done_;
    int num_checkpoints_;
    Vec2 cp_centers_[NUM_CP];

    void build_track() {
        std::memset(map_, GRASS, sizeof(map_));

        struct Seg { int x1,y1,x2,y2,w; bool boost; };
        Seg segs[] = {
            {5,10, 44,10, 3, false},
            {44,10, 44,15, 3, false},
            {44,15, 38,20, 3, false},
            {38,20, 30,20, 3, true},
            {30,20, 24,20, 3, true},
            {24,20, 18,25, 3, false},
            {18,25, 18,32, 3, false},
            {18,32, 22,37, 3, false},
            {22,37, 30,38, 3, false},
            {30,38, 38,35, 3, false},
            {38,35, 42,38, 3, false},
            {42,38, 44,40, 3, false},
            {44,40, 44,42, 3, false},
            {44,42, 5,42, 3, false},
            {5,42, 5,38, 3, false},
            {5,38, 10,33, 3, false},
            {10,33, 10,25, 3, false},
            {10,25, 5,20, 3, false},
            {5,20, 5,10, 3, false},
        };

        for (auto& s : segs) {
            float dx = static_cast<float>(s.x2 - s.x1);
            float dy = static_cast<float>(s.y2 - s.y1);
            float len = std::sqrt(dx*dx+dy*dy);
            if (len < 0.001f) continue;
            float nx = -dy/len;
            float ny = dx/len;
            int steps_i = static_cast<int>(len) + 1;
            for (int i = 0; i <= steps_i; ++i) {
                float t = (steps_i > 0) ? static_cast<float>(i)/steps_i : 0.0f;
                float cx = s.x1 + dx*t;
                float cy = s.y1 + dy*t;
                for (int dw = -s.w; dw <= s.w; ++dw) {
                    int tx = static_cast<int>(std::round(cx + nx*dw));
                    int ty = static_cast<int>(std::round(cy + ny*dw));
                    if (tx >= 0 && tx < MAP_W && ty >= 0 && ty < MAP_H) {
                        map_[ty][tx] = s.boost ? BOOST : ROAD;
                    }
                }
            }
        }

        for (int y = 0; y < MAP_H; ++y)
            for (int x = 0; x < MAP_W; ++x)
                if (map_[y][x] == GRASS) {
                    bool adj = false;
                    for (int dy = -1; dy <= 1 && !adj; ++dy)
                        for (int dx = -1; dx <= 1 && !adj; ++dx) {
                            int nx2 = x+dx, ny2 = y+dy;
                            if (nx2>=0 && nx2<MAP_W && ny2>=0 && ny2<MAP_H)
                                if (map_[ny2][nx2] == ROAD || map_[ny2][nx2] == BOOST)
                                    adj = true;
                        }
                }

        for (int x = 0; x < MAP_W; ++x) {
            if (map_[0][x] == GRASS) map_[0][x] = WALL;
            if (map_[MAP_H-1][x] == GRASS) map_[MAP_H-1][x] = WALL;
        }
        for (int y = 0; y < MAP_H; ++y) {
            if (map_[y][0] == GRASS) map_[y][0] = WALL;
            if (map_[y][MAP_W-1] == GRASS) map_[y][MAP_W-1] = WALL;
        }

        int fin_y = 10;
        for (int fx = 3; fx <= 7; ++fx)
            if (fin_y >= 0 && fin_y < MAP_H && fx >= 0 && fx < MAP_W)
                map_[fin_y][fx] = FINISH;

        num_checkpoints_ = NUM_CP;
        float cp_frac_x[] = {24,38,44,44,38,30,18,18,22,38,44,5};
        float cp_frac_y[] = {10,20,15,42,38,20,25,42,37,35,40,10};
        for (int i = 0; i < NUM_CP; ++i) {
            cp_centers_[i] = {cp_frac_x[i] * TILE, cp_frac_y[i] * TILE};
        }
    }

    Tile tile_at(float px, float py) const {
        int tx = static_cast<int>(px / TILE);
        int ty = static_cast<int>(py / TILE);
        if (tx < 0 || tx >= MAP_W || ty < 0 || ty >= MAP_H) return GRASS;
        return static_cast<Tile>(map_[ty][tx]);
    }

    float cast_sensor(float ox, float oy, float angle) const {
        float dx = std::cos(angle);
        float dy = std::sin(angle);
        for (float d = 1.0f; d <= SENSOR_RANGE; d += 1.5f) {
            float px = ox + dx*d;
            float py = oy + dy*d;
            Tile t = tile_at(px, py);
            if (t == WALL || t == GRASS) return d / SENSOR_RANGE;
        }
        return 1.0f;
    }

    int get_closest_cp() const {
        float best = 1e30f;
        int best_i = 0;
        for (int i = 0; i < NUM_CP; ++i) {
            float dx = car_pos_.x - cp_centers_[i].x;
            float dy = car_pos_.y - cp_centers_[i].y;
            float d = dx*dx+dy*dy;
            if (d < best) { best = d; best_i = i; }
        }
        return best_i;
    }

public:
    TrackMania() { reset(); }

    std::string name() const override { return "TrackMania"; }
    int action_space() const override { return 6; }

    void reset() override {
        build_track();
        car_pos_ = {6*TILE, 9*TILE};
        car_angle_ = 0.0f;
        vel_angle_ = 0.0f;
        car_vel_ = {0, 0};
        boost_timer_ = 0;
        steps_ = 0;
        laps_ = 0;
        last_cp_ = 0;
        current_cp_ = 0;
        total_distance_ = 0.0f;
        max_speed_ = 0.0f;
        done_ = false;
    }

    void step(int action) override {
        if (done_) return;
        steps_++;

        float steer = 0;
        bool accel = false;
        bool brake = false;
        switch (action) {
            case 0: accel = true; break;
            case 1: steer = -1.0f; accel = true; break;
            case 2: steer = 1.0f; accel = true; break;
            case 3: brake = true; break;
            case 4: steer = -1.0f; brake = true; break;
            case 5: steer = 1.0f; brake = true; break;
        }

        Tile tile = tile_at(car_pos_.x, car_pos_.y);

        float tile_speed = 1.0f;
        if (tile == GRASS) tile_speed = 0.4f;
        else if (tile == BOOST) { tile_speed = 1.6f; boost_timer_ = 20.0f; }
        if (boost_timer_ > 0) { tile_speed = std::max(tile_speed, 1.4f); boost_timer_ -= 1.0f; }

        bool drifting = brake && std::abs(steer) > 0.0f;
        float steer_mult = drifting ? DRIFT_STEER_MULT : 1.0f;
        car_angle_ += steer * STEER_RATE * steer_mult;

        Vec2 forward = {std::cos(car_angle_), std::sin(car_angle_)};
        if (accel) {
            car_vel_ = car_vel_ + forward * ACCEL * tile_speed;
        }
        if (brake) {
            car_vel_ = car_vel_ * BRAKE;
        }

        float cur_speed = car_vel_.len();
        float eff_max = MAX_SPEED * tile_speed;
        if (cur_speed > eff_max) car_vel_ = car_vel_.norm() * eff_max;

        float fric = drifting ? DRIFT_FRICTION : FRICTION;
        car_vel_ = car_vel_ * fric;

        car_pos_ = car_pos_ + car_vel_;
        if (car_vel_.len() > 0.01f) vel_angle_ = std::atan2(car_vel_.y, car_vel_.x);
        total_distance_ += car_vel_.len();
        max_speed_ = std::max(max_speed_, car_vel_.len());

        Tile new_tile = tile_at(car_pos_.x, car_pos_.y);
        if (new_tile == WALL) {
            car_pos_ = car_pos_ - car_vel_;
            car_vel_ = car_vel_ * 0.2f;
            return;
        }

        if (car_pos_.x < 0 || car_pos_.x >= WORLD_W || car_pos_.y < 0 || car_pos_.y >= WORLD_H) {
            done_ = true;
            return;
        }

        if ((new_tile == GRASS) && cur_speed > 0.5f) {
            car_vel_ = car_vel_ * 0.7f;
        }

        int new_cp = get_closest_cp();
        if (new_cp == (last_cp_ + 1) % NUM_CP) {
            current_cp_++;
            last_cp_ = new_cp;
            if (new_cp == 0 && current_cp_ >= NUM_CP) {
                laps_++;
                current_cp_ = 0;
                if (laps_ >= MAX_LAPS) { done_ = true; return; }
            }
        }

        if (steps_ >= 1200) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return laps_ * 10000.0f + current_cp_ * 200.0f
             + total_distance_ * 0.1f + max_speed_ * 15.0f;
    }

    int action_count() const override { return steps_; }

    void get_state(float* state) const override {
        for (int i = 0; i < NUM_SENSORS; ++i) {
            float sa = car_angle_ + (i - NUM_SENSORS/2) * (2.0f * 3.14159265f / NUM_SENSORS);
            state[i] = cast_sensor(car_pos_.x, car_pos_.y, sa);
        }
        int off = NUM_SENSORS;
        float spd = car_vel_.len();
        state[off]     = spd / MAX_SPEED;
        state[off + 1] = car_vel_.len() > 0.01f ? std::fmod(car_angle_ - vel_angle_ + 3.14159265f*3, 3.14159265f*2) / (3.14159265f*2) : 0.0f;
        if (state[off+1] < 0) state[off+1] += 1.0f;
        state[off + 2] = boost_timer_ > 0 ? 1.0f : 0.0f;

        int next_cp = (last_cp_ + 1) % NUM_CP;
        Vec2 to_next = cp_centers_[next_cp] - car_pos_;
        float angle_to_cp = std::atan2(to_next.y, to_next.x) - car_angle_;
        angle_to_cp = std::fmod(angle_to_cp + 3.14159265f*3, 3.14159265f*2) - 3.14159265f;
        state[off + 3] = angle_to_cp / 3.14159265f;

        state[off + 4] = static_cast<float>(current_cp_ + laps_ * NUM_CP) /
                          static_cast<float>(MAX_LAPS * NUM_CP);
    }

    std::string render_state() const override {
        constexpr int W = 50, H = 50;
        std::string grid(W * H, ' ');
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                char c = ' ';
                switch (map_[y][x]) {
                    case ROAD:    c = '.'; break;
                    case BOOST:   c = '^'; break;
                    case WALL:    c = '#'; break;
                    case FINISH:  c = '='; break;
                    case GRASS:   c = ' '; break;
                }
                grid[y * W + x] = c;
            }

        for (int i = 0; i < NUM_CP; ++i) {
            int gx = static_cast<int>(cp_centers_[i].x / TILE);
            int gy = static_cast<int>(cp_centers_[i].y / TILE);
            if (gx >= 0 && gx < W && gy >= 0 && gy < H)
                grid[gy * W + gx] = (i == (last_cp_+1)%NUM_CP) ? 'G' : 'g';
        }

        int cx = static_cast<int>(car_pos_.x / TILE);
        int cy = static_cast<int>(car_pos_.y / TILE);
        if (cx >= 0 && cx < W && cy >= 0 && cy < H)
            grid[cy * W + cx] = '@';

        std::string result;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) result += grid[y * W + x];
            result += '\n';
        }
        result += "Lap:" + std::to_string(laps_) + "/" + std::to_string(MAX_LAPS)
                + " CP:" + std::to_string(current_cp_) + "/" + std::to_string(NUM_CP)
                + " Spd:" + std::to_string(static_cast<int>(car_vel_.len() * 10));
        if (boost_timer_ > 0) result += " TURBO!";
        return result;
    }

    bool has_sdl_render() const override { return true; }

    void render_sdl(SDL_Renderer* renderer, int x, int y, int w, int h) const override {
        float scale_x = w / WORLD_W;
        float scale_y = h / WORLD_H;
        float scale = std::min(scale_x, scale_y);
        float ox = x + (w - WORLD_W * scale) * 0.5f;
        float oy = y + (h - WORLD_H * scale) * 0.5f;
        auto to_screen = [&](float px, float py) -> std::pair<int,int> {
            return {static_cast<int>(ox + px * scale), static_cast<int>(oy + py * scale)};
        };

        for (int ty = 0; ty < MAP_H; ++ty) {
            for (int tx = 0; tx < MAP_W; ++tx) {
                Tile t = static_cast<Tile>(map_[ty][tx]);
                if (t == GRASS) continue;
                auto [sx, sy] = to_screen(tx * TILE, ty * TILE);
                int s = static_cast<int>(TILE * scale) + 1;
                switch (t) {
                    case ROAD:    SDL_SetRenderDrawColor(renderer, 70, 75, 95, 255); break;
                    case BOOST:   SDL_SetRenderDrawColor(renderer, 50, 100, 180, 255); break;
                    case WALL:    SDL_SetRenderDrawColor(renderer, 55, 58, 75, 255); break;
                    case FINISH:  SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255); break;
                    default: continue;
                }
                SDL_Rect r = {sx, sy, s, s};
                SDL_RenderFillRect(renderer, &r);
            }
        }

        for (int i = 0; i < NUM_CP; ++i) {
            auto [sx, sy] = to_screen(cp_centers_[i].x, cp_centers_[i].y);
            if (i == (last_cp_+1)%NUM_CP) {
                SDL_SetRenderDrawColor(renderer, 107, 191, 160, 255);
                SDL_Rect r = {sx-2, sy-2, 5, 5};
                SDL_RenderFillRect(renderer, &r);
            } else {
                SDL_SetRenderDrawColor(renderer, 80, 84, 100, 180);
                SDL_RenderDrawPoint(renderer, sx, sy);
            }
        }

        for (int i = 0; i < NUM_SENSORS; ++i) {
            float sa = car_angle_ + (i - NUM_SENSORS/2) * (2.0f * 3.14159265f / NUM_SENSORS);
            float d = cast_sensor(car_pos_.x, car_pos_.y, sa) * SENSOR_RANGE;
            float ex = car_pos_.x + std::cos(sa) * d;
            float ey = car_pos_.y + std::sin(sa) * d;
            auto [sx1, sy1] = to_screen(car_pos_.x, car_pos_.y);
            auto [sx2, sy2] = to_screen(ex, ey);
            SDL_SetRenderDrawColor(renderer, 123, 159, 212, 100);
            SDL_RenderDrawLine(renderer, sx1, sy1, sx2, sy2);
        }

        float cs = 5.0f * scale;
        auto [cx2, cy2] = to_screen(car_pos_.x, car_pos_.y);
        float cdx = std::cos(car_angle_) * cs;
        float cdy = std::sin(car_angle_) * cs;
        float cpx = -cdy * 0.6f;
        float cpy = cdx * 0.6f;

        SDL_SetRenderDrawColor(renderer, 212, 184, 122, 255);
        SDL_RenderDrawLine(renderer, static_cast<Sint16>(cx2+cdx), static_cast<Sint16>(cy2+cdy),
                           static_cast<Sint16>(cx2-cdx+cpx), static_cast<Sint16>(cy2-cdy+cpy));
        SDL_RenderDrawLine(renderer, static_cast<Sint16>(cx2-cdx+cpx), static_cast<Sint16>(cy2-cdy+cpy),
                           static_cast<Sint16>(cx2-cdx-cpx), static_cast<Sint16>(cy2-cdy-cpy));
        SDL_RenderDrawLine(renderer, static_cast<Sint16>(cx2-cdx-cpx), static_cast<Sint16>(cy2-cdy-cpy),
                           static_cast<Sint16>(cx2+cdx), static_cast<Sint16>(cy2+cdy));

        if (boost_timer_ > 0) {
            SDL_SetRenderDrawColor(renderer, 50, 100, 180, 180);
            SDL_Rect br = {cx2-6, cy2-6, 12, 12};
            SDL_RenderDrawRect(renderer, &br);
        }
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME(TrackMania, 13, {20}, 1200)
