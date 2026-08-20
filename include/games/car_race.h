#pragma once

#include "game_interface.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>

#ifdef HAS_SDL2
#include <SDL.h>
#endif

namespace games {

class CarRace : public GameInterface {
    static constexpr int NUM_SENSORS = 7;
    static constexpr float SENSOR_RANGE = 40.0f;
    static constexpr float BASE_SPEED = 3.0f;
    static constexpr float TURN_RATE = 0.09f;
    static constexpr float BRAKE_FORCE = 0.6f;
    static constexpr float ACCEL_FORCE = 0.15f;
    static constexpr float MAX_SPEED = 6.0f;
    static constexpr float FRICTION = 0.98f;
    static constexpr int NUM_CP = 16;
    static constexpr int WORLD_SIZE = 400;

    struct Vec2 {
        float x, y;
        Vec2 operator+(const Vec2& o) const { return {x+o.x, y+o.y}; }
        Vec2 operator-(const Vec2& o) const { return {x-o.x, y-o.y}; }
        Vec2 operator*(float s) const { return {x*s, y*s}; }
        float dot(const Vec2& o) const { return x*o.x + y*o.y; }
        float len() const { return std::sqrt(x*x + y*y); }
        Vec2 norm() const { float l = len(); return l > 0 ? Vec2{x/l, y/l} : Vec2{0,0}; }
    };

    struct TrackSegment {
        Vec2 center;
        float half_width;
        char terrain;
    };

    std::vector<TrackSegment> track_;
    Vec2 car_pos_, car_vel_;
    float car_angle_;
    float boost_timer_;
    int steps_;
    int laps_;
    int last_cp_;
    int current_cp_;
    float total_distance_;
    float max_speed_reached_;
    bool done_;
    int num_checkpoints_;
    mutable int cached_seg_;

    float sample_track(float px, float py, Vec2& out_normal, Vec2& out_center, float& out_hw, char& out_terrain) const {
        int N = static_cast<int>(track_.size());
        int best_seg = cached_seg_;
        float best_d = 1e30f;
        for (int i = -3; i <= 3; ++i) {
            int idx = (best_seg + i + N) % N;
            float dx = px - track_[idx].center.x;
            float dy = py - track_[idx].center.y;
            float d = dx*dx + dy*dy;
            if (d < best_d) { best_d = d; best_seg = idx; }
        }
        cached_seg_ = best_seg;
        int next = (best_seg + 1) % N;
        Vec2 seg_dir = (track_[next].center - track_[best_seg].center).norm();
        Vec2 to_point = {px - track_[best_seg].center.x, py - track_[best_seg].center.y};
        float proj = to_point.dot(seg_dir);
        Vec2 closest = track_[best_seg].center + seg_dir * std::max(0.0f, proj);
        Vec2 delta = {px - closest.x, py - closest.y};
        float dist = delta.len();
        out_normal = delta.norm();
        out_center = closest;
        out_hw = track_[best_seg].half_width;
        out_terrain = track_[best_seg].terrain;
        return dist;
    }

    int get_closest_cp(float px, float py) const {
        float best = 1e30f;
        int best_i = 0;
        for (int i = 0; i < num_checkpoints_; ++i) {
            int si = (i * static_cast<int>(track_.size())) / num_checkpoints_;
            float dx = px - track_[si].center.x;
            float dy = py - track_[si].center.y;
            float d = dx*dx + dy*dy;
            if (d < best) { best = d; best_i = i; }
        }
        return best_i;
    }

    void build_track() {
        track_.clear();
        struct WP { float x, y, w; };
        std::vector<WP> wps = {
            {200, 30, 30}, {260, 35, 26}, {310, 60, 22}, {340, 110, 20},
            {350, 170, 18}, {330, 220, 22}, {290, 260, 20}, {340, 300, 18},
            {370, 340, 20}, {350, 370, 22}, {300, 360, 18}, {260, 330, 16},
            {220, 350, 18}, {180, 380, 20}, {130, 360, 22}, {90, 320, 18},
            {60, 270, 20}, {40, 220, 22}, {60, 170, 18}, {100, 140, 16},
            {70, 100, 18}, {40, 60, 22}, {80, 40, 26}, {130, 30, 24},
            {170, 28, 28}
        };

        for (int i = 0; i < static_cast<int>(wps.size()) - 1; ++i) {
            int steps_interp = 4;
            for (int s = 0; s < steps_interp; ++s) {
                float t = static_cast<float>(s) / steps_interp;
                float x = wps[i].x + (wps[i+1].x - wps[i].x) * t;
                float y = wps[i].y + (wps[i+1].y - wps[i].y) * t;
                float w = wps[i].w + (wps[i+1].w - wps[i].w) * t;

                float norm_pos = static_cast<float>(track_.size()) / 380.0f;
                char terrain = 'R';
                if (norm_pos > 0.12f && norm_pos < 0.18f) terrain = 'B';
                else if (norm_pos > 0.38f && norm_pos < 0.44f) terrain = 'S';
                else if (norm_pos > 0.55f && norm_pos < 0.61f) terrain = 'B';
                else if (norm_pos > 0.75f && norm_pos < 0.81f) terrain = 'S';

                track_.push_back({{x, y}, w, terrain});
            }
        }
        num_checkpoints_ = NUM_CP;
    }

public:
    CarRace() { reset(); }

    std::string name() const override { return "CarRace"; }
    int action_space() const override { return 5; }

    void reset() override {
        build_track();
        car_pos_ = track_[0].center;
        Vec2 dir;
        if (track_.size() > 1) dir = (track_[1].center - track_[0].center).norm();
        else dir = {1, 0};
        car_angle_ = std::atan2(dir.y, dir.x);
        car_vel_ = {0, 0};
        boost_timer_ = 0;
        steps_ = 0;
        laps_ = 0;
        last_cp_ = 0;
        current_cp_ = 0;
        total_distance_ = 0.0f;
        max_speed_reached_ = 0.0f;
        cached_seg_ = 0;
        done_ = false;
    }

    void step(int action) override {
        if (done_) return;
        steps_++;

        float steer = 0;
        bool accel = false;
        bool brake = false;
        if (action == 0) steer = -1.0f;
        else if (action == 1) steer = 1.0f;
        else if (action == 2) accel = true;
        else if (action == 3) brake = true;

        car_angle_ += steer * TURN_RATE;

        if (boost_timer_ > 0) {
            boost_timer_ -= 1.0f;
        }

        Vec2 forward = {std::cos(car_angle_), std::sin(car_angle_)};
        char terrain = 'R';
        Vec2 out_n, out_c;
        float out_hw;
        sample_track(car_pos_.x, car_pos_.y, out_n, out_c, out_hw, terrain);

        float speed_mult = 1.0f;
        if (terrain == 'B') { speed_mult = 1.8f; boost_timer_ = 15.0f; }
        else if (terrain == 'S') { speed_mult = 0.5f; }

        if (boost_timer_ > 0) speed_mult = std::max(speed_mult, 1.5f);

        float effective_max = MAX_SPEED * speed_mult;

        if (accel) {
            car_vel_ = car_vel_ + forward * ACCEL_FORCE * speed_mult;
        }
        if (brake) {
            car_vel_ = car_vel_ * BRAKE_FORCE;
        }

        float cur_speed = car_vel_.len();
        if (cur_speed > effective_max) {
            car_vel_ = car_vel_.norm() * effective_max;
        }
        car_vel_ = car_vel_ * FRICTION;

        if (cur_speed < 0.01f && !accel) {
            car_vel_ = forward * BASE_SPEED * 0.3f;
        }

        car_pos_ = car_pos_ + car_vel_;
        total_distance_ += car_vel_.len();
        max_speed_reached_ = std::max(max_speed_reached_, car_vel_.len());

        float dist_to_edge = sample_track(car_pos_.x, car_pos_.y, out_n, out_c, out_hw, terrain);
        if (dist_to_edge > out_hw + 5.0f) {
            done_ = true;
            return;
        }

        int new_cp = get_closest_cp(car_pos_.x, car_pos_.y);
        if (new_cp == (last_cp_ + 1) % num_checkpoints_) {
            current_cp_++;
            last_cp_ = new_cp;
            if (new_cp == 0 && current_cp_ >= num_checkpoints_) {
                laps_++;
                current_cp_ = 0;
            }
        }

        if (steps_ >= 800) done_ = true;
    }

    bool is_done() const override { return done_; }

    float fitness() const override {
        return laps_ * 1000.0f + current_cp_ * 50.0f + total_distance_ * 0.05f + max_speed_reached_ * 10.0f;
    }

    int action_count() const override { return steps_; }

    void get_state(float* state) const override {
        for (int i = 0; i < NUM_SENSORS; ++i) {
            float sa = car_angle_ + (i - NUM_SENSORS / 2) * 0.4f;
            Vec2 out_n, out_c;
            float out_hw;
            char t;
            float tx = car_pos_.x + std::cos(sa) * SENSOR_RANGE;
            float ty = car_pos_.y + std::sin(sa) * SENSOR_RANGE;
            float ed = sample_track(tx, ty, out_n, out_c, out_hw, t);
            state[i] = (ed > out_hw) ? 0.2f : 1.0f;
        }
        int off = NUM_SENSORS;
        float cur_speed = car_vel_.len();
        state[off] = cur_speed / MAX_SPEED;
        state[off + 1] = std::fmod(car_angle_ + 3.14159265f, 2.0f * 3.14159265f) / (2.0f * 3.14159265f);
        state[off + 2] = boost_timer_ > 0 ? 1.0f : 0.0f;
        state[off + 3] = static_cast<float>(current_cp_) / num_checkpoints_;
        state[off + 4] = static_cast<float>(laps_) / 3.0f;

        Vec2 to_next = track_[std::min(static_cast<int>((last_cp_ + 1) % num_checkpoints_) *
                        static_cast<int>(track_.size()) / num_checkpoints_,
                        static_cast<int>(track_.size()) - 1)].center - car_pos_;
        state[off + 5] = std::atan2(to_next.y, to_next.x) - car_angle_;
        state[off + 5] = std::fmod(state[off + 5] + 3.14159265f, 2.0f * 3.14159265f);
        if (state[off + 5] < 0) state[off + 5] += 2.0f * 3.14159265f;
        state[off + 5] /= (2.0f * 3.14159265f);
    }

    std::string render_state() const override {
        int W = 40, H = 40;
        std::string grid(W * H, ' ');
        auto set_char = [&](int x, int y, char c) {
            if (x >= 0 && x < W && y >= 0 && y < H) grid[y * W + x] = c;
        };

        for (int i = 0; i < static_cast<int>(track_.size()); ++i) {
            int ni = (i + 1) % static_cast<int>(track_.size());
            const auto& a = track_[i];
            const auto& b = track_[ni];
            Vec2 dir = (b.center - a.center).norm();
            Vec2 perp = {-dir.y, dir.x};

            int steps_l = static_cast<int>((b.center - a.center).len() / 2.0f) + 1;
            for (int s = 0; s <= steps_l; ++s) {
                float t = (steps_l > 0) ? static_cast<float>(s) / steps_l : 0;
                Vec2 mid = a.center + (b.center - a.center) * t;
                float hw = a.half_width + (b.half_width - a.half_width) * t;

                char tc = (a.terrain == 'B') ? 'B' : (a.terrain == 'S') ? 'S' : 'R';

                for (float dw = -hw; dw <= hw; dw += 3.0f) {
                    Vec2 p = mid + perp * dw;
                    int gx = static_cast<int>(p.x / 10.0f);
                    int gy = static_cast<int>(p.y / 10.0f);
                    set_char(gx, gy, tc);
                }
            }
        }

        for (int i = 0; i < num_checkpoints_; ++i) {
            int si = (i * static_cast<int>(track_.size())) / num_checkpoints_;
            int gx = static_cast<int>(track_[si].center.x / 10.0f);
            int gy = static_cast<int>(track_[si].center.y / 10.0f);
            set_char(gx, gy, (i == last_cp_) ? 'G' : 'g');
        }

        int cgx = static_cast<int>(car_pos_.x / 10.0f);
        int cgy = static_cast<int>(car_pos_.y / 10.0f);
        set_char(cgx, cgy, '@');

        for (int i = 0; i < NUM_SENSORS; ++i) {
            float sa = car_angle_ + (i - NUM_SENSORS / 2) * 0.4f;
            float tx = car_pos_.x + std::cos(sa) * SENSOR_RANGE;
            float ty = car_pos_.y + std::sin(sa) * SENSOR_RANGE;
            Vec2 n, c; float hw; char tr;
            float ed = sample_track(tx, ty, n, c, hw, tr);
            int sx = static_cast<int>((ed > hw ? car_pos_.x + std::cos(sa) * SENSOR_RANGE : tx) / 10.0f);
            int sy = static_cast<int>((ed > hw ? car_pos_.y + std::sin(sa) * SENSOR_RANGE : ty) / 10.0f);
            set_char(sx, sy, '.');
        }

        std::string result;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) result += grid[y * W + x];
            result += '\n';
        }
        result += "Lap:" + std::to_string(laps_) + " CP:" + std::to_string(current_cp_)
                  + "/" + std::to_string(num_checkpoints_)
                  + " Spd:" + std::to_string(static_cast<int>(car_vel_.len() * 10));
        if (boost_timer_ > 0) result += " TURBO!";
        return result;
    }

    bool has_sdl_render() const override { return true; }

    void render_sdl(SDL_Renderer* renderer, int x, int y, int w, int h) const override {
        float scale_x = w / 400.0f;
        float scale_y = h / 400.0f;
        float scale = std::min(scale_x, scale_y);
        float ox = x + (w - 400.0f * scale) * 0.5f;
        float oy = y + (h - 400.0f * scale) * 0.5f;
        auto to_screen = [&](float px, float py) -> std::pair<int,int> {
            return {static_cast<int>(ox + px * scale), static_cast<int>(oy + py * scale)};
        };

        for (int i = 0; i < static_cast<int>(track_.size()); ++i) {
            int ni = (i + 1) % static_cast<int>(track_.size());
            const auto& a = track_[i];
            const auto& b = track_[ni];
            Vec2 dir = (b.center - a.center).norm();
            Vec2 perp = {-dir.y, dir.x};

            for (float dw = -a.half_width; dw <= a.half_width; dw += 4.0f) {
                Vec2 p = a.center + perp * dw;
                auto [sx, sy] = to_screen(p.x, p.y);
                if (a.terrain == 'B')
                    SDL_SetRenderDrawColor(renderer, 60, 120, 200, 255);
                else if (a.terrain == 'S')
                    SDL_SetRenderDrawColor(renderer, 140, 120, 70, 255);
                else
                    SDL_SetRenderDrawColor(renderer, 70, 75, 95, 255);
                SDL_RenderDrawPoint(renderer, sx, sy);
            }
        }

        for (int i = 0; i < static_cast<int>(track_.size()); ++i) {
            int ni = (i + 1) % static_cast<int>(track_.size());
            auto [x1, y1] = to_screen(track_[i].center.x, track_[i].center.y);
            auto [x2, y2] = to_screen(track_[ni].center.x, track_[ni].center.y);
            SDL_SetRenderDrawColor(renderer, 55, 58, 75, 255);
            SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
        }

        for (int i = 0; i < num_checkpoints_; ++i) {
            int si = (i * static_cast<int>(track_.size())) / num_checkpoints_;
            auto [sx, sy] = to_screen(track_[si].center.x, track_[si].center.y);
            if (i == last_cp_) {
                SDL_SetRenderDrawColor(renderer, 107, 191, 160, 255);
                SDL_Rect r = {sx - 3, sy - 3, 6, 6};
                SDL_RenderFillRect(renderer, &r);
            } else {
                SDL_SetRenderDrawColor(renderer, 80, 84, 100, 255);
                SDL_RenderDrawPoint(renderer, sx, sy);
            }
        }

        for (int i = 0; i < NUM_SENSORS; ++i) {
            float sa = car_angle_ + (i - NUM_SENSORS / 2) * 0.4f;
            float tx = car_pos_.x + std::cos(sa) * SENSOR_RANGE;
            float ty = car_pos_.y + std::sin(sa) * SENSOR_RANGE;
            Vec2 n, c; float hw; char tr;
            float ed = sample_track(tx, ty, n, c, hw, tr);
            float end_x = ed > hw ? car_pos_.x + std::cos(sa) * SENSOR_RANGE : tx;
            float end_y = ed > hw ? car_pos_.y + std::sin(sa) * SENSOR_RANGE : ty;
            auto [sx1, sy1] = to_screen(car_pos_.x, car_pos_.y);
            auto [sx2, sy2] = to_screen(end_x, end_y);
            SDL_SetRenderDrawColor(renderer, 123, 159, 212, 120);
            SDL_RenderDrawLine(renderer, sx1, sy1, sx2, sy2);
        }

        float cs = 6.0f * scale;
        auto [cx, cy] = to_screen(car_pos_.x, car_pos_.y);
        float cdx = std::cos(car_angle_) * cs;
        float cdy = std::sin(car_angle_) * cs;
        float cpx = -cdy * 0.6f;
        float cpy = cdx * 0.6f;

        SDL_SetRenderDrawColor(renderer, 212, 184, 122, 255);
        Sint16 vx[] = {static_cast<Sint16>(cx + cdx), static_cast<Sint16>(cx - cdx + cpx), static_cast<Sint16>(cx - cdx - cpx)};
        Sint16 vy[] = {static_cast<Sint16>(cy + cdy), static_cast<Sint16>(cy - cdy + cpy), static_cast<Sint16>(cy - cdy - cpy)};
        SDL_RenderDrawLine(renderer, vx[0], vy[0], vx[1], vy[1]);
        SDL_RenderDrawLine(renderer, vx[1], vy[1], vx[2], vy[2]);
        SDL_RenderDrawLine(renderer, vx[2], vy[2], vx[0], vy[0]);
        SDL_RenderDrawLine(renderer, vx[0], vy[0], vx[1], vy[1]);

        if (boost_timer_ > 0) {
            SDL_SetRenderDrawColor(renderer, 123, 159, 212, 180);
            SDL_Rect br = {cx - 8, cy - 8, 16, 16};
            SDL_RenderDrawRect(renderer, &br);
        }
    }
};

} // namespace games

#include "ga/game_registry.h"
REGISTER_GAME_SYM(CarRace, 13, {16}, 400, {0,6}, {1,5}, {2,4})
