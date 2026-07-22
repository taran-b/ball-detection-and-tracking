#include <edge-impulse-sdk/classifier/ei_run_classifier.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

static cv::Mat resized_frame;

// Edge Impulse expects ONE packed float per pixel (0xRRGGBB), not 3 separate
// floats. total_length == width * height, so out_ptr only has room for
// `length` floats — writing 3 per pixel overran the buffer and corrupted
// memory, which is what caused the original "trace trap" crash.
static int get_data(size_t offset, size_t length, float *out_ptr) {
    size_t pixel_ix = offset;

    for (size_t i = 0; i < length; i++) {
        size_t x = pixel_ix % resized_frame.cols;
        size_t y = pixel_ix / resized_frame.cols;

        cv::Vec3b pixel = resized_frame.at<cv::Vec3b>(y, x);

        // resized_frame is already RGB (converted below), so channel order is R,G,B
        out_ptr[i] = static_cast<float>((pixel[0] << 16) + (pixel[1] << 8) + pixel[2]);

        pixel_ix++;
    }

    return 0;
}

int main() {
    constexpr float min_confidence = 0.75f;
    constexpr int min_box_size = 12;

    // --- Navigation tuning ---
    // Horizontal dead-zone: how close to dead-center the ball must be
    // before we stop calling it "left" or "right" (fraction of frame width).
    constexpr float dead_zone_fraction = 1.0f / 6.0f;
    // If the ball's bounding-box height takes up more than this fraction of
    // the frame height, treat it as "near" -> STOP.
    constexpr float near_height_fraction = 0.45f;

    const int input_w = EI_CLASSIFIER_INPUT_WIDTH;
    const int input_h = EI_CLASSIFIER_INPUT_HEIGHT;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open webcam" << std::endl;
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::cout << "Edge Impulse webcam detection started." << std::endl;
    std::cout << "Input size: " << input_w << "x" << input_h << std::endl;
    std::cout << "Press q to quit." << std::endl;

    // --- Background capture thread ---
    // Runs cap >> frame in its own loop, independent of how long inference
    // takes on the main thread. This is the fix for dropped/stale frames:
    // previously, capture and inference ran back-to-back on one thread, so
    // while run_classifier() was busy the camera driver's buffer would
    // either drop frames or hand back a stale one on the next read. Now the
    // camera is always being read as fast as it can produce frames, and the
    // main thread just picks up whatever the newest one is when it's ready.
    cv::Mat shared_frame;
    std::mutex frame_mutex;
    std::atomic<bool> frame_ready{false};
    std::atomic<bool> running{true};

    std::thread capture_thread([&]() {
        while (running.load()) {
            cv::Mat f;
            cap >> f;
            if (f.empty()) {
                continue;
            }
            std::lock_guard<std::mutex> lock(frame_mutex);
            shared_frame = f;
            frame_ready = true;
        }
    });

    // --- Detection persistence ---
    // The classifier will occasionally miss a frame even while the ball is
    // still in view (motion blur, lighting flicker, etc). Rather than
    // immediately reporting "NO BALL" the moment one frame misses, hold on
    // to the last known position for a short window and keep steering off
    // that. Only report NO BALL once we've missed for longer than that.
    constexpr int max_missed_frames = 15; // ~0.5s at 30fps, tune to your camera's fps
    int missed_frames = 0;
    bool have_last_detection = false;
    int last_x = 0, last_y = 0, last_w = 0, last_h = 0;

    while (true) {
        cv::Mat frame;

        if (!frame_ready.load()) {
            // No new frame from the capture thread yet — don't busy-spin,
            // give it a moment and check again.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame = shared_frame.clone();
            frame_ready = false;
        }

        if (frame.empty()) {
            continue;
        }

        cv::resize(frame, resized_frame, cv::Size(input_w, input_h));
        cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

        signal_t signal;
        signal.total_length = input_w * input_h; // number of pixels, not *3
        signal.get_data = &get_data;

        ei_impulse_result_t result = {0};

        EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
        if (err != EI_IMPULSE_OK) {
            std::cerr << "run_classifier failed: " << err << std::endl;
            continue;
        }

        int detections_drawn = 0;

        // Track the highest-confidence ball detection this frame, so we can
        // issue a single navigation command from it.
        bool ball_found = false;
        int best_x = 0, best_y = 0, best_w = 0, best_h = 0;

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
        for (size_t i = 0; i < result.bounding_boxes_count; i++) {
            const auto &bb = result.bounding_boxes[i];

            if (bb.value < min_confidence) continue;
            if (bb.width < min_box_size || bb.height < min_box_size) continue;

            int x = static_cast<int>((static_cast<float>(bb.x) / input_w) * frame.cols);
            int y = static_cast<int>((static_cast<float>(bb.y) / input_h) * frame.rows);
            int w = static_cast<int>((static_cast<float>(bb.width) / input_w) * frame.cols);
            int h = static_cast<int>((static_cast<float>(bb.height) / input_h) * frame.rows);

            x = std::max(0, std::min(x, frame.cols - 1));
            y = std::max(0, std::min(y, frame.rows - 1));
            w = std::max(1, std::min(w, frame.cols - x));
            h = std::max(1, std::min(h, frame.rows - y));

            cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

            std::string label = std::string(bb.label) + " " + cv::format("%.2f", bb.value);
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            int text_x = x;
            int text_y = std::max(20, y - 8);

            cv::rectangle(
                frame,
                cv::Point(text_x, text_y - text_size.height - 6),
                cv::Point(text_x + text_size.width + 6, text_y + baseline - 2),
                cv::Scalar(0, 255, 0),
                cv::FILLED
            );

            cv::putText(
                frame,
                label,
                cv::Point(text_x + 3, text_y - 3),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 0, 0),
                1
            );

            detections_drawn++;

            // No ranking — if it passes the confidence/size filters above,
            // it's a detection. Take it.
            if (!ball_found) {
                best_x = x;
                best_y = y;
                best_w = w;
                best_h = h;
                ball_found = true;
            }
        }
#else
        size_t best_index = 0;
        float best_value = 0.0f;

        for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            if (result.classification[i].value > best_value) {
                best_value = result.classification[i].value;
                best_index = i;
            }
        }

        if (best_value >= min_confidence) {
            std::string label = std::string(result.classification[best_index].label) +
                                " " + cv::format("%.2f", best_value);

            cv::putText(
                frame,
                label,
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                2
            );
        }
#endif

        // --- Persist through brief missed detections ---
        if (ball_found) {
            last_x = best_x;
            last_y = best_y;
            last_w = best_w;
            last_h = best_h;
            have_last_detection = true;
            missed_frames = 0;
        } else if (have_last_detection && missed_frames < max_missed_frames) {
            // No detection this frame, but we recently had one — keep
            // steering off the last known position instead of dropping out.
            missed_frames++;
            best_x = last_x;
            best_y = last_y;
            best_w = last_w;
            best_h = last_h;
            ball_found = true;
        } else {
            // Missed too many frames in a row — genuinely lost the ball.
            have_last_detection = false;
        }

        // --- Decide navigation command from the best detection ---
        std::string command = "NO BALL";

        if (ball_found) {
            int box_center_x = best_x + best_w / 2;
            int frame_center_x = frame.cols / 2;
            int dead_zone = static_cast<int>(frame.cols * dead_zone_fraction);

            float height_fraction = static_cast<float>(best_h) / static_cast<float>(frame.rows);

            if (height_fraction >= near_height_fraction) {
                // Ball fills enough of the frame -> we're close to it
                command = "STOP";
            } else if (box_center_x < frame_center_x - dead_zone) {
                command = "GO LEFT";
            } else if (box_center_x > frame_center_x + dead_zone) {
                command = "GO RIGHT";
            } else {
                // Roughly centered and not close yet -> drive straight
                command = "GO";
            }
        }

        // Print to console for downstream robot control code to consume
        std::cout << "COMMAND: " << command;
        if (missed_frames > 0 && command != "NO BALL") {
            std::cout << "  (holding, missed " << missed_frames << " frame(s))";
        }
        std::cout << std::endl;

        // Overlay the command on the video feed
        cv::Scalar command_color =
            (command == "STOP") ? cv::Scalar(0, 0, 255) :
            (command == "NO BALL") ? cv::Scalar(128, 128, 128) :
                                      cv::Scalar(0, 255, 0);

        cv::putText(
            frame,
            command,
            cv::Point(10, frame.rows - 20),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            command_color,
            2
        );

        std::string timing_text =
            "DSP: " + std::to_string(result.timing.dsp) +
            " ms  INF: " + std::to_string(result.timing.classification) + " ms";

        cv::putText(
            frame,
            timing_text,
            cv::Point(10, 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 255),
            1
        );

        if (detections_drawn > 0) {
            std::cout << "Detections:" << std::endl;
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
            for (size_t i = 0; i < result.bounding_boxes_count; i++) {
                const auto &bb = result.bounding_boxes[i];

                if (bb.value < min_confidence) continue;
                if (bb.width < min_box_size || bb.height < min_box_size) continue;

                std::cout << "  " << bb.label
                          << " (" << bb.value << ")"
                          << " [x=" << bb.x
                          << ", y=" << bb.y
                          << ", w=" << bb.width
                          << ", h=" << bb.height
                          << "]" << std::endl;
            }
#endif
        }

        cv::imshow("Edge Impulse Webcam Detection", frame);

        char c = (char)cv::waitKey(1);
        if (c == 'q' || c == 'Q') {
            break;
        }
    }

    running = false;
    capture_thread.join();

    cap.release();
    cv::destroyAllWindows();
    return 0;
}