#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {

    // Load face cascade
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load face cascade!\n";
        return -1;
    }

    // Load eye cascade
    cv::CascadeClassifier eyeCascade;
    eyeCascade.load("haarcascades/haarcascade_eye.xml");

    // Open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam!\n";
        return -1;
    }

    std::cout << "Face Detection running! Press Q to quit.\n";

    cv::Mat frame, grey;

    while (true) {
        // Read frame from webcam
        cap >> frame;
        if (frame.empty()) break;

        // Convert to greyscale
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(grey, grey);

        // Detect faces
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(
            grey, faces,
            1.1, 5, 0,
            cv::Size(80, 80)
        );

        // Draw rectangle around each face
        for (size_t i = 0; i < faces.size(); i++) {
            cv::rectangle(frame, faces[i],
                          cv::Scalar(0, 255, 0), 2);

            // Label
            std::string label = "Face " + std::to_string(i + 1);
            cv::putText(frame, label,
                        cv::Point(faces[i].x, faces[i].y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2);

            // Detect eyes inside face
            cv::Mat faceGrey = grey(faces[i]);
            cv::Mat faceColor = frame(faces[i]);

            std::vector<cv::Rect> eyes;
            eyeCascade.detectMultiScale(
                faceGrey, eyes,
                1.1, 10, 0,
                cv::Size(20, 20)
            );

            for (const auto& eye : eyes) {
                cv::Point center(eye.x + eye.width / 2,
                                 eye.y + eye.height / 2);
                cv::circle(faceColor, center,
                           eye.width / 2,
                           cv::Scalar(255, 0, 0), 2);
            }
        }

        // Show face count
        std::string countText = "Faces: " + std::to_string(faces.size());
        cv::putText(frame, countText,
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);

        // Show help text
        cv::putText(frame, "Press Q to quit",
                    cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(180, 180, 180), 1);

        // Show the frame
        cv::imshow("Face Detection - PBL", frame);

        // Press Q to quit
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}