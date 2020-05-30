#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>


using namespace dlib;
using namespace std;

int main() {
    const int noOfLandmarks = 68;

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    cv::VideoCapture cap(0);

    // Check if camera opened successfully
    if(!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("./shape_predictor_68_face_landmarks.dat") >> sp;


    cv::namedWindow("Window");

    int scale = 2; // light ver: 3
    while(1) {

        cv::Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        cv::flip(frame, frame, 1);
        cv::resize(frame, frame, cv::Size(frame.cols / scale, frame.rows / scale));


        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        cv_image<bgr_pixel> img(frame);


        //pyramid_up(img);
        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        std::vector<full_object_detection> shapes;


        if(dets.size()) {

            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(img, dets[j]);
                //cout << "number of detected parts " << shape.num_parts() << endl;
                shapes.push_back(shape);



//			for (int j = 0; j < shape.num_parts(); j++) {
                for (int i = 0; i < noOfLandmarks; i++) {
                    cv::circle(frame, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(0, 0, 255));
                }
            }
        }
        else {
            cv::putText(frame, "No face detected", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }



        // Display the resulting frame
        cv::imshow( "Window", frame);

        // Press  ESC on keyboard to exit
        char c=(char)cv::waitKey(25);
        if(c==27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}
