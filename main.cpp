#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

using namespace std;
using namespace cv::xfeatures2d;
using namespace cv;

class Matcher
{
public:
    struct Object {
        cv::Mat image;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        Matcher* matcher;
        Object(const cv::Mat& image, Matcher* matcher) :
            image(image), matcher(matcher) {
            matcher->detector->detect(image, keypoints);
            matcher->extractor->compute(image, keypoints, descriptors);
        }

        Object(const string &imgPath, Matcher* matcher) :
        matcher(matcher) {
            image = cv::imread(imgPath);
            matcher->detector->detect(image, keypoints);
            matcher->extractor->compute(image, keypoints, descriptors);
        }

        void detectAndCompute() {
            keypoints.clear();
            matcher->detector->detect(image, keypoints);
            matcher->extractor->compute(image, keypoints, descriptors);
        }
    };

    Matcher()
    {}

    void setDetector(cv::Ptr<cv::FeatureDetector> detect)
    {
        detector = detect;
    }

    void setExtractor(cv::Ptr<cv::DescriptorExtractor> extract)
    {
        extractor = extract;
    }

    void setMatcher(cv::Ptr<cv::DescriptorMatcher> match)
    {
        matcher = match;
    }

    std::vector<cv::Rect> detectObjects(Object &object, Object &scene, int N_good_matches = 10, float ratio_boundary = 0.5, int distance_coeff = 3)
    {
        Object workObject = scene;
        std::vector<cv::Rect> objects;
        while (true)
        {
            workObject.detectAndCompute();
            std::vector<cv::DMatch> matches;
            matcher->match(object.descriptors, workObject.descriptors, matches);

            double max_dist = 0; double min_dist = 100;

            for (int i = 0; i < object.descriptors.rows; i++)
            {
                double dist = matches[i].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }

            std::vector<cv::DMatch> good_matches;

            for( int i = 0; i < object.descriptors.rows; i++ )
            {
                if( matches[i].distance < distance_coeff * min_dist )
                    good_matches.push_back(matches[i]);
            }

            std::vector<cv::Point2f> obj_points;
            std::vector<cv::Point2f> scene_points;

            for (int i = 0; i < good_matches.size(); i++)
            {
                obj_points.push_back(object.keypoints[good_matches[i].queryIdx].pt);
                scene_points.push_back(workObject.keypoints[good_matches[i].trainIdx].pt);
            }

            if (good_matches.size() < N_good_matches)
                break;

            cv::Mat H = cv::findHomography(obj_points, scene_points, CV_RANSAC);

            std::vector<cv::Point2f> obj_corners(4);
            obj_corners[0] = cv::Point(0,0);
            obj_corners[1] = cv::Point(object.image.cols, 0);
            obj_corners[2] = cv::Point(object.image.cols, object.image.rows);
            obj_corners[3] = cv::Point(0, object.image.rows);

            std::vector<cv::Point2f> scene_corners(4);
            cv::perspectiveTransform(obj_corners, scene_corners, H);

            std::vector<cv::Point> points(4);
            points[0] = scene_corners[0] ;
            points[1] = scene_corners[1] ;
            points[2] = scene_corners[2] ;
            points[3] = scene_corners[3] ;

            // Ратио сторон прямоугольника равно ратио сторон логотипа (с каким-то допуском)
            // если нет, то мы детектировали что-то не то, заканчиваем

            cv::Rect rect = cv::boundingRect(points);

            float logo_ratio = (float)object.image.cols / (float)object.image.rows;
            float rect_ratio = (float)rect.width / (float)rect.height;

            if (abs(logo_ratio - rect_ratio) > ratio_boundary)
                break;

            cv::rectangle(workObject.image, rect.tl(), rect.br(), cv::Scalar(255,0,0), CV_FILLED,4);

            objects.push_back(rect);
        }

        return objects;
    }

    void drawObjects(cv::Mat& image, const std::vector<cv::Rect>& objects)
    {
        for (int i = 0; i < objects.size(); i++)
        {
            cv::rectangle(image, objects.at(i).tl(), objects.at(i).br(), cv::Scalar(255, 0, 0), 4);
        }

        cv::imshow("Result", image);
    }

private:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
};


int main(int argc, char **argv)
{    
    const char* logo_path = argv[1];
    const char* scene_path = argv[2];
    Matcher* myMatcher = new Matcher;
    cv::Ptr<cv::xfeatures2d::SIFT> sift_detector = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::xfeatures2d::SIFT> sift_extractor = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::BFMatcher> bf_matcher = new cv::BFMatcher(cv::NORM_L2,false);

    myMatcher->setDetector(sift_detector);
    myMatcher->setExtractor(sift_extractor);
    myMatcher->setMatcher(bf_matcher);

    std::cout << "Created detectors";


    Matcher::Object object(logo_path, myMatcher);
    Matcher::Object scene(scene_path, myMatcher);

    cv::Mat draw = scene.image.clone();

    std::vector<cv::Rect> objects;
    objects = myMatcher->detectObjects(object, scene);
    myMatcher->drawObjects(draw, objects);

    cv::waitKey(0);

    return 0;
}

