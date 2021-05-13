#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
//#include <boost_1_66_0/boost/thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class ImageProcessor {

    private:

        cv::Mat inputImage;
        std::vector<cv::Mat> digits;

    public:

        static std::mutex *spacesVector_mutex;

        ImageProcessor(cv::Mat image) {

            inputImage = image;
            //spacesVector_mutex = new std::mutex();
        }

        void setDigit(uint whichDigit, cv::Mat& inputImage) {

            digits.at(whichDigit) = inputImage;
        }

        size_t getColumns() {

            return inputImage.cols;
        }

        size_t getRows() {

            return inputImage.rows;
        }

        // Getter and setter to access any arbitrary digit mat
        cv::Mat getDigit(uint whichDigit) {

            return digits.at(whichDigit);
        }

        // Dilate the image to remove small noise
        // Erode the image to keep readability
        void RemoveNoise() {

            cv::Mat erodedImage;
            cv::dilate(inputImage, erodedImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
            cv::erode(erodedImage, erodedImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));

            ShowImage(erodedImage);
            //cv::Mat outputObjectCentroids, outputObjectLabels, outputObjectStats;
            //findSpecs(erodedImage, outputObjectLabels, outputObjectStats, outputObjectCentroids);

            inputImage = erodedImage;
        }

        // Find small specs in image
        void FindSpecs(cv::Mat& inputImage, cv::Mat& outputObjectLabels, cv::Mat& outputObjectStats, cv::Mat& outputObjectCentroids) {

            cv::connectedComponentsWithStats(inputImage, outputObjectLabels, outputObjectStats, outputObjectCentroids, 8);
            //ShowImage(outputObjectLabels);
            //ShowImage(outputObjectStats);
            //ShowImage(outputObjectCentroids);
        }

        // Static version of the above remove noise function for use outside of 
        // class objects
        static cv::Mat RemoveNoise(cv::Mat input) {

            cv::Mat erodedImage;
            cv::dilate(input, erodedImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
            cv::erode(erodedImage, erodedImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));

            return erodedImage;
        }

        // Static method to display an image
        static void ShowImage(cv::Mat outputImage) {

            cv::imshow("Output Image", outputImage);
            cv::waitKey();
        }

        // Remove the horizontal line that's created after the copyTo method
        static void RemoveHorizontalLine(uint startIndex, cv::Mat& input) {

            for(int i = 0; i < input.rows; i++) {

                input.at<uchar>(i, startIndex) = 255;
            }
        }

        // function for the scanning of the image by each thread
        void ThreadFunction(size_t beginIndex, size_t endIndex, 
            std::vector<uint16_t> &spacesVector, int rows, std::mutex *vectorMutex) const {

            bool isBlack = false;
            for(int i = beginIndex; i < endIndex; i++) {

                uint16_t count = 0;
                for(int k = 0; k < rows; k++) {

                    // Mutex not necessary since it is only reading
                    if(inputImage.at<uchar>(k, i) == 255) {

                        count++;
                    }
                    else {

                        if(!isBlack) isBlack = true;
                    }
                }

                if(count == rows && isBlack) {

                    //boost::mutex::scoped_lock scoped_lock(spacesVector_mutex);
                    vectorMutex->lock();
                    spacesVector.push_back(i);
                    vectorMutex->unlock();
                    isBlack = false;

                    std::cout << "Added to output vector" << std::endl;
                }
            }
        }

        // Setup the threads and call the scan function
        //void FindSpaces(std::vector<uint16_t> &firstHalfSpaces, std::vector<uint16_t> &secondHalfSpaces) {
        void FindSpaces(std::vector<uint16_t> &spacesVector) {

            auto beginTime = std::chrono::high_resolution_clock::now();

            std::thread firstThread = std::thread([&spacesVector, *this] {

                this->ImageProcessor::ThreadFunction(0, this->inputImage.cols / 2, spacesVector, 
                    this->inputImage.rows, this->spacesVector_mutex);
            });
            std::thread secondThread = std::thread([&spacesVector, *this] {

                this->ImageProcessor::ThreadFunction(this->inputImage.cols / 2, this->inputImage.cols, spacesVector, 
                    this->inputImage.rows, this->spacesVector_mutex);
            });

            /*
            std::thread firstThread(std::ref(ImageProcessor::ThreadFunction), this, 0, inputImage.cols / 2, spacesVector, 
                inputImage.rows);
            
            std::thread secondThread(std::ref(ImageProcessor::ThreadFunction), this, inputImage.cols / 2, inputImage.cols, spacesVector, 
                inputImage.rows);
            */

            firstThread.join();
            secondThread.join();

            auto endTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime);

            std::cout << "Time to finish space scanning for threads: " << elapsedTime.count() << std::endl;
        }

        // Find the spaces between individual digits without the use of threads
        void FindSpacesWithoutThreads(std::vector<uint16_t> &spacesVector) {

            auto beginTime = std::chrono::high_resolution_clock::now();

            bool isBlack = false;
            for(int i = 0; i < inputImage.cols; i++) {

                uint16_t count = 0;
                for(int k = 0; k < inputImage.rows; k++) {

                    if(inputImage.at<uchar>(k, i) == 255) {

                        count++;
                    }
                    else {

                        if(!isBlack) isBlack = true;
                    }
                }

                if(count == inputImage.rows && isBlack) {

                    spacesVector.push_back(i);
                    isBlack = false;
                }
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime);

            std::cout << "Time to finish space scanning without thread: " << elapsedTime.count() << std::endl;
        }

        // Resize the image to whatever dimsension is desired
        void ImageResize(cv::Mat &inputDigit, cv::Mat &outputImage, cv::Size size) {

            //cv::Mat inputDigit = digits.at(whichDigit);

            // Create the mat that will store the values to be averaged
            cv::Mat roi = cv::Mat(size, CV_8U, 255);

            std::vector<uint> countVector = {0, 0};
            for(int i = 0; i < inputDigit.cols; i += size.width) {

                for(int j = 0; j < inputDigit.rows; j += size.height) {

                    // From the input digit mat, add the intensity values to the cv::Mat
                    inputDigit(cv::Rect(i, j, size.width, size.height)).copyTo(roi(cv::Rect(0, 0, size.width, size.height)));

                    // Calculate the average intensity between the values extracted from the input image and add it to the output Image
                    int average = AverageIntensity(roi);
                    outputImage.at<uchar>(countVector.at(0), countVector.at(1)) = average;

                    countVector.at(0)++;
                }

                countVector.at(0) = 0;
                countVector.at(1)++;
            }

            //return outputImage;
        }

        // Function to find the average intensity form a mat
        static int AverageIntensity(cv::Mat& roi) {

            int average = 0;
            for(int i = 0; i < roi.rows; i++) {

                for(int k = 0; k < roi.cols; k++) {

                    average += roi.at<uchar>(i, k);
                }
            }

            return (average / (roi.rows * roi.cols));
        }

        // Function to separate the digits from the input image
        // and add it to the class member vector where they can later be accessed
        void SeparateDigits(std::vector<uint16_t> separatedDigits) {

            uint startIndex = 0;
            uint height = inputImage.rows;

            for_each(separatedDigits.begin(), separatedDigits.end(), 
                [startIndex, height, this](uint16_t i) {

                cv::Rect myROI = cv::Rect(startIndex, 0, i - startIndex, height);
                digits.push_back(inputImage(myROI));

                ShowImage(inputImage(myROI));
            });
        }

        // Function that will resize and center any arbitrary digit from digit vector
        // to be ready for digit i.d.
        void PrepareDigitId(uint whichDigit) {

            // Extract the correct digit from the digit vector
            cv::Mat digit = digits.at(whichDigit);

            cv::Mat outputMat;
            if(digit.rows > digit.cols) {

                outputMat = cv::Mat(cv::Size(digit.rows, digit.rows), CV_8U, 255);
            }
            else {

                outputMat = cv::Mat(cv::Size(digit.cols, digit.cols), CV_8U, 255);
            }

            std::cout << "Digit specs: rows " << digit.rows << " columns " << digit.cols << std::endl;

            uint lineLocation = digit.cols - 1;
            digit.copyTo(outputMat(cv::Rect(0, 0, digit.cols, digit.rows)));
            RemoveHorizontalLine(lineLocation, outputMat);

            uint remainder = 28 - (outputMat.rows % 28);

            cv::Mat outputImage = cv::Mat(cv::Size(outputMat.rows + remainder, outputMat.cols + remainder), CV_8U, 255);
            outputMat.copyTo(outputImage(cv::Rect(remainder / 2, 0, outputMat.cols, outputMat.rows)));

            uint cropSubtraction = 448;
            cv::Rect myroi(cropSubtraction, cropSubtraction, outputImage.rows - ( 2 * cropSubtraction), outputImage.cols - (2 * cropSubtraction));
            cv::Mat outputMatTwo = outputImage(myroi);

            std::cout << "showing the crop difference" << std::endl;

            uint factor = outputMatTwo.cols / 28;
            cv::Mat finalImage = cv::Mat(cv::Size(28, 28), CV_8U, 255);
            ImageResize(outputMatTwo, finalImage, cv::Size(factor, factor));

            digits.at(whichDigit) = finalImage;
        }
};