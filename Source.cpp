#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <torch/torch.h>
//#include <boost_1_66_0/boost/thread>
#include <eigen-git-mirror/Eigen/Core>
#include "ImageProcessor.hpp"
#include "DigitIdentification.hpp"
#include "ConvDigitIdentification.hpp"
#include <fast-cpp-csv-parser/csv.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#define HIDDENSIZE 512
#define INPUTSIZE 784
#define OUTPUTSIZE 36

// Need to initialize the mutex in ImageProcessor before using it inside the main
// function or any other function or else we receive an undefined reference error
std::mutex *ImageProcessor::spacesVector_mutex = new std::mutex();

cv::Mat ProcessImage(char* inputFile) {

    // Read in the image and set it equal to a cv::Mat
    cv::Mat image;
    image = cv::imread(inputFile, 1);

    // Check to make sure the image was loaded
    if(!image.data) {

        std::cout << "No image data" << std::endl;
    }

    ImageProcessor::ShowImage(image);

    // Convert image to grayscale and convert to binary image
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::threshold(image, image, 90, 255, cv::THRESH_BINARY);

    // Create ImageProcessor object for the input image to be scanned and read
    ImageProcessor imageObject = ImageProcessor(image);
    ImageProcessor* imageObjectPtr = &imageObject;
    imageObjectPtr->RemoveNoise();
    
    // The method will return a vector with all the indices that are spaces
    // and a vector with all the indices marking the beginning of a new digit

    // Find the spaces without the threads to compare execution times
    std::vector<uint16_t> separatedDigits;
    imageObjectPtr->FindSpacesWithoutThreads(separatedDigits);
    
    // Find the spaces using the thread object
    //std::vector<uint16_t> firstHalfSpaces;
    //std::vector<uint16_t> secondHalfSpaces;
    //std::vector<uint16_t> spacesVector(image.cols);
    std::vector<uint16_t> spacesVector;
    imageObjectPtr->FindSpaces(spacesVector);

    for(int i = 0; i < spacesVector.size(); i++) {

        std::cout << "space at: " << spacesVector.at(i) << std::endl;
        cv::line(image, cv::Point(spacesVector.at(i), 0), cv::Point(spacesVector.at(i), 
            image.rows - 1), cv::Scalar(0 , 0, 255));
    }

    std::cout << "Showing Image: " << std::endl;

    imageObjectPtr->SeparateDigits(spacesVector);
    imageObjectPtr->PrepareDigitId(0);

    cv::Mat resizedImage = imageObjectPtr->getDigit(0);
    resizedImage = ~resizedImage;

    std::cout << "Finished preparing the digits" << std::endl;

    return resizedImage;
}

void PrepareNeuralNetwork(std::string numberFolderPath, std::string letterFolderPath, 
std::vector<std::string> fileNames, cv::Mat inputImage) {

    // create the conv neural network object for number identification
    ConvDigitIdentification digitIdObject = ConvDigitIdentification(numberFolderPath);
    ConvDigitIdentification* digitIdPtr = &digitIdObject;

    digitIdPtr->SetupNeuralNetwork(fileNames, 10);

    // create the conv neural network object for letter identification
    ConvDigitIdentification letterIdObject = ConvDigitIdentification(letterFolderPath);
    ConvDigitIdentification* letterIdPtr = &letterIdObject;

    letterIdPtr->SetupNeuralNetwork(fileNames, 26);

    // pass the input image through the conv neural networks
    torch::Tensor outputTensor = digitIdPtr->FeedForward(inputImage);
    torch::Tensor letterOutputTensor = letterIdPtr->FeedForward(inputImage);

    std::cout << "output from the conv neural network: " << outputTensor << std::endl;
}

int main(int argc, char* argv[]) {

    // Check to make sure the right number of arguments was sent to the program
    if(argc <= 3) {

        std::cout << "Wrong number of arguments" << std::endl;
        return -1;
    }

    // process image
    //int Process = ProcessImage(argv[1]);

    std::string fileTemplate = std::string("weights_biases_");

    std::vector<std::string> fileNames(10);
    for(int i = 1; i <= fileNames.size(); ++i) {

        std::string completeFileName = fileTemplate + std::to_string(i) + ".csv";
        fileNames.at(i - 1) = completeFileName;
    }

    // setup the conv neural network for digit identification and process the input image
    cv::Mat resultMatPtr = ProcessImage(argv[1]);
    cv::Mat floatMat;
    resultMatPtr.convertTo(floatMat, CV_32F);
    PrepareNeuralNetwork(argv[2], argv[3], fileNames, floatMat);

    return 0;
}