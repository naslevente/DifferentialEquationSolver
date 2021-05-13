#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <eigen-git-mirror/Eigen/Core>
//#include <Eigen/Dense>
#include <pthread.h>
#include <fast-cpp-csv-parser/csv.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#ifndef HIDDENSIZE
#define HIDDENSIZE 512
#endif

#define OUTPUTSIZE 36

template <class InputWeight, class HiddenWeight, class OutputWeight, class InputBias, 
    class HiddenBias, class OutputBias>
class DigitIdentification {

    using Row = std::tuple<float>;

    private :

        Eigen::Matrix<float, 1, 784> imageMatrix;

        InputWeight layerOneWeights;
        HiddenWeight layerTwoWeights;
        HiddenWeight layerThreeWeights;
        OutputWeight layerFourWeights;

        InputBias layerOneBias;
        HiddenBias layerTwoBias;
        HiddenBias layerThreeBias;
        OutputBias layerFourBias;

        // Convert the 28x28 image into a row vector that can be passed
        // through the neural network
        void ImageToVector(cv::Mat& inputImage) {

            //std::cout << inputImage.cols << std::endl;

            uint count = 0;
            for(int i = 0; i < inputImage.rows; i++) {

                for(int k = 0; k < inputImage.cols; k++) {

                    //std::cout << count << std::endl;
                    float value = inputImage.at<uchar>(i, k);
                    imageMatrix(0, count) = value;

                    count++;
                }
            }
        }

        template <std::size_t... Idx, typename T, typename R>
        bool ReadRow(std::index_sequence<Idx...>, T& row, R& reader) {

            return reader.read_row(std::get<Idx>(row)...);
        }

    public:

        /*
        DigitIdentification(std::string layer1Weights, std::string layer2Weights, std::string layer3Weights,
            std::string layer1Bias, std::string layer2Bias, std::string layer3Bias, cv::Mat inputImage) {

            layerOneWeights = ReadCsv(layer1Weights, 784, HIDDENSIZE);
            layerTwoWeights = ReadCsv(layer2Weights, HIDDENSIZE, HIDDENSIZE);
            layerThreeWeights = ReadCsv(layer3Weights, HIDDENSIZE, 10);

            layerOneBias = ReadCsv(layer1Bias, 1, HIDDENSIZE);
            layerTwoBias = ReadCsv(layer2Bias, 1, HIDDENSIZE);
            layerThreeBias = ReadCsv(layer3Bias, 1, 10);
        }
        */

        DigitIdentification(cv::Mat& inputImage) {

            ImageToVector(inputImage);
        }

        // Template function to accept matrices with varying dimensions
        template <typename M> 
        void ReadCsv(M& outputMatrix, std::string inputFile, uint32_t rows, uint32_t cols) {

            io::CSVReader<1> reader(inputFile);
            Row rowTemplate;

            std::cout << "Reading CSV data" << std::endl;

            uint32_t row = 0;
            try {

                bool isRead = false;

                uint32_t count = 0;
                while(!isRead) {

                    isRead = !ReadRow(std::make_index_sequence<std::tuple_size<Row>::value>{}, 
                        rowTemplate, reader);
                    if(!isRead) {

                        if(std::get<0>(rowTemplate) != 0) {

                            outputMatrix(row, count) = std::get<0>(rowTemplate);
                            count++;
                        }
                    }

                    if((count) == cols) {

                        count = 0;
                        row++;
                    }
                }

                std::cout << "Finished Reading" << std::endl;

            } catch(const io::error::no_digit& err) {

                std::cerr << err.what() << std::endl;
            }
        }

        // Function to read in each csv file containing the weights and biases of the neural network
        void SetUpNeuralNetwork(std::string weightOne, std::string weightTwo, std::string weightThree, std::string weightFour,
            std::string biasOne, std::string biasTwo, std::string biasThree, std::string biasFour) {

            ReadCsv(layerOneWeights, weightOne, 784, HIDDENSIZE);
            ReadCsv(layerTwoWeights, weightTwo, HIDDENSIZE, HIDDENSIZE);
            ReadCsv(layerThreeWeights, weightThree, HIDDENSIZE, HIDDENSIZE);
            ReadCsv(layerFourWeights, weightFour, HIDDENSIZE, OUTPUTSIZE);

            ReadCsv(layerOneBias, biasOne, 1, HIDDENSIZE);
            ReadCsv(layerTwoBias, biasTwo, 1, HIDDENSIZE);
            ReadCsv(layerThreeBias, biasThree, 1, HIDDENSIZE);
            ReadCsv(layerFourBias, biasFour, 1, OUTPUTSIZE);
        }

        // Enter the input image through the neural network to find the detected digit
        int DetectDigit() {

            Eigen::Matrix<float, 1, HIDDENSIZE> layerOneOutputs;
            Dot(imageMatrix, layerOneWeights, layerOneOutputs);
            layerOneOutputs = layerOneOutputs - layerOneBias;

            Eigen::Matrix<float, 1, HIDDENSIZE> layerTwoOutputs;
            Dot(layerOneOutputs, layerTwoWeights, layerTwoOutputs);
            layerTwoOutputs = layerTwoOutputs - layerTwoBias;

            Eigen::Matrix<float, 1, HIDDENSIZE> layerThreeOutputs;
            Dot(layerTwoOutputs, layerThreeWeights, layerThreeOutputs);
            layerThreeOutputs = layerThreeOutputs - layerThreeBias;

            Eigen::Matrix<float, 1, OUTPUTSIZE> layerFourOutputs;
            Dot(layerThreeOutputs, layerFourWeights, layerFourOutputs);
            layerFourOutputs = layerFourOutputs - layerFourBias;

            float output = 0;
            for(int i = 0; i < OUTPUTSIZE; i++) {

                std::cout << layerFourOutputs(0, i) << " " << std::endl;

                if(layerFourOutputs(0, i) > output) {

                    output = i;
                }
            }

            return output;
        }

        // Function for dot product between Eigen matrices
        template <typename M, typename T, typename K>
        void Dot(M firstMatrix, T secondMatrix, K &outputMatrix) {

            std::cout << "First Matrix: rows " << firstMatrix.rows() << " cols " << firstMatrix.cols() << std::endl;
            std::cout << "Second Matrix: rows " << secondMatrix.rows() << " cols " << secondMatrix.cols() << std::endl;

            for(int i = 0; i < secondMatrix.cols(); i++) {

                int sum = 0;
                for(int k = 0; k < firstMatrix.cols(); k++) {

                    sum += (firstMatrix(0, k) * secondMatrix(k, i));
                }

                outputMatrix(0, i) = sum;
            }
        }
};