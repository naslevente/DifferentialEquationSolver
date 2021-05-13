#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
//#include <Eigen/Dense>
#include <pthread.h>
#include <fast-cpp-csv-parser/csv.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <functional>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace func = torch::nn::functional;

struct ConvNeuralNet : torch::nn::Module {

    ConvNeuralNet(const uint outputSize) {

        conv1 = register_module("conv1", torch::nn::Conv2d(1, 6, 3));
        conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, 3));

        linear1 = register_module("linear1", torch::nn::Linear(16 * 5 * 5, 120));
        linear2 = register_module("linear2", torch::nn::Linear(120, 84));
        linear3 = register_module("linear3", torch::nn::Linear(84, outputSize));
    }

    torch::Tensor forward(torch::Tensor input) {

        input = func::max_pool2d(torch::sigmoid(conv1->forward(input)), func::MaxPool2dFuncOptions({2, 2}));
        input = func::max_pool2d(torch::sigmoid(conv2->forward(input)), func::MaxPool2dFuncOptions(2));

        int num_features = num_flat_features(input);

        input = input.view({-1, num_features});
        input = torch::sigmoid(linear1->forward(input));
        input = torch::sigmoid(linear2->forward(input));
        input = linear3->forward(input);

        return input;
    }

    int num_flat_features(torch::Tensor input) {

        int num_features = 1;

        auto shape = input.sizes();
        for(int i = 1; i < shape.size(); ++i) {

            num_features *= shape[i];
        }

        return num_features;
    }

    torch::nn::Conv2d conv1 {nullptr}, conv2 {nullptr};
    torch::nn::Linear linear1 {nullptr}, linear2 {nullptr}, linear3 {nullptr}; 
};

class ConvDigitIdentification {

    public:

        std::string inputFolderPath;
        torch::Tensor ConvLayerOneTensor;
        std::vector<std::string> fileNames;
        std::shared_ptr<ConvNeuralNet> convNeuralNet;

        ConvDigitIdentification(std::string inputFolderPath) {

            this->inputFolderPath = inputFolderPath;
        }

        void SetupNeuralNetwork(std::vector<std::string> fileNames, const uint outputSize) {

            convNeuralNet = std::make_shared<ConvNeuralNet>(outputSize);

            for(int i = 0; i < fileNames.size(); ++i) {

                this->fileNames.push_back(fileNames.at(i));
                
                std::string completeFilePath = getCompletePath(fileNames.at(i));
                ReadCsv(completeFilePath, i);
            }
            
            //std::vector<torch::Tensor> params = NeuralNet->parameters();
            //std::cout << "neural network conv layer 1: " << params.at(3) << std::endl;
        }

        std::string getCompletePath(std::string fileName) {

            fileName = inputFolderPath.append(fileName);
            return fileName;
        }

        template <typename R, typename T>
        void ReadLine(std::string line, R& outputVector, T incomingValue) {

            std::string word;

            std::stringstream stream(line);
            while(std::getline(stream, word, ',')) {

                if(!word.empty()) {

                    float firstValue = std::stod(word, nullptr);
                    outputVector.push_back(firstValue);
                }

                //auto incomingValue = (*function)(word, nullptr, 10);
                stream >> incomingValue;
                outputVector.push_back(incomingValue);
            }
        }

        void ReadCsv(std::string fileName, const uint index) {

            std::vector<torch::Tensor> networkSize = convNeuralNet->parameters(true);
            auto parameterShape = networkSize.at(index).sizes();
            torch::Tensor outputTensor = torch::zeros(parameterShape);

            int size = parameterShape.size();
            while(size < 4) {

                outputTensor.unsqueeze_(0);
                size += 1;
            }

            std::fstream csvFile;
            csvFile.open(fileName);

            std::vector<float> rowTemplate;
            std::string row, line;

            uint rowIndex = 0;
            uint shapeArgOneIndex = 0;
            uint shapeArgTwoIndex = 0;

            while(csvFile >> row) {

                rowTemplate.clear();

                int col = 0;
                double argumentValueTwo;
                ReadLine(row, rowTemplate, argumentValueTwo);
                for(int i = 0; i < outputTensor.sizes()[3]; ++i) {

                    outputTensor[shapeArgOneIndex][shapeArgTwoIndex][rowIndex][i] = rowTemplate.at(i);
                }

                ++rowIndex;

                if(rowIndex == outputTensor.sizes()[2]) {

                    ++shapeArgTwoIndex;
                    rowIndex = 0;

                    if(shapeArgTwoIndex == outputTensor.sizes()[1]) {

                        ++shapeArgOneIndex;
                        shapeArgTwoIndex = 0;
                    }
                }
            }

            size = parameterShape.size();
            while(size < 4) {

                outputTensor.squeeze_();
                size += 1;
            }

            std::cout << "size of output tensor is: " << outputTensor.sizes() << std::endl;
            networkSize.at(index) = outputTensor;
        }

        torch::Tensor FeedForward(cv::Mat inputMat) {

            std::cout << "converting mat image to torch tensor" << std::endl;

            // conver the input immage from mat to torch::tensor
            torch::Tensor inputTensor = torch::from_blob(inputMat.data, { 1, 1, inputMat.rows, inputMat.cols }, torch::kFloat).to(torch::kFloat);
            //torch::Tensor inputTensor = torch::tensor(at::ArrayRef<float>(inputMat.data, 784)).view({ 28, 28 });

            auto outputTensor = convNeuralNet->forward(inputTensor);

            return outputTensor;
        }

    private:


};
