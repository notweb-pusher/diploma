#pragma once

#include <iostream>
#include <vector>
#include "DataProvider.h"
#include "EigenProxy.h"
#include "Layer.h"
#include "Loss.h"


class NeuralNet
{
public:
    NeuralNet() = default;

    void add_layer(CAnyLayer &&layer) { layers_.push_back(std::move(layer)); };

    Matrix pass_forward(const Matrix &input);

    void backprop(CAnyOptimizer &optimizer, const Vector &gradient);

    double GetAccuracy(const std::vector<Vector> &predicted, const std::vector<Vector> &true_labels);

private:
    std::vector<CAnyLayer> layers_;
};

Matrix OneHotEncode(const Vector &labels);

Vector getPredicts(const Matrix &predicts, size_t batchSize);

int countCorrect(const Vector &predicts, const Vector &labels);

void runMNIST();
