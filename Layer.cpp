#include "Layer.h"
#include <cassert>
#include <iostream>

Matrix LinearLayer::passForward(const Matrix &input)
{
    input_ = input;
    const Matrix bias = bias_.replicate(1, input.cols());
    return weights_ * input + bias;
}

Vector LinearLayer::backprop(CAnyOptimizer &optimizer, const Vector &gradient)
{
    Matrix original_weights = weights_;
    Vector original_biases = bias_;

    Matrix dA = Matrix::Zero(weights_.rows(), weights_.cols());
    int batch_size = input_.cols();
    for (size_t index = 0; index < batch_size; index++)
        dA += gradient * input_.col(index).transpose();

    dA /= batch_size;

    Vector u_bar = weights_.transpose() * gradient;
    optimizer->update(weights_, bias_, dA, gradient, params_);
    return u_bar;
}

Vector softMaxVector(const Matrix &input)
{
    // We subtract input.maxCoeff() to deal with overfloat after exp()
    const Vector exps = (input.array() - input.maxCoeff()).exp();
    return exps / exps.sum();
}

Matrix SoftmaxLayer::passForward(const Matrix &input)
{
    input_ = input;
    Matrix ans = Matrix::Zero(input.rows(), input.cols());

    for (int i = 0; i < input.cols(); ++i)
        ans.col(i) = softMaxVector(input.col(i));

    return ans;
}


Matrix SoftmaxLayer::getJacobian(const Matrix &output)
{
    const Matrix diagonal = output.asDiagonal();
    return diagonal - output * output.transpose();
}

Vector SoftmaxLayer::backprop(CAnyOptimizer &optimizer, const Vector &u)
{
    int batch_size = input_.cols();
    Matrix jacobian = Matrix(u.rows(), u.rows());
    Matrix output = passForward(input_);
    for (size_t idx = 0; idx < batch_size; idx++)
        jacobian += getJacobian(output.col(idx));
    return jacobian * u;
}
