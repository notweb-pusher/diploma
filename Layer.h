#pragma once

#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "EigenProxy.h"
#include "Optimizer.h"

#include "AnyObject.h"

template <class TBase>
class ILayer : public TBase
{
private:
    Matrix input_;

public:
    virtual Matrix passForward(const Matrix &input) = 0;

    virtual Vector backprop(CAnyOptimizer &optimizer, const Vector &u) = 0;
};


template <class TBase, class TObject>
class CLayerImpl : public TBase
{
    using CBase = TBase;
    Matrix input_;

public:
    using CBase::CBase;

    Matrix passForward(const Matrix &input) override { return CBase::Object().passForward(input); }

    Vector backprop(CAnyOptimizer &optimizer, const Vector &u) override
    {
        return CBase::Object().backprop(optimizer, u);
    }
};

class CAnyLayer : public CAnyMovable<ILayer, CLayerImpl>
{
    using CBase = CAnyMovable<ILayer, CLayerImpl>;

public:
    using CBase::CBase;
};

class LinearLayer
{
private:
    Matrix input_;
    Matrix weights_;
    Vector bias_;
    Params params_;

public:
    LinearLayer(int input_size, int output_size) :
        weights_(Matrix::Random(output_size, input_size)), bias_(Vector::Random(output_size)),
        params_(Params(output_size, input_size))
    {
    }

    Matrix passForward(const Matrix &input);

    Vector backprop(CAnyOptimizer &optimizer, const Vector &u);
};

class SoftmaxLayer
{
private:
    Matrix input_;

public:
    Matrix passForward(const Matrix &input);

    static Matrix getJacobian(const Matrix &output);

    Vector backprop(CAnyOptimizer &optimizer, const Vector &u);
};


class CwiseActivation
{
    using Function = std::function<double(double)>;

private:
    Function f0_;
    Function f1_;
    Matrix input_;

public:
    CwiseActivation(Function f0, Function f1) : f0_(std::move(f0)), f1_(std::move(f1)) {}

    Matrix passForward(const Matrix &input)
    {
        input_ = input;
        return input.unaryExpr(f0_);
    }

    Matrix backprop(CAnyOptimizer &optimizer, const Vector &u)
    {
        return passForward(input_).unaryExpr(f1_).rowwise().mean().asDiagonal() * u;
    }

    static CwiseActivation ReLu()
    {
        return {[](double x) { return x > 0 ? x : 0.01 * x; }, [](double y) { return y > 0 ? 1 : 0.01; }};
    }

    static CwiseActivation Sigmoid()
    {
        return {[](double x) { return 1.0 / (1.0 + std::exp(-x)); }, [](double y) { return y * (1 - y); }};
    }
};
