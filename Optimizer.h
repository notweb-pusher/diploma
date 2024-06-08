#pragma once

#include <vector>
#include "AnyObject.h"
#include "EigenProxy.h"
#include "OptimizerParam.h"


template <class TBase>
class IOptimizer : public TBase
{
public:
    virtual void update(Matrix &weights, Vector &biases, const Matrix &gradientWeights, const Vector &gradientBiases,
                        Params &params);

    virtual void decayLearningRate(double kDecay);
};

template <class TBase, class TObject>
class COptimizerImpl : public TBase
{
    using CBase = TBase;

public:
    using CBase::CBase;

    void update(Matrix &weights, Vector &biases, const Matrix &gradientWeights, const Vector &gradientBiases,
                Params &params) override
    {
        CBase::Object().update(weights, biases, gradientWeights, gradientBiases, params);
    }

    void decayLearningRate(double kDecay) override { CBase::Object().decayLearningRate(kDecay); }

    double learningRate_;
};

class CAnyOptimizer : public CAnyMovable<IOptimizer, COptimizerImpl>
{
    using CBase = CAnyMovable<IOptimizer, COptimizerImpl>;

public:
    using CBase::CBase;
};

class MomentumOptimizer
{
public:
    MomentumOptimizer(double learning_rate, double momentum) : learningRate_(learning_rate), momentum_(momentum)
    {
        assert(("Invalid learning rate", (learningRate_ > 0) && (learningRate_ < 1)));
        assert(("Invalid momentum", (momentum_ > 0) && (momentum_ < 1)));
    }

    void update(Matrix &weights, Vector &biases, const Matrix &gradientWeights, const Vector &gradientBiases, Params &params) const;

    void decayLearningRate(double kDecay) { learningRate_ /= (1 + kDecay); }

    double learningRate_;
    double momentum_;
};

class Optimizer
{
public:
    Optimizer(double learning_rate) : learningRate_(learning_rate)
    {
        assert(("Invalid learning rate", (learningRate_ > 0) && (learningRate_ < 1)));
    }

    void update(Matrix &weights, Vector &biases, const Matrix &gradientWeights, const Vector &gradientBiases, Params &params) const;

    void decayLearningRate(double kDecay) { learningRate_ /= (1 + kDecay); }

    double learningRate_;
};

class AdamOptimizer
{
public:
    AdamOptimizer(double learning_rate, double beta1, double beta2) :
        learningRate_(learning_rate), beta1_(beta1), beta2_(beta2)
    {
        assert(("Invalid learning rate", (learningRate_ > 0) && (learningRate_ < 1)));
        assert(("Invalid beta1", (beta1_ > 0) && (beta1_ < 1)));
        assert(("Invalid beta2", (beta2_ > 0) && (beta2_ < 1)));
    }

    void update(Matrix &weights, Vector &biases, const Matrix &gradientWeights, const Vector &gradientBiases, Params &params) const;

    void decayLearningRate(double kDecay) { learningRate_ /= (1 + kDecay); }

    double learningRate_;
    double beta1_;
    double beta2_;
};
