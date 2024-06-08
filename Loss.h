#pragma once

#include "EigenProxy.h"

class MSELoss
{
public:
    static Vector computeLoss(const Matrix &predicts, const Matrix &targets)
    {
        return (predicts - targets).array().square().colwise().sum();
    }

    static Vector computeGradient(const Matrix &predicts, const Matrix &targets)
    {
        return 2 * (predicts - targets).rowwise().mean();
    }
};

class CrossEntropyLoss
{
public:
    static double computeLoss(const Matrix &predicts, const Matrix &targets)
    {
        return -((targets.array() * predicts.array().log()).sum()) / predicts.rows();
    }

    static Matrix computeGradient(const Matrix &predicts, const Matrix &targets)
    {
        return -(targets.array() / predicts.array()) / predicts.rows();
    }
};
