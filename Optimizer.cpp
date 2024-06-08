#include "Optimizer.h"
#include <iostream>

void MomentumOptimizer::update(Matrix &weights, Vector &biases, const Matrix &gradientWeights, const Vector &gradientBiases,
                               Params &params) const
{
    params.m_weights_ = momentum_ * params.m_weights_ + (1 - momentum_) * gradientWeights;
    params.m_biases_ = momentum_ * params.m_biases_ + (1 - momentum_) * gradientBiases;

    weights -= learningRate_ * params.m_weights_;
    biases -= learningRate_ * params.m_biases_;
}

// void

void Optimizer::update(Matrix &weights, Vector &biases, const Matrix &gradientWeights, const Vector &gradientBiases,
                       Params &params) const
{
    weights -= learningRate_ * gradientWeights;
    biases -= learningRate_ * gradientBiases;
}

void AdamOptimizer::update(Matrix &weights, Vector &biases, const Matrix &gradientWeights, const Vector &gradientBiases,
                           Params &params) const
{
    double eps = 1e-8;
    params.num_iter_ += 1;

    // m_t
    params.m_weights_ = beta1_ * params.m_weights_ + (1 - beta1_) * gradientWeights;
    params.m_biases_ = beta1_ * params.m_biases_ + (1 - beta1_) * gradientBiases;

    // v_t
    params.v_weights_ = beta2_ * params.v_weights_ + (1 - beta2_) * gradientWeights.array().square().matrix();
    params.v_biases_ = beta2_ * params.v_biases_ + (1 - beta2_) * gradientBiases.array().square().matrix();

    // m_hat
    Matrix m_hat_weights = params.m_weights_ / (1 - std::pow(beta1_, params.num_iter_));
    Vector m_hat_bias = params.m_biases_ / (1 - std::pow(beta1_, params.num_iter_));

    // v_hat
    params.v_hat_weights_ = params.v_hat_weights_.cwiseMax(params.v_weights_);
    params.v_hat_biases_ = params.v_hat_biases_.cwiseMax(params.v_biases_);

    Matrix v_weight_correction = params.v_hat_weights_ / (1 - std::pow(beta2_, params.num_iter_));
    Vector v_bias_correction = params.v_hat_biases_ / (1 - std::pow(beta2_, params.num_iter_));

    // weights update
    weights -= learningRate_ * (m_hat_weights.array() / (v_weight_correction.array().sqrt() + eps)).matrix();
    biases -= learningRate_ * ((m_hat_bias.array()) / (v_bias_correction.array().sqrt() + eps)).matrix();
}
