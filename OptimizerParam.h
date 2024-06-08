#pragma once

#include "EigenProxy.h"

class Params
{
public:
    Params() = default;
    Params(int number_rows, int number_columns) :
        m_weights_(Matrix::Zero(number_rows, number_columns)), m_biases_(Vector::Zero(number_rows)),
        v_weights_(Matrix::Zero(number_rows, number_columns)), v_biases_(Vector::Zero(number_rows)),
        v_hat_weights_(Matrix::Zero(number_rows, number_columns)), v_hat_biases_(Vector::Zero(number_rows)),
        num_iter_(0)
    {
    }

    int num_iter_;
    Matrix m_weights_;
    Vector m_biases_;
    Matrix v_weights_;
    Vector v_biases_;
    Matrix v_hat_weights_;
    Vector v_hat_biases_;
};
