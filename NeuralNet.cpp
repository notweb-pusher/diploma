#include "NeuralNet.h"
#include <chrono>
#include <ranges>

Matrix NeuralNet::pass_forward(const Matrix &input)
{
    Matrix output = input;
    for (auto &layer : layers_)
        output = layer->passForward(output);
    return output;
}

void NeuralNet::backprop(CAnyOptimizer &optimizer, const Vector &gradient)
{
    Vector cur_grad = gradient;
    for (auto &layer : std::ranges::reverse_view(layers_))
        cur_grad = layer->backprop(optimizer, cur_grad);
}

Matrix OneHotEncode(const Vector &labels)
{
    Matrix encoded_labels = Matrix::Zero(10, labels.size());
    for (size_t index = 0; index < labels.size(); index++)
        encoded_labels(static_cast<int>(labels[index]), index) = 1.0;
    return encoded_labels;
}

Vector getPredicts(const Matrix &predicts, size_t batchSize)
{
    Vector labels(batchSize);
    for (Eigen::Index index = 0; index < batchSize; index++)
    {
        Eigen::Index max_index;
        predicts.col(index).maxCoeff(&max_index);
        labels[index] = static_cast<int8_t>(max_index);
    }
    return labels;
}

int countCorrect(const Vector &predicts, const Vector &labels)
{
    assert(("Sizes missmatch", (predicts.cols() == labels.cols()) && (predicts.rows() == labels.rows())));
    int correct_guess_num = 0;
    for (Eigen::Index i = 0; i < predicts.size(); i++)
        correct_guess_num += static_cast<int>(predicts[i] == labels[i]);
    return correct_guess_num;
}

void runMNIST()
{
    CAnyLayer layer1 = LinearLayer(784, 128);
    CAnyLayer activation1 = CwiseActivation::ReLu();

    // CAnyLayer activation_relu = CwiseActivation::ReLu();
    // CAnyLayer activation_sigmoid = CwiseActivation::Sigmoid();

    CAnyLayer layer2 = LinearLayer(128, 10);
    CAnyLayer activation2 = SoftmaxLayer();

    NeuralNet nn;

    nn.add_layer(std::move(layer1));
    nn.add_layer(std::move(activation1));
    nn.add_layer(std::move(layer2));
    nn.add_layer(std::move(activation2));

    // Optimizer
    // CAnyOptimizer optimizer = Optimizer(0.001);  // Learning rate and momentum
    // CAnyOptimizer optimizer = MomentumOptimizer(0.001, 0.92);  // Learning rate and momentum
    CAnyOptimizer optimizer = AdamOptimizer(0.002, 0.96, 0.999); // Learning rate and momentum
    // AdamOptimizer(0.003, 0.95, 0.999); optimal

    // Loss
    MSELoss loss;

    // Hyperparams
    size_t train_size = 60000;
    size_t test_size = 10000;
    size_t batch_size = 3;
    size_t epochs = 100;
    size_t runs_in_epoch = train_size / batch_size;

    // Get data
    DataProvider reader_train;
    DataProvider reader_test;
    reader_train.read_csv(mnist::path_train, train_size);
    reader_test.read_csv(mnist::path_test, test_size);
    Batch test_data = reader_test.getBatch(test_size);

    // double kEarlyStop = 0.005;
    for (size_t epoch = 1; epoch < epochs + 1; epoch++)
    {
        auto tp1 = std::chrono::system_clock::now();
        optimizer->decayLearningRate(0.01);
        for (size_t batch_run = 0; batch_run < runs_in_epoch; batch_run++)
        {
            auto batch = reader_train.getBatch(batch_size);
            Matrix prediction = nn.pass_forward(batch.features.transpose());
            Matrix target_vectors = OneHotEncode(batch.labels);

            Vector loss_vector = loss.computeLoss(prediction, target_vectors);
            Vector predicts = getPredicts(prediction, batch_size);

            Vector gradients = loss.computeGradient(prediction, target_vectors);
            nn.backprop(optimizer, gradients);
        }
        Matrix prediction_test = nn.pass_forward(test_data.features.transpose());
        Vector predicts = getPredicts(prediction_test, test_size);
        Vector loss_vector = loss.computeLoss(prediction_test, OneHotEncode(test_data.labels));
        std::cout << (std::chrono::system_clock::now() - tp1).count() << ' ';
        std::cout << epoch << ';' << 100.0 * countCorrect(predicts, test_data.labels) / test_size << ';'
                  << loss_vector.mean() << std::endl;
    }
}
