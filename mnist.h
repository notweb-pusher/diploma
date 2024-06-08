namespace mnist
{
    constexpr int16_t kRowSize = 28;
    constexpr int16_t kColumnSize = 28;
    constexpr int16_t kImageSize = 784;
    constexpr int16_t kNumLabels = 10;
    constexpr double kMaxPixelValue = 255.0;
    const std::filesystem::path path_train = "../data/mnist_train.csv";
    const std::filesystem::path path_test = "../data/mnist_test.csv";
} // namespace mnist
