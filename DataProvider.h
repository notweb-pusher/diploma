#pragma once

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "EigenProxy.h"
#include "mnist.h"

struct Batch
{
    Batch(size_t size) : labels(size), features(size, mnist::kImageSize) {}
    Batch(Matrix &&features, Vector &&labels) : labels(std::move(labels)), features(std::move(features)) {}

    Vector labels;
    Matrix features;
};

class DataProvider
{
public:
    Batch getBatch(size_t size)
    {
        std::random_device rnd_device;
        // Specify the engine and distribution.
        std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
        std::uniform_int_distribution<int> dist{0, static_cast<int>(labels_.size() - 1)};

        auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

        std::vector<int> vec(size);
        std::generate(begin(vec), end(vec), gen);
        std::vector<int> all_cols(mnist::kImageSize);
        std::iota(std::begin(all_cols), std::end(all_cols), 0);
        return {features_(vec, all_cols), labels_(vec)};
    }

    void read_csv(const std::filesystem::path &path, size_t sample_size)
    {
        std::ifstream file(path);
        assert(("Sample size < 0", sample_size > 0));

        if (!file.is_open())
        {
            std::cerr << "Error while opening.";
            return;
        }

        Index current_batch = 0;
        std::string line;

        labels_.resize(sample_size);
        features_.resize(sample_size, mnist::kImageSize);

        // Skip column naming line
        std::getline(file, line);
        while ((std::getline(file, line)) && (current_batch < sample_size - 1))
        {
            Index current_el = 0;
            std::stringstream ss(line);
            std::string item;
            std::getline(ss, item, ',');
            labels_(current_batch) = std::stod(item);
            while (std::getline(ss, item, ','))
            {
                features_(current_batch, current_el) = std::stod(item) / mnist::kMaxPixelValue;
                current_el++;
            }
            current_batch++;
        }
        file.close();
    }

private:
    Matrix features_;
    Vector labels_;
};
