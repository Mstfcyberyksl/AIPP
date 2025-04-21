#ifndef MYSTRUCT_H
#define MYSTRUCT_H

#include <vector>
#include <string>
#include <optional>
struct FNN{
    std::vector<int> layer_sizes;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;

    std::string activation_func ;
    std::string optimizer ;
    std::string loss_func ;
    FNN(const std::vector<int>& layer_sizes);
    int predict(std::vector<float>& data);
    std::vector<std::vector<float>> forward_pass(std::vector<std::vector<float>>& data, std::string activation) const;
    float calc_loss(std::string loss_func) const;
    void backpropogate(std::string optimizer, std::string loss_func);

    void train( std::optional<std::string> loss_func,
                std::optional<std::string> optimizer ,
                std::optional<std::string> activation_func,
                std::optional<float> learning_rate = 0.1f,
                std::optional<int> batch_size = 16,
                std::optional<int> epoch = 5,
                std::optional<bool> shuffle = true,
                std::vector<std::vector<float>>* X_train = nullptr,
                std::vector<std::vector<float>>* y_train = nullptr);

    void test(std::vector<std::vector<float>>& data_to_test);
    void seed(int seed);
};

#endif
