#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
using namespace std;

vector<float> ReLU(const vector<float> Vector){
    // Change negative numbers to 0
    vector<float> result = Vector;
    int i;

    for (i = 0;i < result.size();i++){
        if (result[i] < 0){
            result[i] = 0;
        }
    }
    return result;
}
vector<float> SoftMax(const vector<float> Vector){
    // 'Squeeze' the distributions between 0 and 1
    vector<float> result = Vector;
    int i;
    float sum = 0.0f;

    for (i = 0;i < result.size();i++){
        result[i] = exp(result[i]);
        sum += result[i];
    }

    for(i = 0;i < result.size();i++){
        result[i] = result[i]/sum;
    }
    return result;
}
vector<float> tanh(const vector<float> Vector){
    // (e^x - e^-x) / (e^x + e^-x)
    vector<float> result = Vector;
    int i;
    float ex, nex;
    for(i = 0;i < result.size();i++){
        ex = exp(result[i]);
        nex = exp(-result[i]);
        result[i] = (ex - nex) / (ex + nex);
    }
    return result;
}
vector<vector<float>> transpose(const vector<vector<float>> Matrix){
    // Change rows to columns and columns to rows
    // example: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    vector<vector<float>> result;
    int i, j, rows, cols;
    rows = Matrix.size();
    cols = Matrix[0].size();
    result.resize(cols);
    for (i = 0;i < cols;i++){
        result[i].resize(rows);
    }
    for (i = 0;i < rows;i++){
        for (j = 0;j < cols;j++){
            result[j][i] = Matrix[i][j];
        }
    }
    return result;
}
vector<float> sigmoid(const vector<float> Vector){
    vector<float> result(Vector.size());
    int i;
    for(i = 0;i < Vector.size();i++){
        result[i] = 1 / (1 + exp(-Vector[i]));
    }
    return result;
}

float mean_absolute_error(const vector<float> pred, const vector<float> actual){
    float result = 0.0f;
    int i, pred_size, act_size;
    pred_size = pred.size();
    act_size = actual.size();

    if (pred_size != act_size){
        throw runtime_error("SHAPE ERROR: The shape of the predictions and the shape of the real values don't match.");
    }

    for(i = 0;i < pred_size;i++){
        result += abs(pred[i] - actual[i]);
    }
    return result / pred_size;
}
