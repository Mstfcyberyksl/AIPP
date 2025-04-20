#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
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

float mean_squared_error(const vector<float>& actual, const vector<float>& preds){
    float result = 0, diff;
    int i, size1, size2;
    size1 = actual.size();
    size2 = preds.size();

    if (size1 == size2){
        for (i = 0;i < actual.size();i++){
            diff = actual[i] - preds[i];
            result += diff * diff;
        }
        return result / size1;
    }
    throw runtime_error("SHAPE ERROR: The shape of the predictions and the shape of the real values don't match.");
}

int argmax(const vector<float>& Vector){
    int i, result = 0;
    float max = Vector[0];
    for(i = 1;i < Vector.size();i++){
        if (Vector[i] > max){
            max = Vector[i];
            result = i;
        }
    }
    return result;
}

vector<vector<float>> matrix_mult(const vector<vector<float>>& Matrix1, const vector<vector<float>>& Matrix2){
    int i, j, k;
    vector<vector<float>> result(Matrix1.size(),vector<float>(Matrix2[0].size(),0.0f));

    for(i = 0;i < Matrix1.size();i++){
        for(j = 0;j < Matrix2[0].size();j++){
            for(k = 0;k < Matrix1[0].size();k++){
                result[i][j] += Matrix1[i][k] * Matrix2[k][j];
            }
        }
    }
    return result;
}


// We may ask the user to enter the clamp size later on
// TODO: Ask the user the clamp size and make the default 0.0001
// TODO: Add error handling
static float Cross_Entrophy_Loss_Helper(const vector<int>& actual, const vector<float>& preds){
    float result = 0,pred;
    int i;
    for(i = 0;i < actual.size();i++){
        pred = max(0.0001f, min(preds[i],1.0f - 0.0001f));
        result += actual[i] * log(pred);
    }
    return -result;
}

vector<float> Cross_Entrophy_Loss(const vector<vector<int>>& actual, const vector<vector<float>>& preds){
    vector<float> result(actual.size());
    int i;
    for(i = 0;i < actual.size();i++){
        result[i] = Cross_Entrophy_Loss_Helper(actual[i], preds[i]);
    }
    return result;
}
