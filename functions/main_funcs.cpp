#include <vector>
#include <cmath>

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
