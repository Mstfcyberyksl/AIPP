#include <vector>
#include <cmath>

using namespace std;

vector<float> ReLU(const vector<float> Vector){
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
