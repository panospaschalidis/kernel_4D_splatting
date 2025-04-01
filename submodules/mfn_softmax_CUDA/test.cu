#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <math.h>

int main(){
    glm::vec4 v(1,2,3,4);
    glm::mat4 Q = glm::mat4(
        1,2,3,4,
        5,6,7,8,
        0,0,0,0,
        0,0,0,0);
    glm::vec4 G = Q*v;
    std::cout << G[0] << std::endl;
    std::cout << G[1] << std::endl;
    std::cout << G[2] << std::endl;
    std::cout << G[3] << std::endl;
    float a = 0.0000001;
    //std::cout << a << std::endl;
    printf("%f\n",a);
    return;
}
