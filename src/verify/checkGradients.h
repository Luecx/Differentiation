//
// Created by Luecx on 17.11.2021.
//

#ifndef DIFFERENTIATION_SRC_VERIFY_CHECKGRADIENTS_H_
#define DIFFERENTIATION_SRC_VERIFY_CHECKGRADIENTS_H_

#include "../network/network.h"
#include "../misc/logging.h"

/**
 * checks gradients of the network, given the specific input and target to compute the gradients/loss
 * Checks the gradients within the given layer index. Applies method to bias and weights
 * @param network
 * @param network_input
 * @param target
 * @param layer
 * @param offset
 * @param margin
 */
void check_gradients(Network& network, Input& network_input, Data& target, int layer, float offset=0.01, float margin = 0.05){

    Data* data_to_check[2] = {network.getLayer(layer)->getWeights(),
                              network.getLayer(layer)->getBias()};
    Data* data_grads   [2] = {network.getThreadData(0)->weight_gradient[layer],
                              network.getThreadData(0)->bias_gradient  [layer]};

    data_grads[0]->clear();
    data_grads[1]->clear();

    network.train(network_input, target, false);

    for (int data_index = 0; data_index < 2; data_index++){
        std::cout << " --------------------------------------" << data_index << " --------------------------------------" << std::endl;
        for(int i = 0; i < data_to_check[data_index]->size(); i++){

            // get the current value
            float mid_point = data_to_check[data_index]->get(i);

            // make sure the window is not 0
            float eval_points[2]{
                mid_point - offset,
                mid_point + offset};
            float eval_difference = 2 * offset;
            float evaluations[2]{};

            for(int j = 0; j < 2; j++){
                // set the value to the eval point
                data_to_check[data_index]->get(i) = eval_points[j];

                // evaluate
                network.evaluate(network_input);

                // compute and track loss
                float loss = network.getLoss()->apply(network.getOutput(), &target);
                evaluations[j] = loss;
            }


            // reset values
            data_to_check[data_index]->get(i) = mid_point;

            // compute numerical gradient using:
            // df/dx ~ (f(x+h) - f(x-h)) / 2h
            float numerical_gradient = (evaluations[1] - evaluations[0]) / eval_difference;

            // get analytical (backproped gradient)
            float backprop_gradient  = data_grads[data_index]->get(i);

            // compute the difference
            float normed_difference  = std::abs((numerical_gradient - backprop_gradient) / numerical_gradient);
            if(normed_difference > margin && std::abs(numerical_gradient) > 1e-5){
                logging::write("[ERROR] Gradients do not match; [backprop: "
                               + std::to_string(backprop_gradient)
                               + "   numerical: "
                               + std::to_string(numerical_gradient) + "]" );
            }else{
//                logging::write("[CORRECT] Gradients do match; [backprop: "
//                                   + std::to_string(backprop_gradient)
//                                   + "   numerical: "
//                                   + std::to_string(numerical_gradient) + "]" );
            }
        }
    }


}

#endif    // DIFFERENTIATION_SRC_VERIFY_CHECKGRADIENTS_H_
