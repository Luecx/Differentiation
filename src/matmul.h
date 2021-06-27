//
// Created by Luecx on 26.02.2021.
//

#ifndef DIFFERENTIATION_MATMUL_H
#define DIFFERENTIATION_MATMUL_H

// contains matrix-vector multiplications
// since inputs are defined using sparse indices, this is done for sparse and non-sparse inputs

/**
 * standard multiplication of matrix with a vector.
 * Both the weights and the vector which the matrix will be multiplied with are dense.
 * The target vector will contain the result.
 *
 * Conditions:
 * 1) This will only work if the input vector is a multiple of 8
 * 2) The height of the matrix must equal the height of the target
 * 3) The width of the matrix must equal the height of the vector
 *
 * @param weights the weight-matrix
 * @param vector  the vector which will be transformed using the weight-matrix
 * @param target  the target in which the result will be stored
 */
void matmul(
        const Data* weights,
        const Data* vector,
        Data* target);

/**
 * backprop for the standard multiplication of a matrix with a vector.
 * It will compute the gradients for the weights which will be stored in weights_grad
 * as well as the gradients for the input vector which will be stored within vector_grad.
 * The gradient of the output must be given which will be used to compute
 * the gradients of the weights and input vector.
 * Note that the gradients inside weights_grad will be incremented and not overwritten.
 * However gradients for the input vector will be reset.
 *
 * Conditions:
 * 1) This will only work if the input vector is a multiple of 8
 * 2) The height of the matrix must equal the height of the target
 * 3) The width of the matrix must equal the height of the vector
 * 4) The gradients for the weights and vector must have the same size as the corresponding values.
 *
 * @param weights           the weight-matrix
 * @param vector            the vector which will be transformed using the weight-matrix
 * @param weights_grad      the gradient for the weight-matrix
 * @param vector_grad       the gradient for the input-vector
 * @param target_grad       the gradient of the output of the matrix-multiplication
 */
void matmul_backprop(
        const Data* weights,
        const Data* vector,
        Data* weights_grad,
        Data* vector_grad,
        const Data* target_grad);

/**
 * backprop for the standard multiplication of a matrix with a vector.
 * It will compute the gradients for the weights which will be stored in weights_grad
 * as well as the gradients for the input vector which will be stored within vector_grad.
 * The gradient of the output must be given which will be used to compute
 * the gradients of the weights.
 * Note that the gradients inside weights_grad will be incremented and not overwritten.
 *
 * Conditions:
 * 1) This will only work if the input vector is a multiple of 8
 * 2) The height of the matrix must equal the height of the target
 * 3) The width of the matrix must equal the height of the vector
 * 4) The gradients for the weights and vector must have the same size as the corresponding values.
 *
 * @param weights           the weight-matrix
 * @param vector            the vector which will be transformed using the weight-matrix
 * @param weights_grad      the gradient for the weight-matrix
 * @param target_grad       the gradient of the output of the matrix-multiplication
 */
void matmul_backprop(
    const Data *weights,
    const Data *vector,
    Data *weights_grad,
    const Data *target_grad);

/**
 * backprop for the standard multiplication of a matrix with a vector.
 * The input to vector is given in a sparse format where only the indices of the activated inputs are given.
 * All activated inputs are considered to output "1".
 * The target vector will contain the result.
 * Note that the function does not check if the given indices
 * are valid and in a feasible range for the weight matrix!
 *
 * Conditions:
 * 1) The height of the matrix must equal the height of the target
 *
 * @param weights the weight-matrix
 * @param vector  the sparse vector which will be transformed using the weight-matrix
 * @param target  the target in which the result will be stored
 */
void matmul(
        const Data*  weights,
        const Input* vector,
        Data*  target);

/**
 * backprop for the sparse multiplication of a matrix with a vector.
 * The input to vector is given in a sparse format where only the indices of the activated inputs are given.
 * All activated inputs are considered to output "1".
 * The function will compute the gradients for the weights which have been used during the multiplication based
 * on the gradients of the outcome of the multiplication which is given as target_grad.
 * Note that the gradients inside weights_grad will be incremented and not overwritten.
 * Furthermore note that the function does not check if the given indices
 * are valid and in a feasible range for the weight matrix!
 *
 * Conditions:
 * 1) The height of the matrix must equal the height of the target
 *
 * @param vector        the sparse vector which was transformed using the weight matrix
 * @param weights_grad  the sparse vector which was transformed using the weight matrix
 * @param target_grad   the target in which the result will be stored
 */
void matmul_backprop(
        Input    * vector,
        Data     * weights_grad,
        Data     * target_grad);

/**
 * sparse multiplication of matrix with a vector.
 * The input to vector is given in a sparse format where only the indices of the activated inputs are given.
 * All activated inputs are considered to output "1".
 * The target vector will contain the result.
 * inputOffset will apply a shift to the indices which will also be checked for legality.
 * This is especially useful if only a portion of the inputs is used for the multiplication.
 * Indices will only be used if weights->height + inputOffset > index >= inputOffset
 *
 * Conditions:
 * 1) The height of the matrix must equal the height of the target
 *
 * @param weights       the weight-matrix
 * @param vector        the sparse vector which will be transformed using the weight-matrix
 * @param target        the target in which the result will be stored
 * @param inputOffset   the offset which will be applied to the indices
 */
void matmul(
        const Data*  weights,
        const Input* vector,
        Data*  target,
        int inputOffset);

/**
 * backprop for the sparse multiplication of a matrix with a vector.
 * The input to vector is given in a sparse format where only the indices of the activated inputs are given.
 * All activated inputs are considered to output "1".
 * The function will compute the gradients for the weights which have been used during the multiplication based
 * on the gradients of the outcome of the multiplication which is given as target_grad.
 * Note that the gradients inside weights_grad will be incremented and not overwritten.
 * This is especially useful if only a portion of the inputs was used for the multiplication.
 * Indices will only be used if weights->height + inputOffset > index >= inputOffset
 *
 * Conditions:
 * 1) The height of the matrix must equal the height of the target
 *
 * @param vector        the sparse vector which was transformed using the weight matrix
 * @param weights_grad  the sparse vector which was transformed using the weight matrix
 * @param target_grad   the target in which the result will be stored
 * @param inputOffset   the offset which will be applied to the indices
 */
void matmul_backprop(
        Input    * vector,
        Data     * weights_grad,
        Data     * target_grad,
        int inputOffset);


#endif //DIFFERENTIATION_MATMUL_H
