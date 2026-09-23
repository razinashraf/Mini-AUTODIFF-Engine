import numpy as np

def numerical_gradient(loss_fn, parameter, epsilon=1e-3):
    original = parameter.data

    parameter.data = original + epsilon
    loss_plus = loss_fn().data # relative error

    parameter.data = original - epsilon
    loss_minus = loss_fn().data

    parameter.data = original

    return (loss_plus - loss_minus) / (2 * epsilon)

def relative_error(autodiff, numerical, epsilon=1e-12):
    numerator = abs(autodiff - numerical)
    denominator = max(abs(autodiff), abs(numerical), epsilon)

    return numerator / denominator


def gradient_check(loss_fn, parameter, epsilon=1e-3, tolerance=1e-4):
    loss = loss_fn()
    loss.backward()

    autodiff = parameter.grad
    numerical = numerical_gradient(
        loss_fn,
        parameter,
        epsilon
    )

    error = relative_error(autodiff, numerical)

    return error < tolerance

def numerical_gradient_at_index(
    loss_fn,
    parameter,
    index,
    epsilon=1e-3
):
    original = parameter.data[index]

    parameter.data[index] = original + epsilon
    loss_plus = loss_fn().data

    parameter.data[index] = original - epsilon
    loss_minus = loss_fn().data

    parameter.data[index] = original

    return (loss_plus - loss_minus) / (2 * epsilon)


def check_parameter_gradients(
    loss_fn,
    parameters,
    epsilon=1e-3,
    tolerance=1e-3
):
    loss = loss_fn()
    loss.backward()

    for parameter in parameters:

        for index in np.ndindex(parameter.data.shape):

            autodiff = parameter.grad[index]

            numerical = numerical_gradient_at_index(
                loss_fn,
                parameter,
                index,
                epsilon
            )

            error = relative_error(
                autodiff,
                numerical
            )

            if error >= tolerance:
                print(
                    "FAILED:",
                    index,
                    "autodiff =", autodiff,
                    "numerical =", numerical,
                    "error =", error
                )
                return False

    return True