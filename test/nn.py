__author__ = 'root'

import theano
import theano.tensor as T
from theano.tensor import nnet
import numpy as np

def gradient_example():
    x = T.dscalar(name='x')
    f = T.exp(T.sin(T.sqr(x)))

    theano.function(inputs=[x], outputs=[f])

    print theano.pprint(T.grad(f, wrt=x))


def make_vectors_equal():
    # initialize a numpy array
    E = np.array(np.random.randn(6, 2), dtype=theano.config.floatX)
    # create a theano shared cause its gonna be modified
    t_E = theano.shared(E)
    print t_E.eval()
    # theano array of subindexes
    t_idxs = T.ivector()
    t_embedding_output = t_E[t_idxs]
    t_product = T.dot(t_embedding_output[0], t_embedding_output[1])
    t_label = T.scalar()
    grad = T.grad(cost=abs(t_label - t_product), wrt=t_E)
    updates = [(t_E, t_E - 0.01 * grad)]
    train = theano.function(inputs=[t_idxs, t_label], outputs=[], updates=updates)
    for i in range(0, 10000):
        v1, v2 = np.random.randint(0, 5), np.random.randint(0, 5)

        # by dividing the integer by 2, i only get the integer part.
        # it puts a 1 to pairs: 0-1, 2-3, 4-5 and 0 to other combinations
        label = 1.0 if (v1 / 2 == v2 / 2) else 0.0

        train([v1, v2], label)
        if i % 100 == 0:
            for n, embedding in enumerate(t_E.get_value()):
                print i, n, embedding[0], embedding[1]

def scan_power_example():
    k = T.iscalar("k")
    A = T.ivector("A")
    A0 = T.ivector("A")

    def multiply(acc,A):
        return acc*A

    # Initialization occurs in outputs_info
    # Unchanging variables are passed to scan as non_sequences
    results,updates = theano.scan(fn=multiply,
                outputs_info=[A0],
                non_sequences=[A],
                n_steps=k)

    result = results[-1]
    power = theano.function(inputs=[A,A0,k], outputs=[result], allow_input_downcast=True)
    k = 2

    A = np.random.randint(1,10,size=10)
    A0 = np.ones((10,), dtype=int)

    print(A)
    print(power(A,A0,k))

def scan_coeff_example():
    coefficients = theano.tensor.vector("coefficients")
    x = T.scalar("x")
    x0 = T.scalar("x0")

    max_coefficients_supported = 10000

    def func(coeff,pos,x):
        """
        doesnt get the accumulation. returns the vector
        """
        return coeff*(x**pos)

    def func_inone(coeff,pos,acc,x):
        """
        gets the accumulation from previous iteration
        """
        return acc + coeff*(x**pos)

    results, updates = theano.scan(fn=func,
                sequences=[coefficients, T.arange(max_coefficients_supported)],
                outputs_info=None, #This indicates to scan that it doesnt need to pass the prior result to fn
                non_sequences=x)
                # n_steps=)

    results_inone, _ = theano.scan(fn=func_inone,
                sequences=[coefficients, T.arange(max_coefficients_supported)],
                outputs_info=[x0],
                non_sequences=x)
                # n_steps=)

    result = results.sum()
    results_inone = results_inone[-1]
    compute_poly = theano.function(inputs=[coefficients,x],outputs=[result],updates=[])
    compute_poly_inone = theano.function(inputs=[coefficients,x0,x],outputs=[results_inone],updates=[])

    coefficients = [1,2,3,4]

    print(compute_poly(coefficients,2))
    print(compute_poly_inone(coefficients,0,2))

def scan_upto_example():
    up_to = T.iscalar("up_to")

    seq = T.arange(up_to)

    def accumulate(arange_val, sum_to_date):
        return sum_to_date + arange_val

    # i can determine the initial value, directly here in the symbolic expression
    outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
    results,updates = theano.scan(fn=accumulate,
                                  sequences=[seq],
                                  outputs_info=outputs_info)

    result = results[-1]
    sum_func = theano.function(inputs=[up_to], outputs=[result], updates=[])

    print(sum_func(2))


if __name__ == '__main__':
    # gradient_example()

    # make_vectors_equal()

    # scan_power_example()
    # scan_coeff_example()
    scan_upto_example()