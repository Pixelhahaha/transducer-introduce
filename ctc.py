import torch
import numpy as np
import math

NEG_INF = -float("inf")

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                    for a in args))
    return a_max + lsp



class CTC:
    def __init__(self):
        pass

    def forward(self):
        pass

    def alpha(self, log_y, labels):
        T, V = log_y.shape
        L = len(labels)
        log_alpha = np.ones([T, L]) * NEG_INF

        # init
        log_alpha[0, 0] = log_y[0, labels[0]]
        log_alpha[0, 1] = log_y[0, labels[1]]

        for t in range(1, T):
            for i in range(L):
                s = labels[i]

                a = log_alpha[t - 1, i]
                if i - 1 >= 0:
                    a = logsumexp(a, log_alpha[t - 1, i - 1])
                if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                    a = logsumexp(a, log_alpha[t - 1, i - 2])

                log_alpha[t, i] = a + log_y[t, s]

        return log_alpha


    def beta(self, log_y, labels):
        T, V = log_y.shape
        L = len(labels)
        log_beta = np.ones([T, L]) * NEG_INF

        # init
        log_beta[-1, -1] = log_y[-1, labels[-1]]
        log_beta[-1, -2] = log_y[-1, labels[-2]]

        for t in range(T - 2, -1, -1):
            for i in range(L):
                s = labels[i]

                a = log_beta[t + 1, i]
                if i + 1 < L:
                    a = logsumexp(a, log_beta[t + 1, i + 1])
                if i + 2 < L and s != 0 and s != labels[i + 2]:
                    a = logsumexp(a, log_beta[t + 1, i + 2])

                log_beta[t, i] = a + log_y[t, s]

        return log_beta

    def backward(self, log_y, labels):
        T, V = log_y.shape
        L = len(labels)

        log_alpha = self.alpha(log_y, labels)
        log_beta = self.beta(log_y, labels)
        log_p = logsumexp(log_alpha[-1, -1], log_alpha[-1, -2])

        log_grad = np.ones([T, V]) * NEG_INF
        for t in range(T):
            for s in range(V):
                lab = [i for i, c in enumerate(labels) if c == s]
                for i in lab:
                    log_grad[t, s] = logsumexp(log_grad[t, s],
                                            log_alpha[t, i] + log_beta[t, i])
                log_grad[t, s] -= 2 * log_y[t, s]

        log_grad -= log_p
        return log_grad

torch_ctc_loss = torch.nn.CTCLoss(reduction="sum", zero_infinity=True)

#prob = torch.randn(1, 4, 10)
prob = torch.tensor([[[-0.6129,  2.1921,  1.2780, -0.4573, -0.0031,  2.0034,  0.4039,
                        0.4337, -0.6928,  0.0068],
                     [ 0.9346, -2.1171,  1.1010, -0.1307,  0.2094, -1.6167,  0.2946,
                        0.1866,  0.6334, -0.6826],
                     [-2.3595,  0.4712,  0.6167, -1.0870,  1.1760,  0.8188, -0.5910,
                        -0.6670,  2.2992,  0.6582],
                     [-1.3807, -0.4616,  0.0111,  1.1241,  0.1621, -1.9554,  0.8082,
                        0.1617,  0.3362, -0.3407]]])

log_prob = prob.log_softmax(2)
targets = torch.tensor([[3]])
targets_blank = torch.tensor([0, 3, 0])

input_lengths = torch.tensor([4])
target_lengths = torch.tensor([1])
loss = torch_ctc_loss(log_prob.transpose(0,1), targets, input_lengths, target_lengths)


## my ctc
ctc_loss_ = CTC()
alpha = ctc_loss_.alpha(log_prob[0], targets_blank)
alpha_loss = -logsumexp(alpha[-1, -1], alpha[-1, -2])

beta = ctc_loss_.beta(log_prob[0], targets_blank)
beta_loss = -logsumexp(beta[0, 0], beta[0, 1])
grad_ = ctc_loss_.backward(log_prob.transpose(0, 1)[0], torch.tensor([0, 3, 0]))