import math
import torch
import torch.distributions as D

"""
Normal Inverse Gamma Distribution for evidential uncertainty learning adapted for Torch:
    Deep Evidential Regression, Amini et. al.
        https://arxiv.org/pdf/1910.02600.pdf
        https://www.youtube.com/watch?v=toTcf7tZK8c
        https://github.com/aamini/evidential-deep-learning
"""

def NormalInvGamma(gamma, nu, alpha, beta):
    """
    Normal Inverse Gamma Distribution
    """
    assert torch.all(nu > 0.), 'nu must be more than zero'
    assert torch.all(alpha > 1.), 'alpha must be more than one'
    assert torch.all(beta > 0.), 'beta must be more than zero'

    InvGamma = D.transformed_distribution.TransformedDistribution(
        D.Gamma(alpha, beta),
        D.transforms.PowerTransform(torch.tensor(-1.).to(alpha.device))
    )

    var = InvGamma.rsample()
    mu = D.Normal(gamma, torch.sqrt(beta / (alpha - 1) / nu)).rsample()

    return D.Normal(mu, torch.sqrt(var))


def ShrunkenNormalInvGamma(gamma, nu, alpha, beta):
    """
    Normal Inverse Gamma Distribution
    """
    assert torch.all(alpha > 1.), 'alpha must be more than one'
    assert torch.all(beta > 0.), 'beta must be more than zero'

    var = beta / (alpha - 1)

    return D.Normal(gamma, torch.sqrt(var))


def NIG_uncertainty(gamma, nu, alpha, beta):
    """
    calculates the epistemic uncertainty of a distribution
    the value is effectively expectation of sigma square
    """
    assert torch.all(nu > 0.), 'nu must be more than zero'
    assert torch.all(alpha > 1.), 'alpha must be more than one'
    assert torch.all(beta > 0.), 'beta must be more than zero'

    return torch.sqrt(beta / (alpha - 1) / nu)


def NIG_NLL(label, gamma, nu, alpha, beta, reduce=True):
    """
    Negative Log Likelihood loss between label and predicted output
    """
    twoBlambda = 2 * beta * (1 + nu)

    nll = 0.5 * torch.log(math.pi / nu) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(nu * (label - gamma)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll


def NIG_reg(label, gamma, nu, alpha, beta, reduce=True):
    """
    Regularizer for for NIG distribution, scale the output of this by ~0.01
    """
    loss = torch.abs(gamma - label) * (2*nu + alpha)
    return torch.mean(loss) if reduce else loss
