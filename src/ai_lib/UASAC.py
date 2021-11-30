#!/usr/bin/env python3
import warnings
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as func

from ai_lib.UASACNet import *
from ai_lib.neural_blocks import *
from ai_lib.normal_inverse_gamma import *


class TwinnedQNetwork(nn.Module):
    """
    Twin Q Network
    """
    def __init__(self, num_actions):
        super().__init__()

        # critic, clipped double Q
        self.Q_network1 = Critic(num_actions)
        self.Q_network2 = Critic(num_actions)


    def forward(self, states, actions):
        """
        states is of shape ** x num_inputs
        actions is of shape ** x num_actions
        output is a tuple of [** x 1], [** x 1]
        """
        # get q1 and q2
        q1 = self.Q_network1(states, actions)
        q2 = self.Q_network2(states, actions)

        return q1, q2


class GaussianActor(nn.Module):
    """
    Gaussian Actor Wrapper for Deep Evidential Regression
    """
    def __init__(self, num_actions):
        super().__init__()
        self.net = Actor(num_actions)


    def forward(self, states):
        return self.net(states)


    @staticmethod
    def sample(gamma, nu, alpha, beta):
        """
        output:
            actions is of shape ** x num_actions
            entropies is of shape ** x 1
            log_probs is of shape ** x num_actions
        """
        output = gamma, nu, alpha, beta
        normals = ShrunkenNormalInvGamma(*output)

        # compute epistemic uncertainty
        uncertainty = NIG_uncertainty(*output)

        # sample from dist
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate entropies
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = log_probs.sum(dim=-1, keepdim=True)

        return actions, entropies, uncertainty

    @staticmethod
    def infer(gamma, nu, alpha, beta):
        return torch.tanh(gamma)



class UASAC(nn.Module):
    """
    Uncertainty Aware Actor Critic
    """
    def __init__(self, num_actions, entropy_tuning=True, target_entropy=None, confidence_scale=3., confidence_cutoff=0.2):
        super().__init__()

        self.num_actions = num_actions
        self.use_entropy = entropy_tuning
        self.confidence_scale = confidence_scale
        self.confidence_cutoff = confidence_cutoff

        # backbone
        self.backbone = Backbone()

        # actor head
        self.actor = GaussianActor(num_actions)

        # twin delayed Q networks
        self.critic = TwinnedQNetwork(num_actions)
        self.critic_target = TwinnedQNetwork(num_actions).eval()

        # copy weights and disable gradients for the target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        self.entropy_tuning = entropy_tuning
        if entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -float(num_actions)
            else:
                if target_entropy > 0.:
                    warnings.warn(f"Target entropy is recommended to be negative,\
                                  currently it is {target_entropy},\
                                  I hope you know what you're doing...")
                self.target_entropy = target_entropy
            self.log_alpha = nn.Parameter(torch.tensor(0., requires_grad=True))
        else:
            self.log_alpha = nn.Parameter(torch.tensor(0., requires_grad=True))


    def update_q_target(self, tau=0.1):
        # polyak averaging update for target q network
        for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)


    def calc_critic_loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        """
        states is of shape B x img_size
        actions is of shape B x 3
        rewards is of shape B x 1
        dones is of shape B x 1
        """
        dones = 1. - dones

        # current Q
        curr_q1, curr_q2 = self.critic(states, actions)

        # target Q
        with torch.no_grad():
            # sample the next actions based on the current policy
            output = self.actor(next_states)
            next_actions, _, _ = self.actor.sample(*output)

            next_q1, next_q2 = self.critic_target(next_states, next_actions)

            # concatenate both qs together then...
            next_q  = torch.cat((next_q1, next_q2), dim=-1)

            # ...take the min at the cat dimension
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)

            # TD learning, targetQ = R + gamma*nextQ*done
            target_q = rewards + dones * gamma * next_q

        # critic loss is mean squared TD errors
        q1_loss = func.mse_loss(curr_q1, target_q, reduction='none')
        q2_loss = func.mse_loss(curr_q2, target_q, reduction='none')
        q_loss = q1_loss + q2_loss

        # NIG regularizer scale
        reg_scale = q_loss.abs().detach()

        return (q_loss.mean() / 2.), reg_scale


    def calc_actor_loss(self, states, dones, labels):
        """
        states is of shape B x img_size
        dones is of shape B x 1
        labels is of shape B x 3
        """
        dones = 1. - dones

        # We re-sample actions to calculate expectations of Q.
        output = self.actor(states)
        actions, entropies, uncertainty = self.actor.sample(*output)

        # expectations of Q with clipped double Q
        q1, q2 = self.critic(states, actions)
        q, _ = torch.min(torch.cat((q1, q2), dim=-1), dim=-1, keepdim=True)

        # reinforcement target is maximization of (Q + alpha * entropy) * done
        rnf_loss = 0.
        if self.use_entropy:
            rnf_loss = -((q - self.log_alpha.exp().detach() * entropies) * dones) + 1e-6
        else:
            rnf_loss = -(q * dones) + 1e-6

        # supervised loss is NLL loss between label and output
        sup_loss = NIG_NLL(torch.atanh(labels), *output, reduce=False) + 1e-6

        # supervision scale
        sup_scale = (1. - torch.exp(-self.confidence_scale * uncertainty)).detach()

        # cutoff for supervision scale
        blank = torch.zeros_like(sup_scale)
        sup_scale = torch.where(sup_scale > self.confidence_cutoff, sup_scale, blank)

        # NIG regularizer scale
        output = self.actor(states)
        reg_loss = 2*output[1] + output[2]

        return rnf_loss, sup_loss, sup_scale.detach(), reg_loss


    def calc_alpha_loss(self, states):
        if not self.entropy_tuning:
            return torch.zeros(1)

        output = self.actor(states)
        _, entropies, _ = self.actor.sample(*output)

        # Intuitively, we increse alpha when entropy is less than target entropy, vice versa.
        entropy_loss = (self.log_alpha * (self.target_entropy - entropies).detach()).mean()

        return entropy_loss


