#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as func

from ai_lib.neural_blocks import *
from ai_lib.autoencoder import *


class TwinnedQNetwork(nn.Module):
    """
    Twin Q Network
    """
    def __init__(self, num_inputs, num_actions):
        super().__init__()

        # critic, clipped double Q
        features = [num_inputs + num_actions, 64, 64, 1]
        actions = ['lrelu', 'lrelu', 'identity']
        self.Q_network1 = Neural_blocks.generate_linear_stack(features, actions, batch_norm=False)
        self.Q_network2 = Neural_blocks.generate_linear_stack(features, actions, batch_norm=False)


    def forward(self, states, actions):
        """
        states is of shape ** x num_inputs
        actions is of shape ** x num_actions
        output is a tuple of [** x 1], [** x 1]
        """
        x = torch.cat((states, actions), dim=-1)

        # get q1 and q2
        q1 = self.Q_network1(x)
        q2 = self.Q_network2(x)

        return q1, q2


class GaussianActor(nn.Module):
    """
    Actor module
    """
    def __init__(self, num_inputs, num_actions, log_std_min=-20., log_std_max=2.):
        super().__init__()

        self.num_actions = num_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # actor, outputs means and log std of action along a linear space
        features = [num_inputs, 64, 64, 2*num_actions]
        actions = ['lrelu', 'lrelu', 'identity']
        self.actor = Neural_blocks.generate_linear_stack(features, actions, batch_norm=False)


    def forward(self, input):
        """
        input is of shape ** x data_size
        output:
            actions is of shape ** x num_actions
            entropies is of shape ** x 1
            log_probs is of shape ** x num_actions
        """
        action = self.actor(input)

        # if we're not training, just return the actions
        if not self.training:
            return torch.tanh(action[..., :self.num_actions]), 0.

        log_std = action[..., self.num_actions:].clamp(self.log_std_min, self.log_std_max)
        normals = dist.Normal(action[..., :self.num_actions], log_std.exp())

        # sample actions
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate entropies
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = log_probs.sum(dim=-1, keepdim=True)

        return actions, entropies



class SoftActorCritic(nn.Module):
    """
    Actor critic model
    """
    def __init__(self, num_inputs, num_actions, entropy_tuning=True, target_entropy=None):
        super().__init__()

        self.num_actions = num_actions

        # autoencoder
        self.backbone = autoencoder()

        # actor network
        self.actor = GaussianActor(num_inputs, num_actions)

        # twin delayed Q networks
        self.critic = TwinnedQNetwork(num_inputs, num_actions)
        self.critic_target = TwinnedQNetwork(num_inputs, num_actions).eval()

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
                assert target_entropy < 0, 'target entropy must be negative'
                self.target_entropy = target_entropy
            self.log_alpha = nn.Parameter(torch.tensor(0., requires_grad=True))
        else:
            self.log_alpha = nn.Parameter(torch.tensor(0., requires_grad=True))


    def update_q_target(self, tau=0.1):
        # polyak averaging update for target q network
        for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)


    def calc_critic_loss(self, states, next_states, actions, next_actions, rewards, done, gamma=0.7):
        """
        states is of shape B x 64
        actions is of shape B x 3
        rewards is of shape B x 1
        dones is of shape B x 1
        """
        # current Q
        curr_q1, curr_q2 = self.critic(states, actions)

        # target Q
        with torch.no_grad():
            next_q1, next_q2 = self.critic_target(next_states, next_actions)

            # concatenate both qs together then...
            next_q  = torch.cat((next_q1, next_q2), dim=-1)

            # ...take the min at the cat dimension
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)

            # TD learning, targetQ = R + gamma*nextQ*done
            # intuitively this is a slightly less optimal loss function
            # incorporating entropy into the q value will make the agent
            # explore areas which are more robust to noise in the actions
            target_q = rewards + done * gamma * next_q

        # critic loss is mean squared TD errors
        q1_loss = func.smooth_l1_loss(curr_q1, target_q)
        q2_loss = func.smooth_l1_loss(curr_q2, target_q)

        return (q1_loss + q2_loss) / 2.


    def calc_actor_loss(self, states, done, use_entropy=True):
        """
        states is of shape B x 64
        """
        # We re-sample actions to calculate expectations of Q.
        actions, entropies = self.actor(states)

        # expectations of Q with clipped double Q
        q1, q2 = self.critic(states, actions)
        q, _ = torch.min(torch.cat((q1, q2), dim=-1), dim=-1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) * done
        if use_entropy:
            actor_loss = torch.mean((q - self.log_alpha.exp().detach() * entropies) * done)
        else:
            actor_loss = torch.mean(q * done)

        return -actor_loss


    def calc_alpha_loss(self, states):
        if not self.entropy_tuning:
            return torch.zeros(1)

        _, entropies = self.actor.forward(states)

        # Intuitively, we increse alpha when entropy is less than target entropy, vice versa.
        entropy_loss = (self.log_alpha * (self.target_entropy - entropies).detach()).mean()

        return entropy_loss


