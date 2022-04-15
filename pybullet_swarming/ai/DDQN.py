#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from ai_lib.neural_blocks import *
from ai_lib.autoencoder import *


class TwinnedQNetwork(nn.Module):
    """
    Twin Q Network
    """

    def __init__(self, num_inputs, num_actions):
        super().__init__()

        # critic, clipped double Q
        features = [num_inputs, 64, 64, num_actions]
        actions = ["lrelu", "lrelu", "identity"]
        self.Q_network1 = Neural_blocks.generate_linear_stack(
            features, actions, batch_norm=False
        )
        self.Q_network2 = Neural_blocks.generate_linear_stack(
            features, actions, batch_norm=False
        )

    def forward(self, states):
        """
        states is of shape ** x num_actions
        output is a tuple of [** x num_actions], [** x num_actions]
        """
        # get q1 and q2
        q1 = self.Q_network1(states)
        q2 = self.Q_network2(states)

        return q1, q2


class DoubleDeepQNetwork(nn.Module):
    """
    Double Deep Q Network
    """

    def __init__(self, num_inputs, num_actions):
        super().__init__()

        self.backbone = autoencoder()

        self.num_actions = num_actions

        # twin delayed Q networks
        self.q = TwinnedQNetwork(num_inputs, num_actions)
        self.q_target = TwinnedQNetwork(num_inputs, num_actions).eval()

        # copy weights and disable gradients for the target network
        self.q_target.load_state_dict(self.q.state_dict())
        for param in self.q_target.parameters():
            param.requires_grad = False

        map = [
            [+0.0, +0.8, +0.0],
            [+0.0, +0.0, +0.8],
            [-1.0, +0.0, +0.0],
            [+1.0, +0.0, +0.0],
            [+0.0, +0.0, +0.0],
        ]
        map = torch.tensor(map)
        self.register_buffer("map", map, persistent=False)

    def map_actions(self, actions):
        return actions.matmul(self.map)

    def update_q_target(self, tau=0.1):
        # polyak averaging update for target q network
        for target, source in zip(self.q_target.parameters(), self.q.parameters()):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def calc_critic_loss(
        self, states, next_states, actions, next_actions, rewards, done, gamma=0.9
    ):
        """
        states is of shape B x 64
        actions is of shape B x 4
        rewards is of shape B x 1
        dones is of shape B x 1
        """
        # current Q, gather the ones only where actions were taken
        curr_q1, curr_q2 = self.q(states)
        curr_q1 = curr_q1.gather(-1, actions.argmax(dim=-1, keepdim=True))
        curr_q2 = curr_q2.gather(-1, actions.argmax(dim=-1, keepdim=True))

        # target Q
        with torch.no_grad():
            next_q1, next_q2 = self.q_target(next_states)

            # take the max along the action dimensions
            next_q1 = next_q1.max(dim=-1)[0]
            next_q2 = next_q2.max(dim=-1)[0]

            # cat both qs together then...
            next_q = torch.stack((next_q1, next_q2), dim=-1)

            # ...take the min at the stack dimension
            next_q = next_q.min(dim=-1, keepdim=True)[0]

            # TD learning, targetQ = R + gamma*max(nextQ)*done
            target_q = rewards + done * gamma * next_q

        # critic loss is mean squared TD errors
        q1_loss = F.smooth_l1_loss(curr_q1, target_q)
        q2_loss = F.smooth_l1_loss(curr_q2, target_q)

        return (q1_loss + q2_loss) / 2.0

    def sample(self, states):
        """
        The sampling operation
        """
        q1, q2 = self.q(states)

        q = torch.stack((q1, q2), dim=-1).mean(dim=-1)
        a = D.categorical.Categorical(q.softmax(dim=-1)).sample()
        a = F.one_hot(a, num_classes=self.num_actions).float()

        return a

    def exploit(self, states):
        """
        The sampling operation
        """
        q1, q2 = self.q(states)

        q = torch.stack((q1, q2), dim=-1).mean(dim=-1)
        a = q.argmax(dim=-1)
        a = F.one_hot(a, num_classes=self.num_actions).float()

        return a
