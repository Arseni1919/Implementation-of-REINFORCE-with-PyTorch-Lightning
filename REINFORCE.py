from IMPORTS import *


class Model(nn.Module):
    def __init__(self, dim_observation, n_actions):
        super(Model, self).__init__()

        self.n_actions = n_actions
        self.dim_observation = dim_observation

        self.net = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.n_actions),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        action = torch.multinomial(self.forward(state), 1)
        return action


def main():
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    model = Model(env.observation_space.shape[0], env.action_space.n)
    print(f'The model we created correspond to:\n{model}')


if __name__ == '__main__':
    main()
