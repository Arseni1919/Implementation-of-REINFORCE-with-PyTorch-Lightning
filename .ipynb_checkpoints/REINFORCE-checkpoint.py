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
        self.net.double()

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        # action = torch.multinomial(self.forward(state), 1)
        # return action

        # state = torch.from_numpy(state).float().unsqueeze(0)
        # probs = self.forward(Variable(state))

        tnsr = torch.tensor(state, dtype=torch.double)
        curr_out = self(tnsr)
        highest_prob_action = np.random.choice(self.n_actions, p=np.squeeze(curr_out.detach().numpy()))
        log_prob = torch.log(tnsr.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


class RLDataset(torch.utils.data.IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        model: model of the agent
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, model: Model, env, sample_size: int = 1) -> None:
        self.model = model
        self.env = env
        self.sample_size = sample_size

    def __iter__(self):  # -> Tuple:
        max_steps = 10000
        state = self.env.reset()
        log_probs = []
        rewards = []
        states = []
        actions = []

        for steps in range(max_steps):
            action, log_prob = self.model.select_action(state)
            new_state, reward, done, _ = self.env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = new_state
            if done:
                break
        yield rewards, log_probs, states, actions
        # # states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        # for i in range(len(dones)):
        #     # print(f'\n sample num {i}')
        #     yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class REINFORCELightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, env_name, ) -> None:
        super().__init__()
        # self.hparams = hparams

        self.env = gym.make(env_name)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = Model(obs_size, n_actions)
        # self.target_net = DQN(obs_size, n_actions)

        # self.buffer = ReplayBuffer(self.hparams.replay_size)
        # self.agent = Agent(self.env, self.buffer)

        self.total_reward = 0
        self.episode_reward = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output

    def training_step(self, batch, nb_batch): # : Tuple[torch.Tensor, torch.Tensor]
        """

        """
        # print(f'\n[------------] new batch: {nb_batch} ; inside batch: {len(batch[0])}')
        device = self.get_device(batch)
        # rewards, log_probs = batch
        rewards = batch[0]
        # log_probs = batch[1]
        states = batch[2]
        actions = batch[3]

        states = torch.tensor(states)
        probs = self.forward(Variable(states))
        log_probs = torch.log(probs.squeeze(0)[actions])


        discounted_rewards = []
        GAMMA = 0.9

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        # normalize discounted rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        # policy_network.optimizer.zero_grad()
        loss = torch.stack(policy_gradient).sum()
        # loss.backward()

        log = {'total_reward': np.sum(rewards)}
        #
        return OrderedDict({'loss': loss, 'log': log})

    def configure_optimizers(self):  # -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        learning_rate = 3e-4
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        return [optimizer]

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.net, self.env)
        dataloader = DataLoader(dataset=dataset)
        return dataloader

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

def main():
    env_id = 'CartPole-v1'

    model = REINFORCELightning(env_id)
    print(f'The model we created correspond to:\n{model}')

    trainer = pl.Trainer(
        # gpus=1,
        # distributed_backend='dp',
        max_epochs=1000,
        early_stop_callback=False,
        val_check_interval=10
    )

    trainer.fit(model)
    trainer.save_checkpoint("example.ckpt")

    new_model = REINFORCELightning.load_from_checkpoint(checkpoint_path="example.ckpt")
    env = gym.make(env_id)
    obs = env.reset()
    for i in range(30000):

        highest_prob_action, log_prob = REINFORCELightning.net.select_action(torch.tensor(obs))
        # _, action = torch.max(q_values, dim=1)
        # action = int(action.item())

        obs, rew, done, info = env.step(highest_prob_action)
        env.render()
        # print(rew)
        if done:
            obs = env.reset()
    env.close()


if __name__ == '__main__':
    main()
