import argparse
from utils.make_env import make_env
import torch
import numpy as np
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
from torch.autograd import Variable
import time

def obs_to_tensor(obs):
    obs_tensor_list = []
    # print("obs type---------", type(obs))
    # print("obs[0] type----------", type(obs[0]))
    for o in obs:
        # print("o_type--------------", type(o).shape)
        tensor_data = torch.from_numpy(o).float()
        tensor_data = tensor_data.unsqueeze(0)
        obs_tensor_list.append(tensor_data)
    
    return obs_tensor_list


def act_to_tensor(torch_agent_actions):
    actoin_array_list = []
    for a_tensor in torch_agent_actions:
        array_data = np.array(a_tensor.squeeze())
        actoin_array_list.append(array_data)
    #print("actoin_array_list-----", actoin_array_list)

    return actoin_array_list


def evaluate(config):
    #config.env_id = "multi_speaker_listener"  # treelet
    #config.env_id = "fullobs_collect_treasure"  # treelet

    seed=1
    config.n_rollout_threads=1
    #config.load_dir = "C:/Users/treelet/Downloads/MAAC-master/MAAC-master/models/multi_speaker_listener/listener_speaker_test/run2/model.pt"
    #config.load_dir = "C:/Users/treelet/Downloads/MAAC-master/MAAC-master/models/fullobs_collect_treasure/fullobs_collect_treasure_treelet_0616/run2/model.pt"

    #env = env = make_env(scenario_name="multi_speaker_listener", discrete_action=True)
    env = env = make_env(scenario_name=config.env_id, discrete_action=True)
    #model = AttentionSAC.init_from_save("C:/Users/treelet/Downloads/MAAC-master/MAAC-master/models/multi_speaker_listener/listener_speaker_test/run2/model.pt")  #
    model = AttentionSAC.init_from_save(config.load_dir)  #

    for ep_i in range(0, config.n_eval_episode, config.n_rollout_threads):
        episode_rewards = np.full(model.nagents, 0.0)
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_eval_episode))
        obs = env.reset()
        
        env.render()
        time.sleep(0.1)
        model.prep_rollouts(device='cpu')
        for et_i in range(config.episode_length):
            torch_obs = obs_to_tensor(obs)
            torch_agent_actions = model.step(torch_obs, explore=True)
            agent_actions = act_to_tensor(torch_agent_actions)
            # agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(agent_actions)
            episode_rewards += np.array(rewards)
            env.render()
            time.sleep(0.1)
            obs = next_obs
        time.sleep(0.5)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment")
    parser.add_argument("--load_dir", help="Name of directory to store model")
    parser.add_argument("--n_eval_episode", default=12, type=int,  help="The number of inference episodes")  # 
    parser.add_argument("--episode_length", default=25, type=int)
    config = parser.parse_args()

    evaluate(config)