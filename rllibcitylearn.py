import numpy as np
from marllib.envs.base_env import ENV_REGISTRY
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib import marl
from citylearn.citylearn import CityLearnEnv

# register all scenario with env class
REGISTRY = {}
REGISTRY["fontana"] = CityLearnEnv

policy_mapping_dict = {
    "fontana": {
        "description": "CityLearn fontana",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RllibCityLearnEnv(MultiAgentEnv):
    """Wrapper for CityLearn to make it compatible with rllib"""
    
    def __init__(self, env_config: dict):
        self.env = CityLearnEnv(
            schema=env_config["schema"], # type: ignore
            root_directory=env_config.get("root_directory"), # type: ignore
            buildings=env_config.get("buildings"), # type: ignore
            simulation_start_time_step=env_config.get("simulation_start_time_step"), # type: ignore
            simulation_end_time_step=env_config.get("simulation_end_time_step"), # type: ignore
            episode_time_steps=env_config.get("episode_time_steps"), # type: ignore
            rolling_episode_split=env_config.get("rolling_episode_split"), # type: ignore
            random_episode_split=env_config.get("random_episode_split"), # type: ignore
            seconds_per_time_step=env_config.get("seconds_per_time_step"), # type: ignore
            reward_function=env_config.get("reward_function"), # type: ignore
            central_agent=env_config["central_agent"],
            shared_observations=env_config.get("shared_observations"), # type: ignore
            random_seed=env_config.get("random_seed") # type: ignore
            )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_agents = env_config["num_agents"]
        self.episode_limit = env_config["episode_limit"]
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.env_config = env_config
        
        
    def step(self, actions):
        """ Returns reward, terminated, info """
        action_ls = []
        for i, agent in enumerate(self.agents):
            action_ls.append(actions[agent])
        o, r, d, info = self.env.step(action_ls)
        rewards = {}
        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.array(o[i])
            }
            rewards[agent] = r
        dones = d
        return obs, rewards, dones, info

    def reset(self):
        """ Returns initial observations and states"""
        o = self.env.reset()
        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.array(o[i])
            }
        return obs

    def render(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.episode_limit,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

if __name__ == '__main__':
    # register CityLearn
    ENV_REGISTRY["citylearn"] = RllibCityLearnEnv
    # initialize env
    env = marl.make_env(environment_name="citylearn", map_name="fontana", abs_path="../../citylearn.yaml")
    # pick algorithm
    mappo = marl.algos.mappo(hyperparam_source="common")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
    # start learning 
    mappo.fit(env, model, stop={'timesteps_total': 10000000}, local_mode=False, num_gpus=1, share_policy='all', checkpoint_freq=500)