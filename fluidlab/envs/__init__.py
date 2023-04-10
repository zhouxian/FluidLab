import gym
from gym import register

for env_name in ['LatteArt', 'LatteArtStir', 'Scooping', 'GatheringEasy', 'GatheringO', 'IceCreamDynamic', 'IceCreamStatic', 'Transporting', 'Stabilizing', 'Pouring', 'Circulation', 'Mixing']:
    for id in range(1):
        register(
            id = f'{env_name}-v{id}',
            entry_point=f'fluidlab.envs.{env_name.lower()}_env:{env_name}Env',
            kwargs={'version': id, 'loss_type': 'default'},
            max_episode_steps=10000
        )