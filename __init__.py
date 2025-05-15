from gymnasium.envs.registration import register

"""register(
    id="cantstop_env/GridWorld-v0",
    entry_point="cantstop_env.envs:GridWorldEnv",
)"""

register(id="cantstop_env/CantStop-v0",
         entry_point="cantstop_env.envs:CantStopEnv")
