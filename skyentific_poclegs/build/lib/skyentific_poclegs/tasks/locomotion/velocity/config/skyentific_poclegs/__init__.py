# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Velocity-Rough-Skyentific-Poclegs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.SkyentificPoclegsRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SkyentificPoclegsRoughPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-Skyentific-Poclegs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.SkyentificPoclegsRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SkyentificPoclegsRoughPPORunnerCfg",
    },
)
