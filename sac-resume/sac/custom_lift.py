import numpy as np

from robosuite.environments.manipulation.lift import Lift


class CustomLift(Lift):
    def __init__(self, *args, pbrs_gamma=0.99, **kwargs):
        self.pbrs_gamma = pbrs_gamma
        self.last_shaping = 0.0
        self._prev_phi = 0.0
        super().__init__(*args, **kwargs)

    def reset(self):
        obs = super().reset()
        self._prev_phi = self._potential()
        self.last_shaping = 0.0
        return obs

    def _potential(self):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        phi = 1.0 - np.tanh(10.0 * np.linalg.norm(eef_pos - cube_pos))
        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
            z_rest = self.model.mujoco_arena.table_offset[2] + 0.02
            phi += 0.25 + np.clip((cube_pos[2] - z_rest) / 0.02, 0.0, 1.0)
        return phi

    def reward(self, action=None):
        sparse = super().reward(action)
        phi = self._potential()
        self.last_shaping = self.pbrs_gamma * phi - self._prev_phi
        self._prev_phi = phi
        return sparse + self.last_shaping