import os
import tempfile
import xml.etree.ElementTree as ET
import gymnasium as gym
import time
import numpy as np

class ProceduralPusherEnv(gym.Env):
    def __init__(self, randomize_freq=10, render_mode=None):
        self.randomize_freq = randomize_freq
        self.render_mode = render_mode
        self.episode_counter = 0


        self.base_xml_path = os.path.join(
            os.path.dirname(gym.__file__),
            "envs", "mujoco", "assets", "pusher.xml"
        )

        self.current_xml_file = None
        self.env = None


        self.arm_scale_range = (0.1, 1)
        self.true_forearm_scale = 1.0


        self._rebuild_environment()


        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _rebuild_environment(self):
        """Parses the XML, applies kinematic scaling, and re-initializes MuJoCo."""
        if self.env is not None:
            self.env.close()


        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()


        upper_arm = root.find(".//body[@name='r_forearm_link']")
        if upper_arm is not None:
            geom = upper_arm.find("geom")
            delta = 0
            start_coords, end_coords = np.zeros(3), np.zeros(3)
            if geom is not None and "fromto" in geom.attrib:
                coords = np.array([float(x) for x in geom.attrib["fromto"].split()])
                start_coords, end_coords = coords[:3].copy(), coords[3:].copy()
                delta = (end_coords - start_coords) * self.true_forearm_scale
                coords[3:] = start_coords + delta
                geom.attrib["fromto"] = " ".join(map(lambda x: f"{x:.4f}", coords))


            # for child in upper_arm.iter():
            #     if "pos" in child.attrib:
            #         pos = np.array([float(x) for x in child.attrib["pos"].split()])
            #         pos = pos * self.true_forearm_scale
            #         child.attrib["pos"] = " ".join(map(lambda x: f"{x:.4f}", pos))
            #
            # for child in upper_arm.iter():
                # print(child)
            child = root.find(".//body[@name='r_wrist_flex_link']")
            if child is not None and "pos" in child.attrib:
                pos = np.array([float(x) for x in child.attrib["pos"].split()])

                shift = pos-end_coords
                print('shift', shift)
                pos = start_coords + delta + shift
                child.attrib["pos"] = " ".join(map(lambda x: f"{x:.4f}", pos))


        fd, temp_path = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        tree.write(temp_path)


        if self.current_xml_file and os.path.exists(self.current_xml_file):
            os.remove(self.current_xml_file)

        self.current_xml_file = temp_path


        self.env = gym.make("Pusher-v5", xml_file=self.current_xml_file, render_mode=self.render_mode)

    def reset(self, seed=None, options=None):

        if self.episode_counter % self.randomize_freq == 0:
            if seed is not None:
                np.random.seed(seed)


            self.true_forearm_scale = np.random.uniform(*self.arm_scale_range)
            self._rebuild_environment()

        self.episode_counter += 1

        obs, info = self.env.reset(seed=seed, options=options)


        info["true_forearm_scale"] = self.true_forearm_scale
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        if self.env:
            self.env.close()
        if self.current_xml_file and os.path.exists(self.current_xml_file):
            os.remove(self.current_xml_file)

if __name__ == "__main__":

    env = ProceduralPusherEnv(randomize_freq=1, render_mode="human")

    print("Initializing Procedural Pusher. Watch the arm length change on resets.")

    obs, info = env.reset()
    print(f"Arm Scale: {info['true_forearm_scale']:.2f}")


    for _ in range(100):
        action = env.action_space.sample()
        # action = env.action_space
        env.step(action)
        time.sleep(0.1)

    env.close()
