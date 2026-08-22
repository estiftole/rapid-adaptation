import os
import gymnasium as gym

base_xml_path = os.path.join(
    os.path.dirname(gym.__file__),
    "envs", "mujoco", "assets", "pusher.xml"
)

print(base_xml_path)
