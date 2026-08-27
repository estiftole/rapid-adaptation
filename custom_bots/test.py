import xml.etree.ElementTree as ET
from xml.dom import minidom
import mujoco
import mujoco.viewer

def create_arm_xml(link_length=0.5, link_radius=0.04):
    mujoco_elem = ET.Element("mujoco", model="custom_arm")

    # Options
    ET.SubElement(mujoco_elem, "option", gravity="0 0 -9.81")

    # Worldbody
    worldbody = ET.SubElement(mujoco_elem, "worldbody")

    # Ground Plane
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.1",
        rgba="0.8 0.8 0.8 1",
    )

    # LINK 1 (BLUE)
    base = ET.SubElement(worldbody, "body", name="base_link", pos="0 0 0.1")

    # Shoulder Joint Marker
    ET.SubElement(
        base,
        "geom",
        name="shoulder_joint_viz",
        type="sphere",
        size="0.06",
        rgba="0.1 0.1 0.1 1",
    )
    ET.SubElement(base, "joint", name="shoulder", type="hinge", axis="0 1 0")

    # Link 1 Capsule
    ET.SubElement(
        base,
        "geom",
        name="base_geom",
        type="capsule",
        size=f"{link_radius} {link_length/2}",
        pos=f"0 0 {link_length/2}",
        rgba="0.2 0.4 0.9 1",  # Blue
    )

    # LINK 2 (RED) - Positioned at the top tip of Link 1
    link2 = ET.SubElement(
        base, "body", name="child_link", pos=f"0 0 {link_length}"
    )

    # Elbow Joint Marker
    ET.SubElement(
        link2,
        "geom",
        name="elbow_joint_viz",
        type="sphere",
        size="0.05",
        rgba="0.1 0.1 0.1 1",
    )

    ET.SubElement(
        link2,
        "joint",
        name="elbow",
        type="hinge",
        axis="0 1 0",
        ref="45",  # Start bent at 45 degrees so it doesn't look continuous
    )

    # Link 2 Capsule
    ET.SubElement(
        link2,
        "geom",
        name="child_geom",
        type="capsule",
        size=f"{link_radius*0.8} {link_length/2}",
        pos=f"0 0 {link_length/2}",
        rgba="0.9 0.2 0.2 1",  # Red
    )

    # Actuators to control movement
    actuator = ET.SubElement(mujoco_elem, "actuator")
    ET.SubElement(
        actuator, "position", joint="shoulder", name="p_shoulder", kp="100"
    )
    ET.SubElement(actuator, "position", joint="elbow", name="p_elbow", kp="100")

    raw_string = ET.tostring(mujoco_elem, encoding="utf-8")
    return minidom.parseString(raw_string).toprettyxml(indent="  ")


# Render the clear 2-link robot
xml_str = create_arm_xml()
model = mujoco.MjModel.from_xml_string(xml_str)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
