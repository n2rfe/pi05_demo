#!/usr/bin/env python3
"""
pi05_fr3_bridge — Bridge node connecting pi0.5 TensorRT inference to Fairino FR3.

Pipeline:
  /camera/image_raw  ──┐
                        ├──► pi0.5 TensorRT engine ──► FR3 joint commands
  FR3 joint states   ──┘

Joint mapping (FR3 is 6-DOF, pi05_droid expects 7-DOF Franka + gripper):
  - FR3 joints 0-5 map to action dims 0-5
  - Action dim 6 (Franka joint 7) is ignored
  - Action dim 7 (gripper) is passed to FR3 end effector if available
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
import numpy as np
import sys
import os
import threading
import time

# FR3 Python SDK (pure Python XML-RPC)
sys.path.insert(0, '/home/efren/pi05_demo')
import Robot as FairinRobot

# pi0.5 TensorRT inference
sys.path.insert(0, '/home/efren/openpi/packages/openpi-client/src')
sys.path.insert(0, '/home/efren/openpi/src')

ENGINE_PATH = os.path.expanduser(
    '~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/onnx/onnx/model_fp8_nvfp4.engine'
)
CHECKPOINT_DIR = os.path.expanduser(
    '~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch'
)
ROBOT_IP = '192.168.58.2'

# Control parameters
JOINT_SPEED    = 20.0   # % of max speed — conservative for demo
JOINT_ACC      = 20.0   # % of max acceleration
ACTION_SCALE   = 30.0   # degrees per action unit — tune after first test
CONTROL_HZ     = 10     # how many steps per second to execute
NUM_JOINTS     = 6      # FR3 DOF


class Pi05FR3Bridge(Node):
    def __init__(self):
        super().__init__('pi05_fr3_bridge')
        self.get_logger().info('Initializing pi05 FR3 Bridge...')

        # --- Connect to FR3 ---
        self.get_logger().info(f'Connecting to FR3 at {ROBOT_IP}...')
        self.robot = FairinRobot.RPC(ROBOT_IP)
        ret, joints = self.robot.GetActualJointPosDegree()
        if ret != 0:
            self.get_logger().error(f'FR3 connection failed, code: {ret}')
            raise RuntimeError('Cannot connect to FR3')
        self.get_logger().info(f'FR3 connected. Joints: {[round(j,2) for j in joints[:6]]}')
        self.current_joints = np.array(joints[:6], dtype=np.float32)

        # --- Load pi0.5 TensorRT engine ---
        self.get_logger().info('Loading pi0.5 TensorRT engine...')
        self._load_policy()
        self.get_logger().info('TensorRT engine ready.')

        # --- State ---
        self.latest_image = None
        self.prompt = 'pick up the object'
        self.running = False
        self.image_lock = threading.Lock()

        # --- ROS2 subscribers ---
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.get_logger().info('Subscribed to /camera/image_raw')
        self.get_logger().info('Bridge ready. Call start_control() to begin.')

    def _load_policy(self):
        """Load pi0.5 with TensorRT engine inside Docker via subprocess,
        or directly if running inside the container."""
        try:
            # Apply transformers patches required by pi0.5
            import shutil
            src = '/home/efren/openpi/src/openpi/models_pytorch/transformers_replace'
            dst_base = None
            # Find transformers install location
            import transformers
            dst_base = os.path.dirname(transformers.__file__)
            if os.path.exists(src):
                for item in os.listdir(src):
                    s = os.path.join(src, item)
                    d = os.path.join(dst_base, item)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)

            import openpi_client
            from openpi_client import inference as trt_inference
            self.policy = trt_inference.TRTPolicy(
                engine_path=ENGINE_PATH,
                checkpoint_dir=CHECKPOINT_DIR,
                config_name='pi05_droid',
            )
        except Exception as e:
            self.get_logger().warn(f'Direct policy load failed: {e}')
            self.get_logger().warn('Will use HTTP inference server instead.')
            self.policy = None

    def image_callback(self, msg: Image):
        """Store latest camera frame."""
        img = np.frombuffer(msg.data, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, -1))
        # Resize to 224x224 if needed
        if msg.height != 224 or msg.width != 224:
            import cv2
            img = cv2.resize(img, (224, 224))
        # Ensure RGB
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1].copy()
        with self.image_lock:
            self.latest_image = img

    def set_prompt(self, text: str):
        self.prompt = text
        self.get_logger().info(f'Prompt set: "{text}"')

    def get_observation(self):
        """Build observation dict in pi05_droid format."""
        ret, joints = self.robot.GetActualJointPosDegree()
        if ret == 0:
            self.current_joints = np.array(joints[:6], dtype=np.float32)

        with self.image_lock:
            image = self.latest_image.copy() if self.latest_image is not None \
                    else np.zeros((224, 224, 3), dtype=np.uint8)

        # pi05_droid expects 8-dim state: 7 joint angles + 1 gripper
        # We pad FR3's 6 joints with a zero for the missing 7th joint
        state = np.zeros(8, dtype=np.float32)
        state[:6] = np.deg2rad(self.current_joints)  # convert to radians
        # state[6] = 0.0  # unused 7th joint
        # state[7] = 0.0  # gripper state (unknown without sensor)

        return {
            'observation/image':       image,
            'observation/wrist_image': image,   # use same image if no wrist cam
            'observation/state':       state,
            'prompt':                  self.prompt,
        }

    def execute_action_chunk(self, actions: np.ndarray):
        """
        Execute a chunk of actions on the FR3.
        actions: shape (15, 8) — 15 steps, 8 dims (7 joints + gripper)
        We use only dims 0-5 for FR3's 6 joints.
        """
        step_period = 1.0 / CONTROL_HZ

        for step_idx in range(actions.shape[0]):
            if not self.running:
                break

            action = actions[step_idx]  # shape (8,)

            # Convert action deltas to target joint positions in degrees
            delta_deg = np.rad2deg(action[:6]) * ACTION_SCALE / 30.0
            target_joints = self.current_joints + delta_deg

            # Safety clamp — FR3 joint limits (conservative)
            joint_limits = np.array([170, 130, 150, 160, 160, 160], dtype=np.float32)
            target_joints = np.clip(target_joints, -joint_limits, joint_limits)

            self.get_logger().debug(
                f'Step {step_idx}: target={[round(j,1) for j in target_joints]}'
            )

            # Send joint move command to FR3
            ret = self.robot.MoveJ(
                list(target_joints) + [0.0],  # 6 joints + tool angle placeholder
                tool=0, user=0,
                vel=JOINT_SPEED, acc=JOINT_ACC,
                ovl=100.0, exaxis_pos=[0,0,0,0],
                blendT=-1, offset_flag=0, offset_pos=[0,0,0,0,0,0]
            )
            if ret != 0:
                self.get_logger().warn(f'MoveJ returned code {ret} at step {step_idx}')

            time.sleep(step_period)

    def run_once(self):
        """Run one inference + execution cycle."""
        obs = self.get_observation()

        if self.policy is not None:
            actions = self.policy.infer(obs)  # shape (15, 8)
        else:
            self.get_logger().warn('No policy loaded — skipping inference.')
            return

        self.get_logger().info(
            f'Inference complete. Actions range: [{actions.min():.3f}, {actions.max():.3f}]'
        )
        self.execute_action_chunk(actions)

    def start_control(self, prompt: str = None):
        """Start the control loop."""
        if prompt:
            self.set_prompt(prompt)
        self.running = True
        self.get_logger().info(f'Starting control loop. Prompt: "{self.prompt}"')
        while self.running and rclpy.ok():
            self.run_once()

    def stop_control(self):
        self.running = False
        self.robot.StopMotion()
        self.get_logger().info('Control stopped.')


def main(args=None):
    rclpy.init(args=args)
    node = Pi05FR3Bridge()

    # Spin in background thread so callbacks work
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # Wait for first camera frame
        node.get_logger().info('Waiting for camera image...')
        timeout = 10.0
        t0 = time.time()
        while node.latest_image is None and (time.time() - t0) < timeout:
            time.sleep(0.1)

        if node.latest_image is None:
            node.get_logger().warn('No camera image received. Using blank image.')

        # Start control — change prompt as needed
        node.start_control(prompt='pick up the object')

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        node.stop_control()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
