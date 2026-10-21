#!/usr/bin/env python3
"""
demo_runner.py — pi0.5 + FR3 Demo Runner
- Natural language prompt input at startup
- Grasp detection via gripper position (stops when object blocked)
- Live camera view saved to disk for VS Code
- Clean startup, execution, and shutdown sequence
"""

import sys, os, time, shutil, subprocess, threading
sys.path.insert(0, '/pi05_demo')
sys.path.insert(0, '/workspace/packages/openpi-client/src')
sys.path.insert(0, '/workspace/src')
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/openpi_on_thor')

import numpy as np
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
ROBOT_IP         = '192.168.58.2'
CAMERA_INDEX     = 0
JOINT_SPEED      = 15.0
JOINT_ACC        = 15.0
ACTION_SCALE     = 0.1
CONTROL_HZ       = 2
CHECKPOINT_DIR   = '/root/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch'
ENGINE_PATH      = CHECKPOINT_DIR + '/onnx/onnx/model_fp8_nvfp4.engine'
CONFIG_NAME      = 'pi05_droid'
CAMERA_SAVE_PATH = '/pi05_demo/camera_view.jpg'

# Gripper — DH Robotics AG-105-145-W-S
GRIPPER_INDEX          = 1
GRIPPER_COMPANY        = 4
GRIPPER_DEVICE         = 2
GRIPPER_VEL            = 50
GRIPPER_FORCE          = 50
GRIPPER_MAXTIME        = 3000
GRIPPER_OPEN_THRESHOLD = 0.5   # action dim 7 above this = open

# Grasp detection — position based
GRASP_BLOCKED_MIN      = 10    # if gripper stops above this % when closing = object detected
GRASP_CONFIRM_STEPS    = 3     # consecutive confirmations before declaring success

# Safety
MAX_INFERENCE_CYCLES   = 50    # hard stop after N cycles with no grasp
JOINT_LIMITS           = np.array([170, 130, 150, 160, 160, 160], dtype=np.float32)
# ──────────────────────────────────────────────────────────────────────────────

latest_frame = None
frame_lock   = threading.Lock()


# ── Setup ─────────────────────────────────────────────────────────────────────

def apply_patches():
    import transformers
    src = '/workspace/src/openpi/models_pytorch/transformers_replace'
    dst = os.path.dirname(transformers.__file__)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    subprocess.run(
        ['python', '/workspace/openpi_on_thor/patches/apply_gemma_fixes.py'],
        capture_output=True
    )
    print('[setup] Patches applied.')


def load_policy():
    print('[setup] Loading pi0.5 TensorRT policy...')
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    from trt_model_forward import setup_pi0_tensorrt_engine

    config = training_config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    policy = setup_pi0_tensorrt_engine(policy, ENGINE_PATH)
    print('[setup] TensorRT engine ready.')
    return policy


def connect_robot():
    import Robot as FR
    print(f'[setup] Connecting to FR3 at {ROBOT_IP}...')
    robot = FR.RPC(ROBOT_IP)
    ret, joints = robot.GetActualJointPosDegree()
    if ret != 0:
        raise RuntimeError(f'FR3 connection failed: ret={ret}')
    print(f'[setup] FR3 connected. Joints: {[round(j, 2) for j in joints[:6]]}')
    robot.RobotEnable(1)
    time.sleep(2)
    return robot


def setup_gripper(robot):
    print('[setup] Configuring gripper (DH Robotics AG series)...')
    robot.SetGripperConfig(GRIPPER_COMPANY, GRIPPER_DEVICE)
    robot.ActGripper(GRIPPER_INDEX, 1)
    time.sleep(2)
    robot.MoveGripper(
        GRIPPER_INDEX, 100, GRIPPER_VEL, GRIPPER_FORCE,
        GRIPPER_MAXTIME, 1, 0, 0, 0, 0
    )
    print('[setup] Gripper ready (open).')


def open_camera():
    print(f'[setup] Opening camera {CAMERA_INDEX}...')
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera')
    print('[setup] Camera ready.')
    return cap


# ── Camera thread ──────────────────────────────────────────────────────────────

def camera_save_thread(cap):
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
            cv2.imwrite(CAMERA_SAVE_PATH, frame)
        time.sleep(0.5)


def grab_frame():
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None \
                else np.zeros((480, 640, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame_rgb, (224, 224))


# ── Gripper control ───────────────────────────────────────────────────────────

def open_gripper(robot):
    robot.MoveGripper(
        GRIPPER_INDEX, 100, GRIPPER_VEL, GRIPPER_FORCE,
        GRIPPER_MAXTIME, 1, 0, 0, 0, 0
    )


def close_gripper(robot):
    robot.MoveGripper(
        GRIPPER_INDEX, 0, GRIPPER_VEL, GRIPPER_FORCE,
        GRIPPER_MAXTIME, 1, 0, 0, 0, 0
    )


def control_gripper(robot, action_dim7):
    """Control gripper from action output. Returns True if close was commanded."""
    if action_dim7 > GRIPPER_OPEN_THRESHOLD:
        open_gripper(robot)
        return False
    else:
        close_gripper(robot)
        return True


def check_grasp(robot):
    """
    Returns True if gripper is blocked by an object.
    Uses position feedback — if commanded to close (0%)
    but stopped above GRASP_BLOCKED_MIN%, object is in grasp.
    """
    ret, fault, position = robot.GetGripperCurPosition()
    if ret == 0 and fault == 0 and position > GRASP_BLOCKED_MIN:
        print(f'[grasp] Gripper blocked at {position}% — object detected.')
        return True
    return False


# ── Inference + execution ─────────────────────────────────────────────────────

def run_inference(policy, robot, prompt):
    ret, joints = robot.GetActualJointPosDegree()
    current_joints = np.array(joints[:6], dtype=np.float32)
    image = grab_frame()

    obs = {
        'observation/exterior_image_1_left': image,
        'observation/wrist_image_left':      image,
        'observation/joint_position':        np.deg2rad(current_joints).astype(np.float32),
        'observation/gripper_position':      np.zeros(1, dtype=np.float32),
        'prompt':                            prompt,
    }

    t0 = time.time()
    result = policy.infer(obs)
    latency = (time.time() - t0) * 1000

    actions = result.get('actions', result) if isinstance(result, dict) else result

    # Report what the model decided
    gripper_actions = actions[:, 7]
    joint_magnitudes = np.abs(actions[:, :6]).mean(axis=1)
    dominant_joint = np.abs(actions[0, :6]).argmax()
    joint_names = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']

    print(f'[infer] {latency:.1f}ms | '
          f'range [{actions.min():.3f}, {actions.max():.3f}] | '
          f'gripper_action={actions[0, 7]:.3f}')
    print(f'[model] Dominant joint: {joint_names[dominant_joint]} (joint {dominant_joint})')
    print(f'[model] Avg movement magnitude: {joint_magnitudes.mean():.4f} rad')
    print(f'[model] Gripper trend: {"CLOSING" if gripper_actions.mean() < 0.5 else "OPENING"}')
    print(f'[model] First step delta (deg): {[round(float(np.rad2deg(x) * ACTION_SCALE), 2) for x in actions[0, :6]]}')

    return actions, current_joints


def execute_actions(robot, actions, current_joints):
    """
    Execute one action chunk. Returns True if grasp confirmed.
    Monitors gripper position after each close attempt.
    """
    step_dt    = 1.0 / CONTROL_HZ
    grasp_count = 0

    for i in range(actions.shape[0]):
        # Joint motion
        delta_deg = np.rad2deg(actions[i, :6]) * ACTION_SCALE
        target    = np.clip(current_joints + delta_deg, -JOINT_LIMITS, JOINT_LIMITS)

        ret = robot.MoveJ(
            list(target) + [0.0],
            tool=0, user=0,
            vel=JOINT_SPEED, acc=JOINT_ACC,
            ovl=100.0, exaxis_pos=[0, 0, 0, 0],
            blendT=-1, offset_flag=0, offset_pos=[0, 0, 0, 0, 0, 0]
        )
        if ret != 0:
            print(f'[warn] MoveJ step {i} ret={ret}')

        # Gripper
        tried_close = control_gripper(robot, float(actions[i, 7]))

        # Grasp check — only when gripper is closing
        if tried_close:
            time.sleep(0.3)  # brief wait for gripper to actuate
            if check_grasp(robot):
                grasp_count += 1
                print(f'[grasp] Confirmed {grasp_count}/{GRASP_CONFIRM_STEPS}')
                if grasp_count >= GRASP_CONFIRM_STEPS:
                    return True
            else:
                grasp_count = 0

        current_joints = target
        time.sleep(step_dt)

    return False


# ── Prompt ────────────────────────────────────────────────────────────────────

def get_prompt():
    print('\n' + '=' * 50)
    print('  Task Prompt')
    print('  Tip: Be specific — describe the object color,')
    print('  shape, and action clearly.')
    print('  Example: "pick up the white cylinder on the table"')
    print('=' * 50)
    prompt = input('  Your prompt: ').strip()
    if not prompt:
        prompt = 'pick up the object on the table'
        print(f'  Using default: "{prompt}"')
    return prompt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 50)
    print('  pi0.5 + FR3 Demo Runner')
    print('  Jetson AGX Thor | TensorRT NVFP4')
    print('=' * 50 + '\n')

    # --- Setup ---
    apply_patches()
    policy = load_policy()
    robot  = connect_robot()
    setup_gripper(robot)
    cap    = open_camera()

    cam_thread = threading.Thread(target=camera_save_thread, args=(cap,), daemon=True)
    cam_thread.start()
    print(f'[setup] Live camera → {CAMERA_SAVE_PATH}')
    print('[setup] All systems ready.\n')

    # --- Task prompt ---
    prompt = get_prompt()
    print(f'\n[task] "{prompt}"')
    print(f'[task] Max cycles: {MAX_INFERENCE_CYCLES} | '
          f'Action scale: {ACTION_SCALE} | Control Hz: {CONTROL_HZ}')
    print('[task] Starting. Ctrl+C to abort.\n')

    # --- Control loop ---
    try:
        grasped = False
        cycle   = 0

        while not grasped and cycle < MAX_INFERENCE_CYCLES:
            cycle += 1
            print(f'--- Cycle {cycle}/{MAX_INFERENCE_CYCLES} ---')
            actions, current_joints = run_inference(policy, robot, prompt)
            grasped = execute_actions(robot, actions, current_joints)

        # --- Outcome ---
        if grasped:
            print('\n' + '=' * 50)
            print('  ✓ GRASP SUCCESS')
            print(f'  Task: "{prompt}"')
            print('  Robot holding position.')
            print('  Press Ctrl+C to release and exit.')
            print('=' * 50)
            while True:
                time.sleep(1)
        else:
            print(f'\n[done] Max cycles ({MAX_INFERENCE_CYCLES}) reached without grasp.')
            print('[done] Review camera_view.jpg and adjust prompt or positioning.')

    except KeyboardInterrupt:
        print('\n[stop] Aborted by user.')

    finally:
        print('[stop] Shutting down...')
        robot.StopMotion()
        open_gripper(robot)
        time.sleep(1)
        robot.RobotEnable(0)
        cap.release()
        print('[stop] Robot disabled. Session complete.')


if __name__ == '__main__':
    main()