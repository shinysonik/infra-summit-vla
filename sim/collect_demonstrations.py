"""
Collect demonstration episodes from the MuJoCo scene into LeRobot dataset
format, ready for `lerobot-train --policy.path=lerobot/smolvla_base ...`.

STATUS: this is a SCHEMA/PIPELINE SCAFFOLD, not a source of real training
data yet. The action-generation in `_scripted_episode` below is a crude
placeholder (small joint sweeps, no real reaching/grasping/hand-off logic)
-- it exists ONLY to produce structurally valid episodes so this script's
LeRobot-format output can be tested end-to-end (written, then re-loaded by
LeRobotDataset itself) before anyone invests time in real teleoperation or
scripted IK trajectories. Swap `_scripted_episode` for real demonstrations
(teleop or an actual motion planner) before training a policy you intend to
keep -- training on this placeholder's motions would teach a policy to do
exactly the useless small joint sweeps it currently produces.

Owned by: feature/mujoco-scene (schema/integration) + feature/policy-training
(real trajectory generation, once teleop/scripted-IK exists).

Seed discipline: episodes are recorded using ONLY seeds from
randomization.yaml `training.seed_range` -- never the 10 reserved eval
seeds. See DomainRandomizer.is_eval_seed / sample_training_seed.
"""
import sys
import numpy as np
import mujoco
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, "sim")
from randomization import DomainRandomizer  # noqa: E402

MODEL_PATH = "sim/assets/dinner_table_dual_so101.xml"
RANDOMIZATION_CFG = "configs/randomization.yaml"
FPS = 30

# Every actuator, in a fixed order -- this fixed order IS the 12-dim
# action/state vector layout every other component (policy, inference
# runtime) must agree on.
ACTUATOR_NAMES = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]

CAMERA_SPECS = {
    # name -> (width, height), matching configs/sim.yaml exactly.
    "overhead": (640, 480),
    "wrist_left": (320, 240),
    "wrist_right": (320, 240),
}


def build_features():
    features = {
        "action": {
            "dtype": "float32",
            "shape": (len(ACTUATOR_NAMES),),
            "names": ACTUATOR_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(ACTUATOR_NAMES),),
            "names": ACTUATOR_NAMES,
        },
    }
    for cam_name, (w, h) in CAMERA_SPECS.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def get_joint_state(model, data):
    """Read back the 12 actuated joint positions, in ACTUATOR_NAMES order --
    NOT the raw qpos array, since qpos includes free-joint object states too
    and isn't in a stable, policy-relevant order."""
    state = np.zeros(len(ACTUATOR_NAMES), dtype=np.float32)
    for i, act_name in enumerate(ACTUATOR_NAMES):
        act_id = model.actuator(act_name).id
        joint_id = model.actuator_trnid[act_id, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        state[i] = data.qpos[qpos_adr]
    return state


def render_all_cameras(renderers, data):
    frames = {}
    for cam_name, renderer in renderers.items():
        renderer.update_scene(data, camera=cam_name)
        frames[cam_name] = renderer.render()
    return frames


def _scripted_episode(model, data, renderers, control_decimation, n_steps=60):
    """PLACEHOLDER trajectory: small sinusoidal sweeps on each joint. Purely
    to exercise the recording pipeline (see module docstring) -- replace
    with real teleop or scripted IK before using output for training."""
    ctrl0 = data.ctrl.copy()
    for t in range(n_steps):
        for i, act_name in enumerate(ACTUATOR_NAMES):
            act_id = model.actuator(act_name).id
            sweep = 0.05 * np.sin(2 * np.pi * t / n_steps + i)
            data.ctrl[act_id] = ctrl0[act_id] + sweep

        for _ in range(control_decimation):
            mujoco.mj_step(model, data)

        state = get_joint_state(model, data)
        frames = render_all_cameras(renderers, data)
        yield state, data.ctrl.copy(), frames


def collect(n_episodes, repo_id, root, instruction, control_decimation=10, seed_offset=0):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    randomizer = DomainRandomizer(model, RANDOMIZATION_CFG)

    renderers = {name: mujoco.Renderer(model, height=h, width=w)
                 for name, (w, h) in CAMERA_SPECS.items()}

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=build_features(),
        root=root,
        robot_type="dual_so101_placeholder",
        use_videos=True,
    )

    seed_rng = np.random.RandomState(seed_offset)  # picks WHICH training seeds to use
    used_seeds = []

    for ep in range(n_episodes):
        seed = randomizer.sample_training_seed(seed_rng)
        assert not randomizer.is_eval_seed(seed), (
            f"sampled seed {seed} collides with a reserved eval seed -- "
            "this should be impossible given disjoint ranges; stop and check "
            "configs/randomization.yaml training.seed_range"
        )
        used_seeds.append(seed)

        randomizer.reset(data, seed)

        for state, action, frames in _scripted_episode(model, data, renderers, control_decimation):
            frame = {
                "action": action[: len(ACTUATOR_NAMES)].astype(np.float32),
                "observation.state": state,
                "task": instruction,
            }
            for cam_name, img in frames.items():
                frame[f"observation.images.{cam_name}"] = img
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"[collect] episode {ep+1}/{n_episodes} recorded (seed={seed})")

    dataset.finalize()
    print(f"\nWrote {n_episodes} episodes to {root}")
    print(f"Training seeds used (first 10 shown): {used_seeds[:10]}"
          + (" ..." if len(used_seeds) > 10 else ""))
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--repo-id", type=str, default="local/dinner_table_scaffold")
    parser.add_argument("--root", type=str, default="./demo_dataset_scaffold")
    parser.add_argument("--instruction", type=str,
                         default="open the drawer and set the table")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    collect(args.episodes, args.repo_id, args.root, args.instruction,
             seed_offset=args.seed_offset)