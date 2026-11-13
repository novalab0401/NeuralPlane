import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = '/home/jaeho/jaeho_ws/f16_combat/src/NeuralPlane/renders/result'
OUT_PATH = '/tmp/claude-1000/-home-jaeho-jaeho-ws-f16-combat-src-NeuralPlane/669cf53d-b683-4b46-9adf-e7197a904c00/scratchpad/oscillation_evidence.png'

yaw = np.load(f'{RESULT_DIR}/yaw.npy') * 180 / np.pi
target_heading = np.load(f'{RESULT_DIR}/target_heading.npy') * 180 / np.pi
pitch = np.load(f'{RESULT_DIR}/pitch.npy') * 180 / np.pi
target_pitch = np.load(f'{RESULT_DIR}/target_pitch.npy') * 180 / np.pi
el = np.load(f'{RESULT_DIR}/el.npy')
ail = np.load(f'{RESULT_DIR}/ail.npy')
rud = np.load(f'{RESULT_DIR}/rud.npy')

t = np.arange(len(yaw)) * 0.02

# pick the first full episode segment (before first target-reassignment jump) to zoom in on
jumps = np.where(np.abs(np.diff(target_heading)) > 0.5)[0]
end = jumps[0] if len(jumps) > 0 else min(2500, len(yaw) - 1)
start = 50  # skip the very first transient

print(f'Zooming into steps [{start}:{end}]  (~{(end-start)*0.02:.0f}s of one episode)')

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

axes[0].plot(t[start:end], yaw[start:end], color='royalblue', label='actual heading')
axes[0].plot(t[start:end], target_heading[start:end], color='crimson', linestyle='--', label='target heading')
axes[0].set_ylabel('heading (deg)')
axes[0].legend()
axes[0].set_title('Zoomed-in single episode: heading tracking')

axes[1].plot(t[start:end], pitch[start:end], color='royalblue', label='actual pitch')
axes[1].plot(t[start:end], target_pitch[start:end], color='crimson', linestyle='--', label='target pitch')
axes[1].set_ylabel('pitch (deg)')
axes[1].legend()

axes[2].plot(t[start:end], el[start:end], color='k')
axes[2].set_ylabel('elevator (el)')
axes[2].set_title('Control surface commands (raw, same time window)')

axes[3].plot(t[start:end], ail[start:end], color='k')
axes[3].set_ylabel('aileron (ail)')

axes[4].plot(t[start:end], rud[start:end], color='k')
axes[4].set_ylabel('rudder (rud)')
axes[4].set_xlabel('time (s)')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=110)
print('saved plot to', OUT_PATH)

# quantitative "jerk" metric: mean absolute step-to-step change
def jerk(x):
    return np.mean(np.abs(np.diff(x)))

print('\n--- quantitative control smoothness (mean |delta| per 0.02s step) ---')
print(f'elevator jerk : {jerk(el[start:end]):.4f}')
print(f'aileron  jerk : {jerk(ail[start:end]):.4f}')
print(f'rudder   jerk : {jerk(rud[start:end]):.4f}')
print(f'heading error std within episode: {np.std(yaw[start:end] - target_heading[start:end]):.2f} deg')

# how many sign-flips (direction reversals) in elevator command -> proxy for oscillation
sign_flips = np.sum(np.diff(np.sign(np.diff(el[start:end]))) != 0)
print(f'elevator direction reversals in {end-start} steps: {sign_flips}  ({sign_flips/(end-start)*100:.1f}% of steps)')
