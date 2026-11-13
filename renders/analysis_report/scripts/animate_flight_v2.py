import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

RESULT_DIR = '/home/jaeho/jaeho_ws/f16_combat/src/NeuralPlane/renders/result'
OUT_PATH = '/tmp/claude-1000/-home-jaeho-jaeho-ws-f16-combat-src-NeuralPlane/669cf53d-b683-4b46-9adf-e7197a904c00/scratchpad/flight_episode_790_with_targets.gif'

npos = np.load(f'{RESULT_DIR}/npos.npy')
epos = np.load(f'{RESULT_DIR}/epos.npy')
alt = np.load(f'{RESULT_DIR}/altitude.npy')
vt = np.load(f'{RESULT_DIR}/vt.npy')
yaw = np.load(f'{RESULT_DIR}/yaw.npy')
pitch = np.load(f'{RESULT_DIR}/pitch.npy')
target_heading = np.load(f'{RESULT_DIR}/target_heading.npy')
target_pitch = np.load(f'{RESULT_DIR}/target_pitch.npy')
target_vt = np.load(f'{RESULT_DIR}/target_vt.npy')

RAD2DEG = 180 / np.pi
yaw_deg = yaw * RAD2DEG
pitch_deg = pitch * RAD2DEG
th_deg = target_heading * RAD2DEG
tp_deg = target_pitch * RAD2DEG

# detect episode-reset points (target jumps = new goal assigned)
reset_idx = np.where(np.abs(np.diff(target_heading)) > 0.01)[0] + 1

step = max(1, len(npos) // 400)
sl = slice(None, None, step)
t_full = np.arange(len(npos)) * 0.02
t_d, npos_d, epos_d, alt_d, vt_d = t_full[sl], npos[sl], epos[sl], alt[sl], vt[sl]
yaw_d, th_d2, pitch_d, tp_d2, tv_d2 = yaw_deg[sl], th_deg[sl], pitch_deg[sl], tp_deg[sl], target_vt[sl]
reset_times = reset_idx * 0.02

fig = plt.figure(figsize=(14, 7))
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax_h = fig.add_subplot(3, 2, 2)
ax_p = fig.add_subplot(3, 2, 4)
ax_v = fig.add_subplot(3, 2, 6)

ax3d.set_xlabel('epos (east, ft)')
ax3d.set_ylabel('npos (north, ft)')
ax3d.set_zlabel('altitude (ft)')
ax3d.set_title('F16 flight path (episode_790)')
ax3d.set_xlim(epos_d.min() - 500, epos_d.max() + 500)
ax3d.set_ylim(npos_d.min() - 500, npos_d.max() + 500)
ax3d.set_zlim(alt_d.min() - 200, alt_d.max() + 200)
line3d, = ax3d.plot([], [], [], color='royalblue', lw=1.5)
point3d, = ax3d.plot([], [], [], marker='^', color='crimson', markersize=10)

def setup_ts_ax(ax, ylabel, ymin, ymax):
    ax.set_xlim(0, t_d[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel)
    for rt in reset_times:
        ax.axvline(rt, color='gray', linestyle=':', alpha=0.6)
    real_line, = ax.plot([], [], color='royalblue', label='actual')
    tgt_line, = ax.plot([], [], color='crimson', linestyle='--', label='target')
    ax.legend(loc='upper right', fontsize=8)
    return real_line, tgt_line

margin = lambda a, b: (min(a.min(), b.min()) - 10, max(a.max(), b.max()) + 10)

h_real, h_tgt = setup_ts_ax(ax_h, 'heading (deg)', *margin(yaw_d, th_d2))
ax_h.set_title('heading: actual vs target')
p_real, p_tgt = setup_ts_ax(ax_p, 'pitch (deg)', *margin(pitch_d, tp_d2))
ax_p.set_title('pitch: actual vs target')
v_real, v_tgt = setup_ts_ax(ax_v, 'airspeed (ft/s)', *margin(vt_d, tv_d2))
ax_v.set_title('airspeed: actual vs target')
ax_v.set_xlabel('time (s)')

status_text = fig.text(0.5, 0.96, '', ha='center', fontsize=11)

def init():
    line3d.set_data([], [])
    line3d.set_3d_properties([])
    point3d.set_data([], [])
    point3d.set_3d_properties([])
    for ln in (h_real, h_tgt, p_real, p_tgt, v_real, v_tgt):
        ln.set_data([], [])
    return line3d, point3d, h_real, h_tgt, p_real, p_tgt, v_real, v_tgt, status_text

def update(i):
    line3d.set_data(epos_d[:i+1], npos_d[:i+1])
    line3d.set_3d_properties(alt_d[:i+1])
    point3d.set_data([epos_d[i]], [npos_d[i]])
    point3d.set_3d_properties([alt_d[i]])

    h_real.set_data(t_d[:i+1], yaw_d[:i+1])
    h_tgt.set_data(t_d[:i+1], th_d2[:i+1])
    p_real.set_data(t_d[:i+1], pitch_d[:i+1])
    p_tgt.set_data(t_d[:i+1], tp_d2[:i+1])
    v_real.set_data(t_d[:i+1], vt_d[:i+1])
    v_tgt.set_data(t_d[:i+1], tv_d2[:i+1])

    err = abs(yaw_d[i] - th_d2[i])
    status_text.set_text(f't={t_d[i]:.1f}s   heading error={err:.1f} deg   (gray dotted = new target assigned)')
    return line3d, point3d, h_real, h_tgt, p_real, p_tgt, v_real, v_tgt, status_text

ani = FuncAnimation(fig, update, frames=len(t_d), init_func=init, interval=40, blit=False)
fig.tight_layout(rect=[0, 0, 1, 0.94])
ani.save(OUT_PATH, writer=PillowWriter(fps=25))
print('saved to', OUT_PATH)
