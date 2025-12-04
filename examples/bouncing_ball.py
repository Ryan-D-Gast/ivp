"""Bouncing ball with event detection."""
import ivp
import numpy as np
import matplotlib.pyplot as plt

def ball(t, state, g, drag):
    y, vy = state
    return [vy, -g - drag * vy * abs(vy)]

def hit_ground(t, state, g, drag):
    return state[0]
hit_ground.terminal = True
hit_ground.direction = -1

# Simulate bounces
g, drag, restitution = 9.81, 0.02, 0.75
t_all, y_all, bounces = [], [], []
t_curr, state = 0, [10.0, 5.0]

for _ in range(20):
    sol = ivp.solve_ivp(ball, (t_curr, 15), state, args=(g, drag),
                        events=hit_ground, dense_output=True, rtol=1e-8, atol=1e-10)
    
    t_dense = np.linspace(t_curr, sol.t[-1], max(2, int((sol.t[-1]-t_curr)*100)))
    y_dense = sol.sol(t_dense)
    t_all.extend(t_dense)
    y_all.extend(y_dense[0])
    
    if len(sol.t_events[0]) == 0:
        break
    t_curr = sol.t_events[0][0]
    bounces.append(t_curr)
    vy_new = -restitution * sol.y_events[0][0][1]
    if abs(vy_new) < 0.1:
        break
    state = [0.0, vy_new]

print(f"Bounces: {len(bounces)}")

plt.figure(figsize=(10, 4))
plt.plot(t_all, y_all, 'b-')
plt.axhline(y=0, color='brown', linewidth=2)
for tb in bounces[:5]:
    plt.axvline(x=tb, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Bouncing Ball with Air Resistance')
plt.grid(True, alpha=0.3)
plt.show()
