import os
import random
from statistics import mean

try:
    import matplotlib

    # Configuration for environments without a graphical window
    if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

MAX_QUEUE = 9
NUM_PHASES = 2  # 0: Green N-S, 1: Green O-E

# Learning hyperparameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 1.0     # Initial exploration probability
epsilon_decay = 0.995
min_epsilon = 0.01


def initialize_q_table():
    """Creates a Q-table with states (ns, oe, phase) and actions (keep/switch)."""
    return [
        [
            [[0.0, 0.0] for _ in range(NUM_PHASES)]
            for _ in range(MAX_QUEUE + 1)
        ]
        for _ in range(MAX_QUEUE + 1)
    ]


q_table = initialize_q_table()


def select_action(ns, oe, phase):
    """Selects an action using an epsilon-greedy policy."""
    global epsilon

    if random.random() < epsilon:
        return random.randint(0, 1)

    values = q_table[ns][oe][phase]
    return max(range(len(values)), key=values.__getitem__)


def simulate_step(ns, oe, phase, action):
    """
    Action: 0 = Keep the current phase, 1 = Switch phase
    """
    if action == 1:
        phase = 1 - phase  # Toggle the traffic light phase

    # Stochastic vehicle arrivals at each queue
    arrivals_ns = random.randint(0, 2)
    arrivals_oe = random.randint(0, 2)

    # Vehicle departures depending on the active phase
    departures_ns = random.randint(1, 3) if phase == 0 else 0
    departures_oe = random.randint(1, 3) if phase == 1 else 0

    # Update queue lengths while respecting state boundaries
    new_ns = min(MAX_QUEUE, max(0, ns + arrivals_ns - departures_ns))
    new_oe = min(MAX_QUEUE, max(0, oe + arrivals_oe - departures_oe))

    # Reward: negative waiting vehicles, plus a small switching cost
    switch_cost = 0.5 if action == 1 else 0.0
    reward = -(new_ns + new_oe) - switch_cost

    return (new_ns, new_oe, phase), reward


def train_model(episodes=1000, steps_per_episode=100):
    """Trains the Q-learning model and returns the reward list per episode."""
    global epsilon, q_table

    q_table = initialize_q_table()
    epsilon = 1.0
    rewards_per_episode = []

    for _ in range(episodes):
        ns, oe, phase = random.randint(0, 4), random.randint(0, 4), 0
        total_reward = 0

        for _ in range(steps_per_episode):
            action = select_action(ns, oe, phase)

            (new_ns, new_oe, new_phase), reward = simulate_step(ns, oe, phase, action)
            total_reward += reward

            max_q_next = max(q_table[new_ns][new_oe][new_phase])
            actual_q = q_table[ns][oe][phase][action]

            q_table[ns][oe][phase][action] = actual_q + alpha * (reward + gamma * max_q_next - actual_q)

            ns, oe, phase = new_ns, new_oe, new_phase

        rewards_per_episode.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return rewards_per_episode


def plot_performance(rewards_per_episode):
    """Generates a chart comparing the model's initial and final performance."""
    if plt is None:
        print("Matplotlib is not installed. To enable charts, run:")
        print("python -m pip install matplotlib")
        return

    if not rewards_per_episode:
        print("No training data is available to plot.")
        return

    initial_window = rewards_per_episode[:50]
    final_window = rewards_per_episode[-50:]
    initial_average = mean(initial_window)
    final_average = mean(final_window)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode, color="royalblue", linewidth=1.5, alpha=0.8)
    plt.axhline(initial_average, color="green", linestyle="--", label=f"Initial avg: {initial_average:.2f}")
    plt.axhline(final_average, color="crimson", linestyle="--", label=f"Final avg: {final_average:.2f}")
    plt.title("Performance by episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    labels = ["Initial", "Final"]
    values = [initial_average, final_average]
    colors = ["forestgreen", "crimson"]
    plt.bar(labels, values, color=colors, width=0.5)
    plt.title("Performance comparison")
    plt.ylabel("Average reward")
    for i, value in enumerate(values):
        plt.text(i, value + 0.05 * max(abs(value), 1), f"{value:.2f}", ha="center", va="bottom")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "traffic_light_performance.png")
    plt.savefig(output_path, dpi=200)
    print(f"Chart saved at: {output_path}")
    plt.show()


if __name__ == "__main__":
    rewards = train_model(episodes=1000, steps_per_episode=100)
    print("Training completed. Q-table optimized for traffic light control.")

    if len(rewards) >= 50:
        initial_average = mean(rewards[:50])
        final_average = mean(rewards[-50:])
        print(f"Initial performance average (50 episodes): {initial_average:.2f}")
        print(f"Final performance average (50 episodes): {final_average:.2f}")

    plot_performance(rewards)