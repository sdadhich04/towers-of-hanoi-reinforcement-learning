from src.toh_rl.mdp import (
    TowersOfHanoiMdp,
    custom_alpha,
    custom_epsilon,
    extract_policy,
    initial_q_table,
    initial_v_table,
    q_update,
    value_iteration,
)


def main() -> None:
    mdp = TowersOfHanoiMdp(disks=3, gamma=0.95, slip_probability=0.1)
    values = initial_v_table(mdp)
    q_values = initial_q_table(mdp)

    for _ in range(50):
        values, q_values, delta = value_iteration(mdp, values)
        if delta < 1e-9:
            break

    policy = extract_policy(mdp, q_values)
    transition = (mdp.start_state, policy[mdp.start_state], -1.0, mdp.step(mdp.start_state, policy[mdp.start_state]))
    q_update(mdp, q_values, transition, alpha=custom_alpha(0))

    print(f"states: {len(mdp.states)}")
    print(f"start: {mdp.start_state}")
    print(f"goal: {mdp.goal_state}")
    print(f"first action from start: {policy[mdp.start_state]}")
    print(f"epsilon step 10: {custom_epsilon(10):.4f}")
    print(f"alpha step 10: {custom_alpha(10):.4f}")


if __name__ == "__main__":
    main()
