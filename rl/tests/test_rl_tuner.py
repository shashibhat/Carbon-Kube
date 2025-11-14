from rl.rl_tuner import RLTuner, RLTunerConfig


def test_threshold_moves_with_actions():
    cfg = RLTunerConfig(threshold_values=[100.0, 200.0, 300.0])
    tuner = RLTuner(cfg)
    assert tuner.current_threshold == 200.0

    state = (150.0, 0.2)
    next_state = (140.0, 0.1)

    tuner.observe(state, +1, reward=-1.0, next_state=next_state)
    assert tuner.current_threshold in {200.0, 300.0}
