from env import GridWorld


def trst_reset_restroes_initial_state():
    env = GridWorld()

    env.state = (2,3)
    env.steps = 12

    result = env.reset()

    assert result == (0,0)
    assert env.state == (0,0)
    assert env.steps == 0

#Arrange → Act → Assert
#准备 → 执行 → 验证

def test_transition_stays_when_hitting_boundary():
    env = GridWorld()

    result = env.transition((0,0),0)

    assert result == ((0,0),-1.0,False)
