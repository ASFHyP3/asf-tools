def test_make_composite(script_runner):
    ret = script_runner.run('make_composite', '-h')
    assert ret.success


def test_water_map(script_runner):
    ret = script_runner.run('water_map', '-h')
    assert ret.success


def test_make_hand(script_runner):
    ret = script_runner.run('calculate_hand', '-h')
    assert ret.success
