def test_make_composite(script_runner):
    ret = script_runner.run('make_composite', '-h')
    assert ret.success


def test_water_map(script_runner):
    ret = script_runner.run('make_composite', '-h')
    assert ret.success
