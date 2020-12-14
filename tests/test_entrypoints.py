def test_asf_tools(script_runner):
    ret = script_runner.run('make_composite', '-h')
    assert ret.success
