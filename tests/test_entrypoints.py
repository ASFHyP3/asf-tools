def test_asf_tools(script_runner):
    ret = script_runner.run('make_composite.py', '-h')
    assert ret.success
