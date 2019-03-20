import pytest
from collections import defaultdict

def pytest_runtest_makereport(item, call):
    if item.cls is None: return
    if not hasattr(item.cls, 'failcount'):
        item.cls.failcount = 0
    if call.excinfo is not None:
        item.cls.failcount += 1
    # print("LPW: Running pytest_runtest_makereport", item.name, item.parent.name)
    # import IPython
    # IPython.embed()
    # if "incremental" in item.keywords:
    #     if call.excinfo is not None:
    #         parent = item.parent
    #         parent._previousfailed = item

# def pytest_runtest_setup(item):
#     previousfailed = getattr(item.parent, "_previousfailed", None)
#     if previousfailed is not None:
#         pytest.xfail("previous test failed (%s)" % previousfailed.name)
