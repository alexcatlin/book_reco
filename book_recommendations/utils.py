import os


def fileexists(filepath):
    if os.path.exists(filepath):
        return True
    else:
        return False
