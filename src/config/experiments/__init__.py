import importlib, ntpath, os
REGISTRY = {}

# automatically fill registry from experiment entries
# subdirs = [x[0] for x in os.walk(ntpath.dirname(__file__))]
# for subdir in subdirs:
#     if os.path.isfile(os.path.join(subdir, "config.py")):
#         fname=os.path.split(subdir)[-1]
#         REGISTRY[fname] = importlib.import_module(".{}.config".format(fname), package="config.experiments").get_cfg

for root, dirs, files in os.walk(ntpath.dirname(__file__)):
    path = root.split(os.sep)
    if "config.py" in files:
        folders = []
        path = os.path.relpath(root, ntpath.dirname(__file__))
        while 1:
            path, folder = os.path.split(path)
            if folder != "":
                folders.append(folder)
            else:
                if path != "":
                    folders.append(path)
                break
        folders.reverse()
        REGISTRY["/".join(folders)] = importlib.import_module(".{}.config".format(".".join(folders)), package="config.experiments").get_cfg

