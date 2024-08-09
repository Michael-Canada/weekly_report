import os

from invoke import task

DIRNAME = os.path.dirname(__file__)
FOLDER_NAME = "weekly_report"


@task
def pycodestyle(ctx):
    cmd = "pycodestyle --max-line-length=88 --ignore=E203,E501,W503 {folder}".format(
        # folder=os.path.join(DIRNAME, FOLDER_NAME)
        folder=os.path.join(DIRNAME)
    )
    print(cmd)
    ctx.run(cmd)


@task
def mypy(ctx):
    config_file = os.path.join(DIRNAME, "mypy.ini")
    # root_dir = os.path.join(DIRNAME, FOLDER_NAME)
    root_dir = os.path.join(DIRNAME)

    cmd = "mypy --config-file {config_file} {root_dir}".format(
        config_file=config_file, root_dir=root_dir
    )
    print(cmd)
    ctx.run(cmd)


@task
def pylint(ctx):
    rcfile = os.path.join(DIRNAME, ".pylintrc")
    # root_dir = os.path.join(DIRNAME, FOLDER_NAME)
    root_dir = os.path.join(DIRNAME)

    cmd = "pylint --rcfile={rcfile} {root_dir}".format(rcfile=rcfile, root_dir=root_dir)
    print(cmd)
    ctx.run(cmd)


@task
def black(ctx):
    # cmd = "black --check {folder}".format(folder=os.path.join(DIRNAME, FOLDER_NAME))
    cmd = "black --check {folder}".format(folder=os.path.join(DIRNAME))
    print(cmd)
    ctx.run(cmd)


@task
def publish(ctx):
    # root_dir = os.path.join(DIRNAME, FOLDER_NAME)
    root_dir = os.path.join(DIRNAME)
    root = "gs://marginalunit-placebo-metadata/metadata"

    cmds = [
        f"python {root_dir}/ercot_resourcedb/register.py publish --root {root}/ercot.resourcedb",
        f"python {root_dir}/ercot_phase_shifters/register.py publish --root {root}/ercot.phaseshifter",
        f"python {root_dir}/ercot_unmonitored_constraints/register.py publish --root {root}/ercot.unmonitored_constraints",
        f"python {root_dir}/ercot_registered_pnodes/register.py publish --root {root}/ercot.registered_pnodes",
        f"python {root_dir}/ercot_topology_manual_override/register.py publish --root {root}/ercot.topology_manual_overrides",
    ]

    for cmd in cmds:
        print(cmd)
        ctx.run(cmd)


@task
def check(ctx):
    # root_dir = os.path.join(DIRNAME, FOLDER_NAME)
    root_dir = os.path.join(DIRNAME)

    cmds = [
        f"python {root_dir}/run sanity",
    ]

    for cmd in cmds:
        print(cmd)
        ctx.run(cmd)


# @task(black, pycodestyle, mypy, pylint, check)
@task(black)
def build(_):
    print("Done with Python checks.")
