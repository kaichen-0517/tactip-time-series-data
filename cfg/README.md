Configuration
====

The scripts in the [`data`](../data/) directory use [Hydra](https://hydra.cc/) to manage their configuration.
This replaces the command line argument parsing previously provided by argparse; instead,
the possible configuration options are listed in the `yaml` files in this directory.

This has some advantages:
 - Configs can be reused/composed: per-script configs only add what they need to the base config.
 - Easier argument specification: defaults are transparently tracked in version control but can be overridden on the CLI
 - Less boilerplate: we no longer need to repeatedly write similar argparse parsers for each file
 - Fun extra stuff: can do multi-runs with different parameters, enable tab-completion, run on a grid ...

 The base configuration is provided in `runtime/base.yaml`; this is the shared configuration between all of our scripts.
 Script-specific configuration is provided in `app/*.yaml`.
 Configuration files defining different experiments/robots/sensors are in `collect/`, `robot/`, `sensor/`
 respectively.

Example
----
> [!TIP]
> You can see a list of all available options with:
> ```
> uv run python tg3/data/collect.py --help
> ```

Run the scripts from the command line as normal:
```
uv run python tg3/data/collect.py
```

To override the options, you can either change the config file (more permanent; leaves a trace for what you ran) or
make the change on the command line (easier to quickly change things).

To specify an argument on the command line, use:
```
uv run python tg3/data/collect.py sample_nums=[10,20] data_dirs=["tmp1","tmp2"]
```
Lists are provided by using [square brackets] with comma-separated (no spaces!) contents.
Dictionaries are similar but with {curly braces}.

### Changing the Embodiment/Experiment
These can be specified with
```
uv run python tg3/data/collect.py collect=surface_zRxy
```
The available embodiments and experiments are listed in
the yaml files in [`collect/`](collect/) and [`embody/`](embody/).


### Multirun
You can run a script several times with different options 
(in the same process) by using the `--multirun` or `-m` flag:
```
uv run python tg3/data/collect.py --multirun collect=edge_xRz,surface_zRxy sample_nums=[3],[5]
```
Note that this will run 4 times - once for each combination of `collect` and `sample_nums`.

> [!WARNING]
> Multiple runs using hydra run in the same python process
> (like a for-loop.) This means that if a previous run breaks something,
> (e.g. if it sets a global variable to a bad value) then subsequent runs may also break.

For Developers
----
How it all works

### Inheritance
We want there to be a tree-like config structure, where some config (e.g. the embodiment) is shared by several scripts.
This is achieved by having a base-level config file (`cfg/runtime/base.yaml`) and script-specific config files (`cfg/app/*.yaml`) which
provide additional options - e.g., the test/train split for `process.py`.

### Global Namespace
The nested config structure can quickly become unwieldy.
For example, we have our base config in `runtime/base.yaml` and our script-specific config in `app/collect.yaml`.
This means that, naively, we would have to access things in the base config from our python script using e.g. `cfg.app.runtime.path`.
We can instead add everything to the root of the config dictionary by writing
```
# @package _global_
```
at the top of each config file; this means we can access things in the config directly with `cfg.path`.
This is slightly neater, as it means that we don't have to worry about the directory structure of the config files when we're writing our python scripts.
It exposes a slight risk of the config file becoming bloated, however.

### The `hydra.main` Decorator
Python knows to use hydra for config collection and composition because of the `hydra.main` decorator around the main function.
A minimal example is:

```
import hydra

@hydra.main(config_path="my_config_dir", config_name="example/collect", version_base=None):
def run(cfg):
    print(cfg)

if __name__ == "__main__":
    run()
```

This will call the `run` function, which is decorated with `hydra.main`.
`hydra.main` triggers python to collect the config files (here, located at `my_config_dir/example/collect.yaml`).

`version_base=None` tells hydra to use defaults according to the current hydra version.
These may change depending on the exact version of hydra installed, so you may want to change this to a specific
version (e.g. `version_base="1.1"`) if you are relying on specific default behaviour.
