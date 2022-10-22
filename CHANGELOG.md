# Changelog

## Working on



## Previous versions

- v0.1.7
  - add `SimpleMetricTask` for training and evaluation with metric instances
  - add `MetricBase` as a reference base class for `SimpleMetricTask`
  - fix online testing bug when data is not provided
  - show dataset name in cached transforming
  - show the number of model parameters when training
  - add `type_idx=None` to support micro-averaged-only `tagging_f1` scores calculation

- v0.1.6
    - add `warmup_proportion` to default config. ref to [#7](https://github.com/Spico197/REx/issues/7)
    - add `rex version` command
    - fix train loss == eval loss problem in issue [#6](https://github.com/Spico197/REx/issues/6)
    - fix type err in [#9](https://github.com/Spico197/REx/issues/9)
    - add `encode_one` and `decode_one` into `LabelEncoder`
    - add json-friendly dumping type convertion function (as a `dump_json*` built-in convertion)
    - add tests for relation extraction tasks
- v0.1.5: add `update_before_tensorify` into `GeneralCollateFn`, fix logging level displaying problem
- v0.1.4: move accelerate to `rex.__init__`, update multi process tqdm & logging (only show in the main process in default), remove cache in debug mode, fix bugs in `rex.cmds.new`, add `rank_zero_only` in task dump, `load_best_ckpt` if `resumed_training`
- v0.1.3: fix emb import
- v0.1.1: update registry and add `accelerate` multi-gpu support
- v0.1.0: huge update with lots of new features, check the usage in `examples/IPRE` ~
- v0.0.15: add safe_try to kill ugly statements in example main call
- v0.0.14: update vocab embedding loading to be compatible with other embedding files
- v0.0.13: update vocab, label_encoder, fix bugs in cnn reshaping and crf importing
- v0.0.12: fix crf module import issue
- v0.0.11: fix templates data resources
- v0.0.10: update `utils.config` module, `StaticEmbedding` decoupling, remove eps in metrics, add templates generation entrypoints, add more tests (coverage stat for the whole repo, lots of codes are not test case covered)
- v0.0.9: use `argparse` instead of `click`, move loguru logger into `rex.utils.logging`, add hierarchical config setting
- v0.0.8: fix config loading, change default of `makedirs` and `dump_configfile` to be `True`
- v0.0.7: fix recursive import bug
- v0.0.6: integrate omega conf loading into the inner task, add `load_*_data` option to data managers
- v0.0.5: update ffn
- v0.0.4: return detailed classification information in `mc_prf1`, support nested dict tensor movement
- v0.0.3: fix packaging bug in `setup.py`
- v0.0.2: add black formatter and pytest testing
- v0.0.1: change `LabelEncoder.to_binary_labels` into `convert_to_multi_hot` or `convert_to_one_hot`