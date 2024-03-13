
# def extras(cfg: DictConfig) -> None:
#     """Applies optional utilities before the task is started.

#     Utilities:
#     - Ignoring python warnings
#     - Setting tags from command line
#     - Rich config printing

#     Args:
#         cfg (DictConfig): Main config.
#     """

#     # return if no `extras` config
#     if not cfg.get("extras"):
#         log.warning("Extras config not found! <cfg.extras=null>")
#         return

#     # disable python warnings
#     if cfg.extras.get("ignore_warnings"):
#         log.info(
#             "Disabling python warnings! <cfg.extras.ignore_warnings=True>"
#         )
#         warnings.filterwarnings("ignore")

#     # prompt user to input tags from command line if none are provided in the config
#     if cfg.extras.get("enforce_tags"):
#         log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
#         rich_utils.enforce_tags(cfg, save_to_file=True)

#     # pretty print config tree using Rich library
#     if cfg.extras.get("print_config"):
#         log.info(
#             "Printing config tree with Rich! <cfg.extras.print_config=True>"
#         )
#         rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)