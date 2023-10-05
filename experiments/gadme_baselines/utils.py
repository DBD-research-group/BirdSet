import wandb 
from omegaconf import OmegaConf
from gadme import datasets

def initialize_wandb(args):
    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        group=args.wandb.group,
        reinit=args.wandb.reinit,
        mode = args.wandb.mode,
        name=args.model.name+'_'+args.dataset.name+'#'+str(args.random_seed),
        config = OmegaConf.to_container(
            args, 
            resolve=True, 
            throw_on_missing=True
        )
    )

# def build_dataset(args):
#     if args.dataset.name == 'sapsucker':
#         data = datasets.base