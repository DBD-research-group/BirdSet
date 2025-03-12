# Scripts for multiple runs

To execute runs for multiple different models and datasets scripts can be very helpful to automatically start new runs once the previous ones are finished. Additionally they make it easier to use different settings and hydra arguments.

The scripts use the [anti_crash script](../train_anti_crash.sh) to detect GPU crashes which works mot of the time and retries up to 3 times.

####  General script logic:
The specific script is called and then additional params can be added using `--param arg1,arg2,...`

The `--config` param is always needed and defines the experiment path within the biofoundation project.

The following optional params are available:
| Parameter | Description |
|-----------|-------------|
| `--models` | Specifies the models to use |
| `--datasets` | Specifies the datasets to use |
| `--seeds` | Specifies the seed(s) to use |
| `--tags` | Specifies a list of tags to add |
| `--gpu` | Specifies the GPU number to use |
| `--extras` | allows parsing any additional hydra arguments |

With `Ctrl + C` the current experiment can be skipped and with a second press within 3 seconds the entire run script can be quit. 

## BEANS
For the BEANS benchmark the [run_beans.sh](run_beans.sh) script is available:

Example:

`projects/biofoundation/scripts/run_beans.sh --config beans/linearprobing --seeds 2 --models hubert,beats --datasets beans_watkin,beans_cbi --tags test,run --gpu 0 --extras logger.wandb.group=example`

The correct number of classes are automatically added.

#### BEANS Datasets

| Dataset          |
|------------------|
| beans_watkins    |
| beans_cbi        |
| beans_dogs       |
| beans_humbugdb   |
| beans_bats       |

## Birdset
The birdset script [run_birdset.sh](run_birdset.sh) script is very similar to the BEANS version but for the birdset benchmark:

Example:

