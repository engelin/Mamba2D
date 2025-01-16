import os
import random
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.tuner import Tuner

# Custom LightningCLI
class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--seed-from-id",
                            action='store_true',
                            help="Use (wandb) run ID as input to seed_everything")
        
        parser.add_argument("--use-lr-finder",
                            action='store_true',
                            help="Use Lightning's LR finder at start of training")

    def before_instantiate_classes(self) -> None:
        # Resume from last checkpoint if available otherwise
        # train/val/test/pred from untrained model and run learning rate finder

        self.run_lr_finder = False

        if hasattr(self.config, "fit"):
            run = self.config.fit
            msg = "Checkpoint not found, starting new run..."

            # Use LR finder if lr is config is set to null/None
            if (self.config.fit.use_lr_finder):
                self.run_lr_finder = True
        elif hasattr(self.config, "validate"):
            run = self.config.validate
            msg = "WARNING: Checkpoint not found, validating on untrained model..."
        elif hasattr(self.config, "test"):
            run = self.config.test
            msg = "WARNING: Checkpoint not found, testing on untrained model..."
        elif hasattr(self.config, "predict"):
            run = self.config.predict
            msg = "WARNING: Checkpoint not found, predicting on untrained model..."
        else:
            print("Run type not implemented!")
            exit(1)

        # If checkpoint doesn't exist under given name, start a new run
        if run.ckpt_path == None or not (os.path.exists(run.ckpt_path)):
            run.ckpt_path = None
            print(msg)
        else:
            # Ensure LR finder is turned off if resuming
            self.run_lr_finder = False

    def before_fit(self):
        # Handle --seed-from-id argument, get int seed from id
        if (self.config.fit.seed_from_id):
            random.seed(self.config.fit.trainer.logger[0].init_args.id)
            seed = int(''.join(map(str,[random.randrange(0,10,1) for _ in range(9)])))

            # Then manually seed everything from wandb id
            L.seed_everything(seed=seed, workers=True)

        # Run learning rate finder only on new run
        if (self.run_lr_finder):
            tuner = Tuner(self.trainer)
            tuner.lr_find(self.model, datamodule=self.datamodule)

 
# Custom SaveConfigCallback, sourced:
# https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html
class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
        print("\nCONFIG: ")
        print(config)
        if isinstance(trainer.logger, L.logging.Logger):
            trainer.logger.log_hyperparams({"config": config})

# Setup LightningCLI
def cli_main():
    cli = CustomCLI(save_config_callback=LoggerSaveConfigCallback,
                    save_config_kwargs={"overwrite":True},
                    parser_kwargs={"parser_mode":"omegaconf"})

if __name__ == "__main__":
    # Start training
    cli_main()
