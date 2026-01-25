"""
Callback to monitor learnable temperature parameter during training
"""
from lightning.pytorch import Callback
from loguru import logger as logging


class TauMonitor(Callback):
    """Monitor and log the learnable tau parameter during training"""
    
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log tau value periodically"""
        if batch_idx % self.log_every_n_steps == 0:
            # Get tau from the cpc_loss module
            if hasattr(pl_module, 'cpc_loss') and hasattr(pl_module.cpc_loss, 'learnable_tau'):
                if pl_module.cpc_loss.learnable_tau:
                    tau_value = pl_module.cpc_loss.tau.item()
                    
                    # Log to trainer
                    pl_module.log('train/tau', tau_value, on_step=True, on_epoch=False)
                    
                    # Print to console
                    if batch_idx % (self.log_every_n_steps * 10) == 0:
                        logging.info(f"Step {trainer.global_step}: tau = {tau_value:.6f}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log tau at end of validation"""
        if hasattr(pl_module, 'cpc_loss') and hasattr(pl_module.cpc_loss, 'learnable_tau'):
            if pl_module.cpc_loss.learnable_tau:
                tau_value = pl_module.cpc_loss.tau.item()
                pl_module.log('val/tau', tau_value, on_step=False, on_epoch=True)
                logging.info(f"Epoch {trainer.current_epoch}: tau = {tau_value:.6f}")
