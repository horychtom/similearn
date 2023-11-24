"""Initialize the /wandb_ directory and log into wandb_ and hugginface."""
from config import WANDB_API_KEY
from config import HF_TOKEN
import wandb
import huggingface_hub
from .wandb_client import WandbClient
import logging
from .logger_formatter import CustomFormatter

wandb.login(key=WANDB_API_KEY,relogin=True)
huggingface_hub.login(HF_TOKEN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)