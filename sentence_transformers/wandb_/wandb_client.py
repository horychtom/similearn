from config import PROJECT_NAME
import wandb
from sentence_transformers.util import get_dataset_name
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class WandbClient:
    """Client for interacting with wandb."""

    def __init__(self, run_name=str):
        """Initialize the client."""
        self.project_name = PROJECT_NAME  # global name of the wandb project
        self.run = wandb.init(
            project=self.project_name,
            name=run_name,
        )  # initialize the run
        self.artifacts_remote_path = (
            Path(".") / self.run.entity / self.project_name
        )
        self.artifacts_local_path = Path(".") / "data" / "wandb_artifacts"
        # self

    def upload_model(self, pt_model) -> None:
        """
        Uploads a PyTorch model to the W&B server.

        Args:
            pt_model (str): The local path to the PyTorch model file.

        Returns:
            None
        """
        artifact = wandb.Artifact(name="first_attempt", type="model")
        artifact.add_file(local_path=pt_model, name="cool_model")
        self.run.log_artifact(artifact)

    def upload_dataset(self, dataset) -> None:
        """
        Uploads a dataset to the W&B server.

        Args:
            dataset: The path to the dataset file.

        Returns:
            None
        """
        artifact = wandb.Artifact(
            name=get_dataset_name(dataset),
            type="dataset",
        )
        artifact.add_file(local_path=dataset, name=get_dataset_name(dataset))
        self.run.log_artifact(artifact)

    def load_dataset(self, dataset_name, version="latest") -> Path:
        """
        Loads a dataset from the specified version and returns the local path to the dataset.

        Args:
            dataset_name (str): The name of the dataset to load.
            version (str): The version of the dataset to load. Defaults to "latest".

        Returns:
            Path: The local path to the dataset.
        """

        local_path = self.artifacts_local_path / f"{dataset_name}:{version}"

        if (local_path / dataset_name).is_file():
            logger.info(f"Dataset {dataset_name} already exists locally.")
            return local_path / dataset_name

        artifact = self.run.use_artifact(
            str(self.artifacts_remote_path / f"{dataset_name}:{version}"),
            type="dataset",
        )
        artifact_dir = Path(artifact.download(local_path))
        return artifact_dir / dataset_name

    def finish(self):
        """Uploads all the artifacts and finishes the run."""
        wandb.finish()
