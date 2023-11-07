from sentence_transformers.evaluation import (
    SentenceEvaluator,
    SimilarityFunction,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MultipleNegativesRankingLossEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        dev_dataloader: DataLoader,
        main_similarity: SimilarityFunction = None,
    ):
        self.main_similarity = main_similarity
        self.dev_dataloader = dev_dataloader

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        self.dev_dataloader.collate_fn = model.smart_batching_collate
        self.loss_fn = MultipleNegativesRankingLoss(model=model)

        losses = []
        logger.info("Evaluating on the dev set...")
        for batch in tqdm(self.dev_dataloader, total=len(self.dev_dataloader)):
            features, labels = batch
            loss = self.loss_fn(features, labels)
            losses.append(loss.item())

        return np.mean(losses)
