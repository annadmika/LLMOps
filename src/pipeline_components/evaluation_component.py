"""Evaluation component using BLEU, ROUGE, and Semantic Similarity metrics."""

from kfp.dsl import Dataset, Input, Metrics, Output, OutputPath, component


@component(
    base_image="cicirello/pyaction:3.11",
    packages_to_install=[
        "ragas>=0.3.5",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.5.1",
        "pandas>=2.3.2",
        "tqdm",
        "sentence-transformers>=3.0.0",  # For semantic similarity
        "numpy",
    ],
)
def evaluation_component(
    predictions: Input[Dataset],
    metrics: Output[Metrics],
    evaluation_results: OutputPath("Dataset"),  # type: ignore
):
    """
    Computes BLEU, ROUGE, and semantic similarity metrics for tarot model predictions.
    
    Expected input CSV columns:
        - user_input:  user question or prompt
        - response:    model-generated reading
        - reference:   ground truth / target reading
    """

    import logging
    import pandas as pd
    import numpy as np
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, RougeScore
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer, util

    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation component...")

    logger.info(f"Loading predictions dataset from {predictions.path}")
    predictions_df = pd.read_csv(predictions.path)

    required_cols = {"user_input", "response", "reference"}
    missing_cols = required_cols - set(predictions_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Initialize metrics and models
    bleu_metric = BleuScore()
    rouge_metric = RougeScore()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # --------------------------------------------------------------
    # Metric computation functions
    # --------------------------------------------------------------
    def compute_metrics(user_input: str, response: str, reference: str) -> dict[str, float]:
        """Compute BLEU, ROUGE, and semantic similarity for a single row."""
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            reference=reference,
        )

        # BLEU & ROUGE
        bleu = bleu_metric.single_turn_score(sample)
        rouge = rouge_metric.single_turn_score(sample)

        # Semantic similarity (cosine similarity between embeddings)
        emb_ref = embedder.encode(reference, convert_to_tensor=True)
        emb_resp = embedder.encode(response, convert_to_tensor=True)
        semantic_similarity = float(util.cos_sim(emb_resp, emb_ref).cpu().item())

        return {
            "bleu": bleu,
            "rouge": rouge,
            "semantic_similarity": semantic_similarity,
        }

    # --------------------------------------------------------------
    # Compute per-row scores
    # --------------------------------------------------------------
    logger.info("Computing BLEU, ROUGE, and semantic similarity metrics...")
    results = []
    for _, row in tqdm(predictions_df.iterrows(), total=predictions_df.shape[0]):
        try:
            row_scores = compute_metrics(
                str(row["user_input"]),
                str(row["response"]),
                str(row["reference"]),
            )
            results.append(row_scores)
        except Exception as e:
            logger.warning(f"Skipping row due to error: {e}")
            results.append({"bleu": np.nan, "rouge": np.nan, "semantic_similarity": np.nan})

    results_df = pd.DataFrame(results)
    evaluations_df = pd.concat([predictions_df, results_df], axis=1)

    # --------------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------------
    avg_bleu = evaluations_df["bleu"].mean(skipna=True)
    avg_rouge = evaluations_df["rouge"].mean(skipna=True)
    avg_semantic = evaluations_df["semantic_similarity"].mean(skipna=True)

    logger.info("Average scores:")
    logger.info(f"  BLEU: {avg_bleu:.4f}")
    logger.info(f"  ROUGE: {avg_rouge:.4f}")
    logger.info(f"  Semantic Similarity: {avg_semantic:.4f}")

    # --------------------------------------------------------------
    # Save results and log metrics
    # --------------------------------------------------------------
    logger.info(f"Saving evaluation results to {evaluation_results}")
    evaluations_df.to_csv(evaluation_results, index=False)

    metrics.log_metric("avg_bleu", avg_bleu)
    metrics.log_metric("avg_rouge", avg_rouge)
    metrics.log_metric("avg_semantic_similarity", avg_semantic)

    logger.info("Evaluation completed successfully.")
