import logging
import os
from typing import Any

import pandas as pd
from langchain_ollama import ChatOllama
from matplotlib import pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field

from src.engine import query_engine
from data.benchmark.benchmark_data import benchmark_data
from src.utils import get_config, get_logger


def check_evals(
    config_file_path: str,
    log_dir_path: str
) -> pd.DataFrame:

    # ----------------------------------------------------------------------------------------------------------
    # Basic Setup
    # ----------------------------------------------------------------------------------------------------------

    # Get Logger
    logger = get_logger(log_dir_path=log_dir_path)

    # create benchmark dataframe
    df_benchmark = pd.DataFrame(benchmark_data)
    df_benchmark = df_benchmark.reset_index().rename(columns={"index": "question_id"})
    df_benchmark["question_id"] = df_benchmark["question_id"] + 1

    # initialize evals llm
    llm_model_name = get_config("params.evals.llm.model_name", config_path=config_file_path)
    llm_temperature = get_config("params.evals.llm.temperature", config_path=config_file_path)
    evals_llm = ChatOllama(model=llm_model_name, temperature=llm_temperature)

    # load evals path
    evals_path = get_config("paths.evals", config_path=config_file_path)

    # ----------------------------------------------------------------------------------------------------------
    # Core Logic
    # ----------------------------------------------------------------------------------------------------------

    # initialize evals dataframe by querying questions and fetching llm answers
    logger.info("build evals dataframe: start")
    df_evals = build_evals_dataframe(
        df_benchmark=df_benchmark,
        config_file_path=config_file_path,
        logger=logger
    )
    logger.info("build evals dataframe: end")

    # check rephrase
    logger.info("check rephrase: start")
    rephrase_results = df_evals.apply(check_rephrase, args=(evals_llm, logger, ), axis=1)
    df_evals["rephrase_think"] = rephrase_results.apply(lambda x: x.think)
    df_evals["rephrase_flag"] = rephrase_results.apply(lambda x: x.flag)
    df_evals["rephrase_reason"] = rephrase_results.apply(lambda x: x.reason)
    logger.info("check rephrase: end")

    # check relevance
    logger.info("check relevance: start")
    relevance_results = df_evals.apply(check_relevance, args=(evals_llm, logger, ), axis=1)
    df_evals["relevance_think"] = relevance_results.apply(lambda x: x.think)
    df_evals["relevance_flag"] = relevance_results.apply(lambda x: x.flag)
    df_evals["relevance_reason"] = relevance_results.apply(lambda x: x.reason)
    logger.info("check relevance: end")

    # check groundedness
    logger.info("check groundedness: start")
    groundedness_results = df_evals.apply(check_groundedness, args=(evals_llm, logger, ),  axis=1)
    df_evals["groundedness_think"] = groundedness_results.apply(lambda x: x.think)
    df_evals["groundedness_flag"] = groundedness_results.apply(lambda x: x.flag)
    df_evals["groundedness_reason"] = groundedness_results.apply(lambda x: x.reason)
    logger.info("check groundedness: end")

    # check correctness
    logger.info("check correctness: start")
    correctness_results = df_evals.apply(check_correctness, args=(evals_llm, logger, ),  axis=1)
    df_evals["correctness_think"] = correctness_results.apply(lambda x: x.think)
    df_evals["correctness_flag"] = correctness_results.apply(lambda x: x.flag)
    df_evals["correctness_reason"] = correctness_results.apply(lambda x: x.reason)
    logger.info("check correctness: end")

    # generate and save plots
    logger.info("generate and save plots: start")
    generate_and_save_plots(df_evals=df_evals, evals_path=evals_path)
    logger.info("generate and save plots: end")

    # save df_evals
    logger.info("save df_evals: start")
    df_evals.to_csv(os.path.join(evals_path, "df_evals.csv"), index=False)
    logger.info("save df_evals: end")

    # generate and save reports
    logger.info("generate and save reports: start")
    generate_and_save_reports(df_evals=df_evals, evals_path=evals_path)
    logger.info("generate and save reports: end")

    return


# ----------------------------------------------------
# pydantic schema
# ----------------------------------------------------

class JudgeResult(BaseModel):
    think: str = Field(description="Short reasoning used to decide the evaluation result")
    flag: bool = Field(description="True if the evaluation passes, otherwise False")
    reason: str = Field(description="Brief explanation for the final decision")


# ----------------------------------------------------
# initialize evals dataframe
# ----------------------------------------------------

def build_evals_dataframe(
    df_benchmark: pd.DataFrame,
    config_file_path: str,
    logger: logging.Logger
) -> pd.DataFrame:

    # initialize rows list
    rows = []

    # answer each question found in df_benchmark
    for row in df_benchmark.itertuples():

        logger.info(f"Answering question {row.question_id}...")

        result = query_engine(
            question=row.question,
            config_file_path=config_file_path,
            chat_history=row.chat_history
        )

        rows.append({
            "question_id": row.question_id,
            "question": row.question,
            "ground_truth_answer": row.answer,
            "group": row.group,

            # generated answer
            "formatted_chat_history": result["formatted_chat_history"],
            "retrieval_question": result["retrieval_question"],
            "predicted_answer": result["answer"],
            "relevance_check": result["relevance_check"],
            "plan": result["plan"],
            "groundedness_check": result["groundedness_check"],
            "citation_check": result["citation_check"],
            "usefulness_check": result["usefulness_check"],

            # retrieval
            "context": result["context"],
            "chunks": result["chunks"],
            "llm_error": result["llm_error"],
            "num_chunks": len(result["chunks"])
        })

    # create df_evals using rows list
    df_evals = pd.DataFrame(rows)

    return df_evals


# ----------------------------------------------------
# LLM judges
# ----------------------------------------------------

def check_rephrase(row, llm: ChatOllama, logger: logging.Logger) -> Any:

    logger.info(f"Evaluating retrieval question for question {row["question_id"]}...")

    question = row["question"]
    retrieval_question = row["retrieval_question"]
    formatted_chat_history = row["formatted_chat_history"]

    prompt = f"""
    You are judging whether a rewritten question is a good
    standalone retrieval question.

    Original question:
    {question}

    Conversation history:
    {formatted_chat_history}

    Rewritten question:
    {retrieval_question}

    Set flag=True if the rewritten question:
    - preserves the original meaning,
    - correctly uses conversation history when needed,
    - is clear, concise, and retrieval-friendly,
    - or correctly keeps the original question unchanged if it is already clear.

    Set flag=False if the rewritten question:
    - changes the meaning,
    - adds unsupported assumptions,
    - drops important context,
    - becomes vague, misleading, or incorrect.

    Do not use outside knowledge.

    Return ONLY valid JSON:
    {{
      "think": "Short reasoning",
      "flag": <true_or_false>,
      "reason": "Brief reason"
    }}
    """

    try:
    
        structured_llm = llm.with_structured_output(JudgeResult)
    
        response = structured_llm.invoke(prompt)
        
    except Exception as e:

        response = JudgeResult(
            think="N/A",
            flag=False,
            reason=f"Error: {e}"
        )

    return response


def check_relevance(row, llm: ChatOllama, logger: logging.Logger) -> Any:

    logger.info(f"Evaluating relevance for question {row["question_id"]}...")

    question = row["question"]
    context = row["context"]

    prompt = f"""
    You are judging retrieval relevance for a RAG system.

    Question:
    {question}

    Retrieved context:
    {context}

    Set flag=True only if atleast one of the retrieved chunks in context contains enough relevant and specific information to answer most or all important parts of the question.

    Set flag=False if the context:
    - all chunks are irrelevant to the question,
    - all chunks miss major information needed to answer the question.

    Do not use outside knowledge.
    Judge relevance only based on the retrieved context and the question.

    Return ONLY valid JSON:
    {{
      "think": "Short reasoning about whether the retrieved context is sufficiently relevant and informative",
      "flag": <true_or_false>,
      "reason": "Brief reason"
    }}
    """

    try:
        
        # initialize structure llm with pydantic schema
        structured_llm = llm.with_structured_output(JudgeResult)
    
        # call llm
        response = structured_llm.invoke(prompt)

    except Exception as e:

        response = JudgeResult(
            think="N/A",
            flag=False,
            reason=f"Error: {e}"
        )

    return response


def check_groundedness(row, llm: ChatOllama, logger: logging.Logger) -> Any:

    logger.info(f"Evaluating groundedness for question {row["question_id"]}...")

    question = row["question"]
    predicted_answer = row["predicted_answer"]
    context = row["context"]

    prompt = f"""
    Judge if the predicted answer is fully cited and grounded in the retrieved context.

    Question:
    {question}

    Retrieved context:
    {context}

    Predicted answer:
    {predicted_answer}

    Set flag=True if:
    - the answer is a safe refusal or error message, OR
    - every factual claim has a citation and is supported by the retrieved context.

    Set flag=False if the predicted answer:
    - contains any factual claim that is missing a citation,
    - is unsupported or partially supported by the retrieved context,
    - contradicts the retrieved context,
    - or uses outside knowledge.

    Judge only the predicted answer. Do not penalize missing facts from the context.

    Return ONLY valid JSON:
    {{
      "think": "Brief support check",
      "flag": <true_or_false>,
      "reason": "Brief reason"
    }}
    """

    try:
    
        # initialize structure llm with pydantic schema
        structured_llm = llm.with_structured_output(JudgeResult)
    
        # call llm
        response = structured_llm.invoke(prompt)

    except Exception as e:

        response = JudgeResult(
            think="N/A",
            flag=False,
            reason=f"Error: {e}"
        )

    return response


def check_correctness(row, llm: ChatOllama, logger: logging.Logger) -> Any:

    logger.info(f"Evaluating correctness for question {row["question_id"]}...")

    question = row["question"]
    ground_truth_answer = row["ground_truth_answer"]
    predicted_answer = row["predicted_answer"]

    prompt = f"""
    You are judging answer correctness for a RAG system.

    Question:
    {question}

    Predicted answer:
    {predicted_answer}

    Ground truth answer:
    {ground_truth_answer}

    Set flag=True only if the predicted answer is factually consistent with the ground truth and covers the main expected points, even if wording differs.

    Set flag=False if the predicted answer:
    - contradicts the ground truth,
    - misses important information,
    - contains incorrect or misleading claims,
    - or is too vague to properly answer the question.

    Do not use outside knowledge.
    Judge correctness only by comparing the predicted answer against the ground truth answer.

    Return ONLY valid JSON:
    {{
      "think": "Short reasoning about how well the predicted answer matches the ground truth",
      "flag": <true_or_false>,
      "reason": "Brief reason"
    }}
    """

    try:
    
        # initialize structure llm with pydantic schema
        structured_llm = llm.with_structured_output(JudgeResult)
    
        # call llm
        response = structured_llm.invoke(prompt)

    except Exception as e:

        response = JudgeResult(
            think="N/A",
            flag=False,
            reason=f"Error: {e}"
        )

    return response


# ----------------------------------------------------
# generate plots
# ----------------------------------------------------

def generate_and_save_plots(
    df_evals: pd.DataFrame,
    evals_path: str
) -> None:
    """Generate and save evaluation metric plots.

    Creates:
        1. Overall evaluation metric plot.
        2. Group-level evaluation metric plot.

    Args:
        df_evals: Evaluation results dataframe.
        evals_path: Directory path to save generated plots.

    Returns:
        None

    """

    # -----------------------------------------
    # overall metrics dataframe
    # -----------------------------------------

    # create df
    df_metrics = (
            df_evals[['rephrase_flag', 'relevance_flag', 'groundedness_flag', 'correctness_flag']]
            .mean() * 100
    ).reset_index()
    # rename cols
    df_metrics.columns = ["metric", "value"]

    # -----------------------------------------
    # overall metrics plot
    # -----------------------------------------

    # Aggregate Plot
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.barplot(
        data=df_metrics,
        x="metric",
        y="value",
        ax=ax
    )
    ax.set_title("RAG Evaluation Metrics", fontsize=14)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(f"{evals_path}/overall_metrics.png")
    plt.close()

    # -----------------------------------------
    # group-level metrics dataframe
    # -----------------------------------------

    # create df
    df_group_metrics = (
        df_evals[['rephrase_flag', 'relevance_flag', 'groundedness_flag', 'correctness_flag', 'group']]
        .groupby(['group'])
        .mean()
        .reset_index()
    )
    # melt the df
    df_group_metrics = df_group_metrics.melt(id_vars='group', var_name='metric', value_name='proportion')
    # make score a percentage
    df_group_metrics['percentage'] = df_group_metrics['proportion'] * 100

    # -----------------------------------------
    # group-level metrics plot
    # -----------------------------------------

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.barplot(
        data=df_group_metrics,
        x="metric",
        y="percentage",
        hue='group',
        ax=ax
    )
    ax.set_title("RAG Evaluation Metrics", fontsize=14)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(f"{evals_path}/group_metrics.png")
    plt.close()

    return


# ----------------------------------------------------
# generate report
# ----------------------------------------------------

def generate_and_save_reports(
    df_evals: pd.DataFrame,
    evals_path: str
) -> None:

    for question_id in df_evals["question_id"]:

        # filter row
        row = df_evals[df_evals["question_id"] == question_id].iloc[0]

        # Create report text
        report = "\n".join([
            f"{'=' * 80}",
            f"QUESTION",
            f"{'=' * 80}",
            f"{row['question']}",
            "",

            f"{'=' * 80}",
            f"GROUP",
            f"{'=' * 80}",
            f"{row['group']}",
            "",

            f"{'=' * 80}",
            f"CHAT HISTORY",
            f"{'=' * 80}",
            f"{row['formatted_chat_history']}",
            "",

            f"{'=' * 80}",
            f"RETRIEVAL QUESTION",
            f"{'=' * 80}",
            f"{row['retrieval_question']}",
            "",

            f"{'=' * 80}",
            f"GROUND TRUTH",
            f"{'=' * 80}",
            f"{row['ground_truth_answer']}",
            "",

            f"{'=' * 80}",
            f"RETRIEVED CONTEXT",
            f"{'=' * 80}",
            f"{row['context']}",
            "",

            f"{'=' * 80}",
            f"PREDICTED ANSWER",
            f"{'=' * 80}",
            f"{row['predicted_answer']}",
            "",

            f"{'=' * 80}",
            f"SELF EVALUATION",
            f"{'=' * 80}",
            f"Relevance Check    : {row['relevance_check']}",
            f"Plan               : {row['plan']}",
            f"Groundedness Check : {row['groundedness_check']}",
            f"Citation Check     : {row['citation_check']}",
            f"Usefulness Check   : {row['usefulness_check']}",
            f"LLM Error          : {row['llm_error']}",
            "",

            f"{'=' * 80}",
            f"EVALUATION",
            f"{'=' * 80}",
            "",

            f"[Rephrase]",
            f"Think  : {row['rephrase_think']}",
            f"Flag   : {row['rephrase_flag']}",
            f"Reason : {row['rephrase_reason']}",
            "",

            f"[Relevance]",
            f"Think  : {row['relevance_think']}",
            f"Flag   : {row['relevance_flag']}",
            f"Reason : {row['relevance_reason']}",
            "",

            f"[Groundedness]",
            f"Think  : {row['groundedness_think']}",
            f"Flag   : {row['groundedness_flag']}",
            f"Reason : {row['groundedness_reason']}",
            "",

            f"[Correctness]",
            f"Think  : {row['correctness_think']}",
            f"Flag   : {row['correctness_flag']}",
            f"Reason : {row['correctness_reason']}",
        ])

        # save report
        save_path = os.path.join(evals_path, f"{question_id}.txt")
        with open(save_path, "w") as file:
            file.write(report)

    return


if __name__ == "__main__":

    # check evals
    check_evals(
        config_file_path="config/config.yaml",
        log_dir_path="logs"
    )
