import logging
import os
import warnings
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()

class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        raw_df = pd.read_csv(filepath, low_memory=False)
        df = self.preprocess_dataset(raw_df)

        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)
        
        self.response = response
        return None
    
    def preprocess_dataset(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Pre process and normalize and format aspects of the dataset to be more conducive to analysis.
        TODO: Deal with columns that have long text strings. One idea is simply to change or create a new column that is a boolean value (has text vs no text)
        """
        df = raw_df.copy()
        # Normalize the column names to be lowercase and replace spaces with underscores.
        df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        # getting categorical columns to determine if there are any currency symbols in columns that need to be converted to float values
        # These will start off categorical but we want to convert them to numerical values.
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            # make sure its not a boolean column and that more than 90% of the values contain $ signs. 90% is arbitrary but seems like a good threshold.
            try:
                if df[col].str.contains(r"\$", regex=True, na=False).mean() > .9:
                    df[col] = df[col].str.replace(r"[\$,\s]", "", regex=True).astype(float) 
                    col_usd = col + "_usd"
                    df = df.rename(columns={col: col_usd})
            except:
                pass

        # Leaving this filtering in place for now.
        if df.shape[0] > 10000:
            return df.sample(10000).reset_index(drop=True)
        else:
            return df
        
        # return df

    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        categorical_columns: dict
        numeric_columns: list[str]
        recommendations: list[str]

    def describe_add_skew(df: pd.DataFrame) -> dict:
        """ Helper function to add skew to describe output."""
        described = df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
        described['skew'] = df.skew(numeric_only=True) # should only be passing in numeric columns but adding anyways
        return described.T.to_dict()

    def filter_numeric_columns(df: pd.DataFrame) -> list[str]:
        """Filter the numeric columns in the dataset."""
        # numeric columns are likely to be unique identifiers or data that are not useful for analysis.
        # zipcode/postal code are exceptions but these should actually be analyzed as categorical columns but not going to deal with these for now.
        columns_to_ignore = ["id", "index", "row_number", "row_id", "row_index", "row_number_id", "row_number_index", "row_number_index_id", "latitude", 
            "longitude", "lng", "lon", "long", "lat", "zip_code", "zipcode", "postal_code", "postcode", "postal_code_id", "phone_number", "host_id", "license"]
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if ((col.lower() not in columns_to_ignore) and (df[col].isna().all() == False))]
        return numeric_cols if numeric_cols else []

    def filter_categorical_columns(df: pd.DataFrame) -> dict:
        """Filter the categorical columns in the dataset."""
        # TODO: treat zipcode/postal code as categorical columns. Also figure out what to do about columns that contain items like phone number/email addresses where it may be null or missing.
        all_categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Remove categorical columns that are unique for all rows as they are likely a primary key or a unique identifier.
        # Initialize a dictionary to store the categorical columns.
        categorical_cols = {}
        categorical_cols["datetime_cols"] = []
        categorical_cols["cols_to_ignore"] = []
        categorical_cols["cols_to_analyze"] = []
        categorical_cols["cols_for_nunique"] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in all_categorical_cols:
                num_unique = df[col].nunique()
                try:
                    if df[col].isna().all(): # a completely null column will successfully convert to datetime so we need to ignore these.
                        categorical_cols["cols_to_ignore"].append(col)
                    else:
                        # attempt dt conversion.  No actual need to convert to dt as structure may not hold. 
                        pd.to_datetime(df[col])
                        categorical_cols["datetime_cols"].append(col)
                except:
                    if (num_unique <= 1) or (num_unique/df.shape[0] >= 0.1):
                        categorical_cols["cols_to_ignore"].append(col)
                    elif num_unique <= 20: # 2 is binary usually boolean, the up to 20 is completely arbitrary
                        categorical_cols["cols_to_analyze"].append(col)
                    else:
                        categorical_cols["cols_for_nunique"].append(col)
                    pass

        cardinality = {col: df[col].nunique() for col in categorical_cols["cols_to_analyze"]}
        # Sort the categorical columns by cardinality (low to high) as a heuristic for the most interesting columns to analyze.
        categorical_cols["cols_to_analyze"] = sorted(cardinality, key=cardinality.get)
        return categorical_cols if categorical_cols else {}

    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        categorical_cols = filter_categorical_columns(df)        
        numeric_cols = filter_numeric_columns(df)
        subset_keys = ["cols_to_analyze", "cols_for_nunique"]
        subset_flattened = [col for key in subset_keys if key in categorical_cols for col in categorical_cols[key]]
        ignore_keys = ["cols_to_ignore", "datetime_cols"]
        ignore_flattened = [col for key in ignore_keys if key in categorical_cols for col in categorical_cols[key]]
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": subset_flattened,
            "categorical_columns_to_ignore": ignore_flattened,
            "numeric_summary": (
                describe_add_skew(df[numeric_cols]) if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(20).to_dict() for col in subset_flattened
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
            "categorical_columns": categorical_cols,
            "numeric_columns": numeric_cols,
        }
    
    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        # TODO: Segment Data based on a few categorical columns, and check missingness by segment
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )
        
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }
    
    def flag_outliers_node(state: EDAState):
        """Flag outliers in the dataset via returning the row and the reason for flagging."""
        logger.info("Flagging outliers")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        numeric_cols = state.get("numeric_columns", [])
        # Compute upper and lower bounds for each numeric column based on IQR (Interquartile Range)
        # Create a new dataframe with the outliers and the reason for flagging.
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            lb_df = df[df[col] < lower_bound].copy()
            lb_df['Outlier Reason'] = col + ' is a low outlier'
            ub_df = df[df[col] > upper_bound].copy()
            ub_df['Outlier Reason'] = col + ' is a high outlier'
            outlier_df = pd.concat([lb_df, ub_df])        
        outliers = {
            "outlier_df": outlier_df.to_dict() if not outlier_df.empty else {},
            "outlier_count": len(outlier_df),
            "outlier_percentage": len(outlier_df) / len(df) * 100,
        }
        results["flag_outliers"] = outliers

        return {
            "current_step": "flag_outliers",
            "results": results,
        }

    def check_skewness_node(state: EDAState):
        """Check the skew of the numeric columns in the dataset. This is included in the profile_dataset_node but the LLM isn't picking up on it."""
        logger.info("Checking skewness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        numeric_cols = state.get("numeric_columns", [])
        # Pandas calculates the unbiased Fisher-Pearson coefficient of skewness.
        skewness = {
            "skewness": df[numeric_cols].skew().to_dict(),
        }
        results["check_skewness"] = skewness

        return {
            "current_step": "check_skewness",
            "results": results,
        }

    def compute_aggregates_node(state: EDAState):
        """Compute group-by aggregates on key columns"""
        logger.info("Computing aggregates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        numeric_cols = state.get("numeric_columns", [])
        categorical_cols = state.get("categorical_columns", {}) # these are already ordered by cardinality (low to high)
        aggregates = {
            "global": df[numeric_cols].agg(["count", "mean", "median", "std", "min", "max", "sum", "var"]).to_dict(),
        }
        # Compute the aggregates for categorical columns according to cardinality.
        # Limit to the top N values for each categorical column.
        for col in categorical_cols["cols_to_analyze"]:
            aggregates[col] = df.groupby(col)[numeric_cols].agg(["count", "mean", "median", "std", "min", "max", "sum", "var"]).head(20).to_dict()
        for col in categorical_cols["cols_for_nunique"]:
            aggregates[col] = df[col].value_counts().head(20).to_dict()
        results["compute_aggregates"] = aggregates

        return {
            "current_step": "compute_aggregates",
            "results": results,
        }
    
    def analyze_relationships_node(state: EDAState):
        """Analyze relationships between variables.
        Correlation matrix of the numeric columns. 
        Visualizations of the relationships between variables don't work well with the current setup.
        """
        logger.info("Analyzing relationships")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        numeric_cols = state.get("numeric_columns", [])
        relationships = {
            "correlation_matrix": df[numeric_cols].corr().to_dict(),
        }

        results["analyze_relationships"] = relationships

        return {
            "current_step": "analyze_relationships",
            "results": results,
        }
    
    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations}
        
        step_results = results.get(current_step, {})
        
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": str(step_results)
        })
        
        observations[current_step] = response.observations
        
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {})
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")
        
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
        
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("flag_outliers", flag_outliers_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("check_skewness", check_skewness_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_5", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_6", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "flag_outliers")
    workflow.add_edge("flag_outliers", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "check_skewness")
    workflow.add_edge("check_skewness", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_5")
    workflow.add_edge("extract_observations_5", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_6")
    workflow.add_edge("extract_observations_6", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app
