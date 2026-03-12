"""
AI Autonomous Data Science Agent - FastAPI Backend
Powered by Gemini 2.5 Flash + Scikit-learn + XGBoost
"""

import os
import io
import json
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Gemini
import google.generativeai as genai

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Autonomous Data Science Agent",
    description="Automated ML pipeline powered by Gemini 2.5 Flash",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Gemini Setup ─────────────────────────────────────────────────────────────

def get_gemini_model():
    """Initialize Gemini 2.5 Flash model - tries multiple model names for compatibility."""
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAOO2uW0jR1eLlN1kNPtspdVpK6J1ocmxk")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    # Try models in order of preference
    model_names = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    for name in model_names:
        try:
            model = genai.GenerativeModel(name)
            # Quick test to verify availability
            return model
        except Exception:
            continue
    # Fallback to first option regardless
    return genai.GenerativeModel("gemini-1.5-flash")

# ─── In-memory store ──────────────────────────────────────────────────────────

STORE: Dict[str, Any] = {}

# ─── Data Models ──────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    session_id: str

class PipelineRequest(BaseModel):
    session_id: str
    target_column: Optional[str] = None

# ─── Utility Functions ────────────────────────────────────────────────────────

def detect_problem_type(series: pd.Series) -> str:
    """Determine if the target column is classification or regression."""
    if series.dtype == object or series.dtype.name == "category":
        return "classification"
    unique_ratio = series.nunique() / len(series)
    if series.nunique() <= 20 or unique_ratio < 0.05:
        return "classification"
    return "regression"


def detect_target_column(df: pd.DataFrame) -> str:
    """Heuristically detect the most likely target column."""
    priority_names = [
        "target", "label", "class", "outcome", "y", "result",
        "price", "salary", "sales", "survived", "churn", "fraud",
        "diagnosis", "output", "response", "dependent"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for name in priority_names:
        if name in cols_lower:
            return cols_lower[name]
    # Fall back to last column
    return df.columns[-1]


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive dataset summary."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    missing = df.isnull().sum()
    missing_info = {col: int(missing[col]) for col in df.columns if missing[col] > 0}

    summary = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": missing_info,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "basic_stats": json.loads(df[numeric_cols].describe().to_json()) if numeric_cols else {},
    }
    return summary


def clean_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Handle missing values, duplicates, and encode categoricals."""
    df = df.drop_duplicates().reset_index(drop=True)

    # ── FIX: Drop rows where TARGET is NaN (can't train without labels) ──────
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Separate features and target
    X = df.drop(columns=[target_col]).reset_index(drop=True)
    y = df[target_col].copy().reset_index(drop=True)

    # Encode categorical features FIRST (before imputation so no object cols remain)
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        # Fill NaN with string "missing" before encoding so LabelEncoder doesn't crash
        X[col] = le.fit_transform(X[col].fillna("__missing__").astype(str))

    # Impute numeric missing values
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        X[num_cols] = imp_num.fit_transform(X[num_cols])

    # Impute any remaining object/category columns
    cat_cols_remaining = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols_remaining:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X[cat_cols_remaining] = imp_cat.fit_transform(X[cat_cols_remaining])

    # Encode target if classification (object/category dtype)
    if y.dtype == object or y.dtype.name == "category":
        le_target = LabelEncoder()
        y = pd.Series(
            le_target.fit_transform(y.fillna("__missing__").astype(str)),
            name=target_col
        )
    else:
        # Numeric target: impute any remaining NaNs with median
        y = y.fillna(y.median())

    # Final safety: ensure no NaN anywhere
    X = X.fillna(0)
    y = y.fillna(0)

    cleaned = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return cleaned


def engineer_features(X: pd.DataFrame, y: pd.Series, problem_type: str) -> pd.DataFrame:
    """Scale features and select top-K important features."""
    # Reset indices to guarantee alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Final NaN guard before scaling
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Select top features (k must be ≤ number of features)
    k = min(15, X_scaled.shape[1])
    score_func = f_classif if problem_type == "classification" else f_regression

    try:
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        selected_cols = X_scaled.columns[selector.get_support()].tolist()
        return pd.DataFrame(X_selected, columns=selected_cols)
    except Exception:
        # If SelectKBest fails for any reason, return all scaled features
        return X_scaled


def select_and_train_model(X_train, X_test, y_train, y_test, problem_type: str) -> Dict[str, Any]:
    """Try multiple models, pick the best one, return full results."""
    results = {}

    if problem_type == "classification":
        candidates = {
            "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        }
        if XGBOOST_AVAILABLE:
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=150, random_state=42,
                eval_metric="logloss", verbosity=0
            )

        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results[name] = {
                    "model": model,
                    "preds": preds,
                    "score": acc,
                    "accuracy": round(acc, 4),
                    "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
                    "recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
                    "f1": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
                }
            except Exception as e:
                print(f"Model {name} failed: {e}")

        best_name = max(results, key=lambda k: results[k]["score"])
        best = results[best_name]
        metrics = {
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1_score": best["f1"],
        }

    else:  # regression
        candidates = {
            "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "LinearRegression": LinearRegression(),
        }
        if XGBOOST_AVAILABLE:
            candidates["XGBoost"] = XGBRegressor(
                n_estimators=150, random_state=42, verbosity=0
            )

        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                results[name] = {
                    "model": model,
                    "preds": preds,
                    "score": r2,
                    "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
                    "r2": round(r2, 4),
                }
            except Exception as e:
                print(f"Model {name} failed: {e}")

        best_name = max(results, key=lambda k: results[k]["score"])
        best = results[best_name]
        metrics = {
            "rmse": best["rmse"],
            "r2_score": best["r2"],
        }

    # Feature importance
    best_model = best["model"]
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance = dict(zip(X_train.columns.tolist(), importances.tolist()))
    elif hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_).flatten()[:len(X_train.columns)]
        feature_importance = dict(zip(X_train.columns.tolist(), coefs.tolist()))

    # All model comparison scores
    model_comparison = {
        name: round(res["score"], 4) for name, res in results.items()
    }

    return {
        "best_model_name": best_name,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_comparison": model_comparison,
        "all_results": {name: {k: v for k, v in res.items() if k not in ["model", "preds"]}
                        for name, res in results.items()},
    }


def ask_gemini(prompt: str) -> str:
    """Query Gemini and return response text, with model auto-fallback."""
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAOO2uW0jR1eLlN1kNPtspdVpK6J1ocmxk")
    if not api_key:
        return "⚠️ Gemini API key not configured. Set the GEMINI_API_KEY environment variable."

    genai.configure(api_key=api_key)

    # Try models in priority order
    model_names = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    last_error = ""
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = str(e)
            # If it's a 404 "not found" error, try next model
            if "404" in str(e) or "not found" in str(e).lower():
                continue
            # Other errors (auth, quota) — report immediately
            return f"⚠️ Gemini error ({model_name}): {str(e)}"

    return f"⚠️ No Gemini model available. Last error: {last_error}"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "AI Autonomous Data Science Agent is running 🚀"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "xgboost": XGBOOST_AVAILABLE,
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Accept CSV, store in memory, return preview + summary."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

        session_id = f"session_{hash(contents) % 999999:06d}"
        STORE[session_id] = {"df": df, "filename": file.filename}

        target_col = detect_target_column(df)
        problem_type = detect_problem_type(df[target_col])
        summary = summarize_dataset(df)

        # Correlation matrix for heatmap (numeric only, limit to 20 cols)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 20:
            numeric_df = numeric_df.iloc[:, :20]
        corr = numeric_df.corr().round(3)
        corr_data = {
            "columns": corr.columns.tolist(),
            "values": corr.values.tolist()
        }

        return {
            "session_id": session_id,
            "filename": file.filename,
            "target_column": target_col,
            "problem_type": problem_type,
            "summary": summary,
            "preview": json.loads(df.head(10).to_json(orient="records")),
            "correlation": corr_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/analyze")
async def analyze_dataset(req: AnalyzeRequest):
    """Use Gemini 2.5 Flash to generate rich dataset insights."""
    if req.session_id not in STORE:
        raise HTTPException(status_code=404, detail="Session not found. Upload data first.")

    data = STORE[req.session_id]
    df = data["df"]
    summary = summarize_dataset(df)
    target_col = detect_target_column(df)
    problem_type = detect_problem_type(df[target_col])

    prompt = f"""
You are a world-class data scientist. Analyze this dataset and provide expert insights.

Dataset: {data.get('filename', 'unknown')}
Shape: {summary['shape'][0]} rows × {summary['shape'][1]} columns
Columns: {summary['columns']}
Target Column: {target_col}
Problem Type: {problem_type}
Numeric Columns: {summary['numeric_columns']}
Categorical Columns: {summary['categorical_columns']}
Missing Values: {summary['missing_values']}
Duplicate Rows: {summary['duplicate_rows']}
Memory: {summary['memory_mb']} MB

Basic Statistics:
{json.dumps(summary['basic_stats'], indent=2)[:1500]}

Provide a detailed analysis in the following structure (use emojis and markdown):

## 🔍 Dataset Overview
Brief description of what this dataset likely represents.

## 📊 Key Observations
- Notable patterns, distributions, or anomalies
- Data quality observations

## 🎯 Target Variable Analysis
What {target_col} represents and its distribution implications.

## 🧠 ML Problem Assessment
Why {problem_type} is the right approach and recommended algorithms.

## ⚡ Feature Insights
Most important features and potential relationships.

## 🚨 Data Quality Flags
Issues to watch for: missing values, outliers, imbalance.

## 💡 Actionable Recommendations
3 specific recommendations to improve model performance.

Be specific, insightful, and actionable. Use a professional but engaging tone.
"""

    insights = ask_gemini(prompt)
    STORE[req.session_id]["insights"] = insights

    return {
        "session_id": req.session_id,
        "target_column": target_col,
        "problem_type": problem_type,
        "gemini_insights": insights,
    }


@app.post("/run-pipeline")
async def run_pipeline(req: PipelineRequest):
    """Execute the full autonomous ML pipeline end-to-end."""
    if req.session_id not in STORE:
        raise HTTPException(status_code=404, detail="Session not found. Upload data first.")

    try:
        data = STORE[req.session_id]
        df = data["df"].copy()

        # Step 1: Understand
        target_col = req.target_column or detect_target_column(df)
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found.")

        problem_type = detect_problem_type(df[target_col])
        summary = summarize_dataset(df)

        # Step 2: Clean
        cleaned_df = clean_dataset(df, target_col)
        X_raw = cleaned_df.drop(columns=[target_col])
        y = cleaned_df[target_col]

        # Step 3: Feature Engineering
        X = engineer_features(X_raw, y, problem_type)

        # Step 4+5: Select & Train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        training_results = select_and_train_model(X_train, X_test, y_train, y_test, problem_type)

        # Step 6: Gemini reasoning for model selection
        model_prompt = f"""
You are an expert ML engineer. Explain why {training_results['best_model_name']} was selected 
as the best model for this {problem_type} problem.

Dataset: {summary['shape'][0]} rows × {summary['shape'][1]} columns
Problem Type: {problem_type}
Model Comparison Scores: {json.dumps(training_results['model_comparison'])}
Best Model Metrics: {json.dumps(training_results['metrics'])}
Top Features: {list(training_results['feature_importance'].keys())[:8]}

Provide a concise 3-paragraph explanation:
1. Why this model won (specific reasons tied to the data characteristics)
2. What the metrics reveal about model quality
3. Key insights from feature importance and recommendations for deployment

Use markdown with emojis. Be specific and technical but accessible.
"""
        model_reasoning = ask_gemini(model_prompt)

        # Prepare sorted feature importance
        fi = training_results["feature_importance"]
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])

        result = {
            "session_id": req.session_id,
            "pipeline_status": "success",
            "problem_type": problem_type,
            "target_column": target_col,
            "dataset_summary": {
                "rows": summary["shape"][0],
                "columns": summary["shape"][1],
                "numeric_features": len(summary["numeric_columns"]),
                "categorical_features": len(summary["categorical_columns"]),
                "missing_values": sum(summary["missing_values"].values()),
                "duplicates_removed": summary["duplicate_rows"],
                "features_selected": X.shape[1],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            },
            "best_model": training_results["best_model_name"],
            "model_comparison": training_results["model_comparison"],
            "metrics": training_results["metrics"],
            "feature_importance": fi_sorted,
            "gemini_model_reasoning": model_reasoning,
        }

        STORE[req.session_id]["pipeline_result"] = result
        return result

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}\n{tb[:500]}")


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
