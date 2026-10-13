import os
import math

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from groq import Groq

from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="Student Placement Prediction API")

# =========================================================
# FRONTEND
# =========================================================

templates = Jinja2Templates(directory="templates")

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("placement_model.joblib")

# Column order the model was trained with (used when batching)
MODEL_CLASSES = list(model.classes_)

if 1 not in MODEL_CLASSES:
    raise RuntimeError("Placed class (1) was not found in the trained model.")

PLACED_INDEX = MODEL_CLASSES.index(1)

# =========================================================
# GROQ CLIENT
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If the key is missing the app still starts; only the AI mentor is disabled.
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

FALLBACK_EXPLANATION = (
    "Your prediction has been generated successfully. "
    "The AI mentor explanation is temporarily unavailable. "
    "Please focus on consistent study, attendance, "
    "assignment completion and exam preparation."
)

# =========================================================
# STUDENT DATA MODEL (with validation)
# =========================================================


class StudentData(BaseModel):

    gender: str

    branch: str

    study_hours: float = Field(ge=0, le=24)

    attendance: float = Field(ge=0, le=100)

    sleep_hours: float = Field(ge=0, le=24)

    internet_usage: float = Field(ge=0)

    assignments_completed: float = Field(ge=0)

    previous_score: float = Field(ge=0, le=100)

    extracurricular: str

    exam_score: float = Field(ge=0, le=100)


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def get_placed_probability(input_data: pd.DataFrame) -> float:
    """
    Returns the probability of the PLACED class (1).
    """
    probabilities = model.predict_proba(input_data)[0]
    return float(probabilities[PLACED_INDEX])


def build_range(current: float, cap: int, step: int = 1) -> list:
    """
    Values from the student's current value (rounded UP, so we never
    suggest a decrease) to the cap. If the current value is already
    at or above the cap, only the current value is used.
    """
    start = math.ceil(current)

    if start > cap:
        return [current]

    values = list(range(start, cap + 1, step))

    # Make sure the cap itself is always tested
    if values[-1] != cap:
        values.append(cap)

    return values


def improvement_block(before: dict, after: dict) -> dict:
    keys = [
        "study_hours",
        "attendance",
        "assignments_completed",
        "exam_score",
    ]
    return {
        k: {"before": before[k], "after": after[k]}
        for k in keys
    }


# =========================================================
# HOME PAGE
# =========================================================


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


# Browsers automatically ask for a favicon; return "no content"
# instead of logging a 404 every time.
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# =========================================================
# PREDICTION API
# =========================================================


@app.post("/predict")
def predict(data: StudentData):

    input_data = pd.DataFrame([data.model_dump()])

    # Single model call for probabilities, derive everything from it
    probabilities = model.predict_proba(input_data)[0]
    predicted_index = int(np.argmax(probabilities))
    prediction = MODEL_CLASSES[predicted_index]

    result = "Placed" if prediction == 1 else "Not Placed"

    # Confidence of the class predicted by the model
    predicted_probability = float(probabilities[predicted_index])

    # Probability specifically of PLACED
    placed_probability = float(probabilities[PLACED_INDEX])

    # -----------------------------------------------------
    # Groq AI Mentor
    # -----------------------------------------------------

    prompt = f"""
Act like a caring and experienced college placement mentor.

Talk directly to the student like a real human teacher.

Your goal is to explain the prediction honestly while keeping
the student motivated and mentally positive.

Prediction: {result}

Placed probability: {placed_probability * 100:.2f}%

Student details:

- Study hours: {data.study_hours}
- Attendance: {data.attendance}%
- Sleep hours: {data.sleep_hours}
- Internet usage: {data.internet_usage}
- Assignments completed: {data.assignments_completed}
- Previous score: {data.previous_score}
- Exam score: {data.exam_score}
- Extracurricular activities: {data.extracurricular}
- Branch: {data.branch}

Do NOT explain:

- Decision Trees
- machine learning algorithms
- thresholds
- branches of the tree
- encoded values
- feature names
- technical model rules

Instead, explain the student's academic situation
in simple, natural language.

If the prediction is "Not Placed":

- Never say the student is a failure.
- Never say the student is incapable.
- Never say they have no chance.
- Never shame or blame the student.
- Explain which academic areas currently need attention.
- Connect study habits, attendance, assignments and scores naturally.
- Explain that these areas can improve.
- Give 2-3 practical and achievable actions.
- End with an encouraging message.

If the prediction is "Placed":

- Explain the positive academic habits contributing to the prediction.
- Encourage the student to maintain those habits.
- Mention that continued preparation is still important.
- Do not guarantee an actual job or placement.

Use a warm and supportive tone.

Avoid statements such as:

"You will not get a job."
"You are weak."
"You are not capable."
"You are a failure."
"You have no chance."

Use phrases such as:

"Right now, you have a few areas to work on."
"These are things you can improve."
"Don't look at this prediction as a final result."
"With consistent effort, you can strengthen your profile."

Do not guarantee future placement.

Keep the response around 120-160 words.

Use this format:

## Your result

## What I noticed

## What you can work on

## A message for you
"""

    explanation = FALLBACK_EXPLANATION

    if groq_client is not None:
        try:
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                # Reasoning tokens count toward this limit, so keep it
                # generous and keep reasoning light.
                max_completion_tokens=1500,
                reasoning_effort="low",
            )

            content = completion.choices[0].message.content

            if content and content.strip():
                explanation = content

        except Exception as e:
            print("Groq error:", str(e))
    else:
        print("Groq error: GROQ_API_KEY is not set.")

    return {
        "prediction": result,

        # Confidence of the class predicted by the model
        "probability": round(predicted_probability * 100, 2),

        # Always means PLACED probability
        "placed_probability": round(placed_probability * 100, 2),

        "explanation": explanation,
    }


# =========================================================
# IMPROVEMENT SIMULATOR
# =========================================================


@app.post("/simulate")
def simulate(data: StudentData):

    current_data = data.model_dump()
    current_input = pd.DataFrame([current_data])

    current_probabilities = model.predict_proba(current_input)[0]
    current_prediction = MODEL_CLASSES[int(np.argmax(current_probabilities))]
    current_placed_probability = float(current_probabilities[PLACED_INDEX])
    current_result = "Placed" if current_prediction == 1 else "Not Placed"

    # -----------------------------------------------------
    # IF ALREADY PLACED -> nothing to improve
    # -----------------------------------------------------

    if current_prediction == 1:
        return {
            "current": {
                "prediction": "Placed",
                "placed_probability": round(current_placed_probability * 100, 2),
            },
            "improved": {
                "prediction": "Placed",
                "placed_probability": round(current_placed_probability * 100, 2),
            },
            "improvements": improvement_block(current_data, current_data),
            "placed_after_improvement": True,
        }

    # -----------------------------------------------------
    # BUILD ALL CANDIDATE COMBINATIONS
    # -----------------------------------------------------

    study_values = build_range(current_data["study_hours"], 12)
    attendance_values = build_range(current_data["attendance"], 100, 5)
    assignment_values = build_range(current_data["assignments_completed"], 15)
    exam_values = build_range(current_data["exam_score"], 100, 5)

    # np.meshgrid builds every combination without Python loops
    grid = np.array(
        np.meshgrid(
            study_values,
            attendance_values,
            assignment_values,
            exam_values,
            indexing="ij",
        )
    ).reshape(4, -1).T.astype(float)

    changing_cols = [
        "study_hours",
        "attendance",
        "assignments_completed",
        "exam_score",
    ]

    candidates = pd.DataFrame([current_data] * len(grid))
    candidates[changing_cols] = grid
    candidates = candidates[current_input.columns]  # keep training column order

    # -----------------------------------------------------
    # ONE batched prediction for all candidates
    # -----------------------------------------------------

    probabilities = model.predict_proba(candidates)
    predictions = np.array(MODEL_CLASSES)[np.argmax(probabilities, axis=1)]
    placed_probs = probabilities[:, PLACED_INDEX]

    # Improvement cost: smaller = less change required
    cost = (
        (grid[:, 0] - current_data["study_hours"])
        + (grid[:, 1] - current_data["attendance"]) / 5
        + (grid[:, 2] - current_data["assignments_completed"])
        + (grid[:, 3] - current_data["exam_score"]) / 5
    )

    placed_mask = predictions == 1

    if placed_mask.any():
        # Lowest cost wins; ties broken by highest placed probability
        placed_idx = np.flatnonzero(placed_mask)
        order = np.lexsort((-placed_probs[placed_idx], cost[placed_idx]))
        best = int(placed_idx[order[0]])
        improved_result = "Placed"
        placed_after_improvement = True
    else:
        # No tested combination produced Placed -> show the best probability
        best = int(np.argmax(placed_probs))
        improved_result = "Placed" if predictions[best] == 1 else "Not Placed"
        placed_after_improvement = False

    improved_data = current_data.copy()

    for col, value in zip(changing_cols, grid[best]):
        improved_data[col] = float(value)

    improved_probability = float(placed_probs[best])

    # -----------------------------------------------------
    # RETURN SIMULATION
    # -----------------------------------------------------

    return {
        "current": {
            "prediction": current_result,
            "placed_probability": round(current_placed_probability * 100, 2),
        },
        "improved": {
            "prediction": improved_result,
            "placed_probability": round(improved_probability * 100, 2),
        },
        "improvements": improvement_block(current_data, improved_data),
        "placed_after_improvement": placed_after_improvement,
    }