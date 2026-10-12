import os
import joblib
import pandas as pd

from dotenv import load_dotenv
from groq import Groq

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

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

# =========================================================
# GROQ CLIENT
# =========================================================

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================================================
# STUDENT DATA MODEL
# =========================================================

class StudentData(BaseModel):

    gender: str

    branch: str

    study_hours: float

    attendance: float

    sleep_hours: float

    internet_usage: float

    assignments_completed: float

    previous_score: float

    extracurricular: str

    exam_score: float

# =========================================================
# HELPER FUNCTION
# =========================================================

def get_placed_probability(input_data: pd.DataFrame):

    """
    Returns the probability of the PLACED class.

    The trained model uses:
        0 = Not Placed
        1 = Placed
    """

    probabilities = model.predict_proba(
        input_data
    )[0]

    classes = list(
        model.classes_
    )

    if 1 not in classes:
        raise ValueError(
            "Placed class (1) was not found in the trained model."
        )

    placed_index = classes.index(1)

    return float(
        probabilities[placed_index]
    )

# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )

# =========================================================
# PREDICTION API
# =========================================================

@app.post("/predict")
def predict(data: StudentData):

    # -----------------------------------------------------
    # Convert input into DataFrame
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        [data.model_dump()]
    )

    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]

    # -----------------------------------------------------
    # Prediction result
    # -----------------------------------------------------

    result = (
        "Placed"
        if prediction == 1
        else "Not Placed"
    )

    # -----------------------------------------------------
    # Calculate probabilities
    # -----------------------------------------------------

    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)

    # Confidence of the class predicted by the model
    predicted_probability = float(
        probabilities[classes.index(prediction)]
    )

    # Probability specifically of PLACED
    placed_probability = get_placed_probability(
        input_data
    )

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

    try:

        completion = groq_client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4,

            max_tokens=1000
        )

        explanation = (
            completion
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        explanation = (
            "Your prediction has been generated successfully. "
            "The AI mentor explanation is temporarily unavailable. "
            "Please focus on consistent study, attendance, "
            "assignment completion and exam preparation."
        )

        print(
            "Groq error:",
            str(e)
        )

    # -----------------------------------------------------
    # API response
    # -----------------------------------------------------

    return {

        "prediction": result,

        # Confidence of the class predicted by the model
        "probability": round(
            predicted_probability * 100,
            2
        ),

        # Always means PLACED probability
        "placed_probability": round(
            placed_probability * 100,
            2
        ),

        "explanation": explanation
    }

# =========================================================
# IMPROVEMENT SIMULATOR
# =========================================================

@app.post("/simulate")
def simulate(data: StudentData):

    # -----------------------------------------------------
    # CURRENT STUDENT
    # -----------------------------------------------------

    current_data = data.model_dump()

    current_input = pd.DataFrame([current_data])

    current_prediction = model.predict(current_input)[0]

    current_placed_probability = (get_placed_probability(current_input))

    current_result = ("Placed" if current_prediction == 1 else "Not Placed")

    # -----------------------------------------------------
    # IF ALREADY PLACED
    # -----------------------------------------------------

    if current_prediction == 1:

        return {

            "current": {

                "prediction": "Placed",

                "placed_probability": round(
                    current_placed_probability * 100,
                    2
                )
            },

            "improved": {

                "prediction": "Placed",

                "placed_probability": round(
                    current_placed_probability * 100,
                    2
                )
            },

            "improvements": {

                "study_hours": {

                    "before":
                        current_data["study_hours"],

                    "after":
                        current_data["study_hours"]
                },

                "attendance": {

                    "before":
                        current_data["attendance"],

                    "after":
                        current_data["attendance"]
                },

                "assignments_completed": {

                    "before":
                        current_data[
                            "assignments_completed"
                        ],

                    "after":
                        current_data[
                            "assignments_completed"
                        ]
                },

                "exam_score": {

                    "before":
                        current_data["exam_score"],

                    "after":
                        current_data["exam_score"]
                }
            },


            "placed_after_improvement": True
        }

    # -----------------------------------------------------
    # BEST PLACED PROFILE
    # -----------------------------------------------------

    best_placed_profile = None

    best_placed_probability = 0.0

    best_improvement_cost = float(
        "inf"
    )

    # -----------------------------------------------------
    # BEST PROBABILITY PROFILE
    #
    # Used if no combination produces Placed.
    # -----------------------------------------------------

    best_probability_profile = (
        current_data.copy()
    )

    best_probability = (
        current_placed_probability
    )

    # -----------------------------------------------------
    # POSSIBLE ACADEMIC VALUES
    # -----------------------------------------------------

    study_values = range(

        int(
            current_data["study_hours"]
        ),

        13
    )

    attendance_values = range(

        int(
            current_data["attendance"]
        ),

        101,

        5
    )

    assignment_values = range(

        int(
            current_data[
                "assignments_completed"
            ]
        ),

        16
    )

    exam_values = range(

        int(
            current_data["exam_score"]
        ),

        101,

        5
    )

    # -----------------------------------------------------
    # SEARCH DIFFERENT COMBINATIONS
    # -----------------------------------------------------

    for study_hours in study_values:

        for attendance in attendance_values:

            for assignments in assignment_values:

                for exam_score in exam_values:

                    candidate = (
                        current_data.copy()
                    )

                    # Academic improvements
                    candidate[
                        "study_hours"
                    ] = study_hours

                    candidate[
                        "attendance"
                    ] = attendance

                    candidate[
                        "assignments_completed"
                    ] = assignments

                    candidate[
                        "exam_score"
                    ] = exam_score

                    candidate_input = pd.DataFrame(
                        [candidate]
                    )

                    # -------------------------------------------------
                    # Prediction
                    # -------------------------------------------------

                    prediction = model.predict(
                        candidate_input
                    )[0]

                    # -------------------------------------------------
                    # Placed probability
                    # -------------------------------------------------

                    placed_probability = (
                        get_placed_probability(
                            candidate_input
                        )
                    )

                    # -------------------------------------------------
                    # Track highest probability
                    # -------------------------------------------------

                    if (
                        placed_probability
                        > best_probability
                    ):

                        best_probability = (
                            placed_probability
                        )

                        best_probability_profile = (
                            candidate.copy()
                        )

                    # -------------------------------------------------
                    # Calculate improvement cost
                    #
                    # Smaller = less change required
                    # -------------------------------------------------

                    improvement_cost = (

                        (
                            study_hours
                            -
                            current_data[
                                "study_hours"
                            ]
                        )

                        +

                        (
                            attendance
                            -
                            current_data[
                                "attendance"
                            ]
                        ) / 5

                        +

                        (
                            assignments
                            -
                            current_data[
                                "assignments_completed"
                            ]
                        )

                        +

                        (
                            exam_score
                            -
                            current_data[
                                "exam_score"
                            ]
                        ) / 5
                    )

                    # -------------------------------------------------
                    # If this combination gives PLACED
                    # keep the smallest improvement
                    # -------------------------------------------------

                    if prediction == 1:

                        if (

                            improvement_cost
                            <
                            best_improvement_cost

                            or

                            (
                                improvement_cost
                                ==
                                best_improvement_cost

                                and

                                placed_probability
                                >
                                best_placed_probability
                            )
                        ):

                            best_placed_profile = (
                                candidate.copy()
                            )

                            best_placed_probability = (
                                placed_probability
                            )

                            best_improvement_cost = (
                                improvement_cost
                            )

    # -----------------------------------------------------
    # CHOOSE FINAL IMPROVED PROFILE
    # -----------------------------------------------------

    if best_placed_profile is not None:

        improved_data = (
            best_placed_profile
        )

        improved_probability = (
            best_placed_probability
        )

        improved_result = "Placed"

        placed_after_improvement = True

    else:

        # No realistic tested combination produced
        # a Placed prediction.

        improved_data = (best_probability_profile)

        improved_probability = (best_probability)

        improved_input = pd.DataFrame(
            [improved_data]
        )

        improved_prediction = model.predict(improved_input)[0]

        improved_result = ("Placed" if improved_prediction == 1 else "Not Placed")

        placed_after_improvement = False

    # -----------------------------------------------------
    # RETURN SIMULATION
    # -----------------------------------------------------

    return {

        "current": {

            "prediction": current_result,

            "placed_probability": round(
                current_placed_probability * 100,
                2
            )
        },

        "improved": {

            "prediction": improved_result,

            "placed_probability": round(
                improved_probability * 100,
                2
            )
        },

        "improvements": {

            "study_hours": {

                "before":
                    current_data[
                        "study_hours"
                    ],

                "after":
                    improved_data[
                        "study_hours"
                    ]
            },

            "attendance": {

                "before":
                    current_data[
                        "attendance"
                    ],

                "after":
                    improved_data[
                        "attendance"
                    ]
            },

            "assignments_completed": {

                "before":
                    current_data[
                        "assignments_completed"
                    ],

                "after":
                    improved_data[
                        "assignments_completed"
                    ]
            },

            "exam_score": {

                "before":
                    current_data[
                        "exam_score"
                    ],

                "after":
                    improved_data[
                        "exam_score"
                    ]
            }
        },

        "placed_after_improvement":
            placed_after_improvement
    }