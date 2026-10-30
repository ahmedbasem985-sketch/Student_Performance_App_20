import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Dashboard")
st.caption("Student Data Analysis | Statistics | Visualization")

uploaded_file = st.file_uploader(
    "Upload Student Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload Student_Performance_Dataset.xlsx")
    st.stop()

try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

st.success("Dataset loaded successfully!")

# Data cleaning
cleaned_df = df.drop_duplicates().copy()

for col in cleaned_df.select_dtypes(include="number").columns:
    cleaned_df[col] = cleaned_df[col].fillna(
        cleaned_df[col].median()
    )

for col in cleaned_df.select_dtypes(
    include=["object", "category"]
).columns:
    mode = cleaned_df[col].mode()
    if not mode.empty:
        cleaned_df[col] = cleaned_df[col].fillna(mode.iloc[0])

df = cleaned_df

# Dashboard metrics
st.header("📊 Dashboard Overview")

total_students = len(df)

if "final_score" in df.columns:
    scores = pd.to_numeric(df["final_score"], errors="coerce")
    average_score = scores.mean()
    pass_rate = (scores.ge(50).mean() * 100)
else:
    scores = pd.Series(dtype=float)
    average_score = None
    pass_rate = None

if "attendance" in df.columns:
    attendance = pd.to_numeric(df["attendance"], errors="coerce")
    average_attendance = attendance.mean()
else:
    average_attendance = None

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Students", total_students)
c2.metric(
    "Average Score",
    f"{average_score:.1f}" if pd.notna(average_score) else "N/A"
)
c3.metric(
    "Average Attendance",
    f"{average_attendance:.1f}%" 
    if pd.notna(average_attendance) else "N/A"
)
c4.metric(
    "Pass Rate",
    f"{pass_rate:.1f}%" if pass_rate is not None else "N/A"
)

st.divider()

# Data preview
with st.expander("👀 View Student Data"):
    st.dataframe(df, use_container_width=True)

# Statistics
st.header("📈 Statistical Analysis")

numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:
    st.subheader("Descriptive Statistics")
    st.dataframe(
        numeric_df.describe().round(2),
        use_container_width=True
    )

# Score distribution
if "final_score" in df.columns:
    st.subheader("🎯 Final Score Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(scores.dropna(), bins=15, kde=True, ax=ax)
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Final Scores")
    st.pyplot(fig)
    plt.close(fig)

# Study hours versus score
if {"study_hours", "final_score"}.issubset(df.columns):
    st.subheader("📚 Study Hours vs Final Score")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.regplot(
        data=df,
        x="study_hours",
        y="final_score",
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "red"},
        ax=ax
    )
    ax.set_title("Relationship Between Study Hours and Final Score")
    st.pyplot(fig)
    plt.close(fig)

# Correlation heatmap
if len(numeric_df.columns) >= 2:
    st.subheader("🔥 Correlation Heatmap")

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Correlation Between Numeric Variables")
    st.pyplot(fig)
    plt.close(fig)

# Participation analysis
if {"participation", "final_score"}.issubset(df.columns):
    st.subheader("🧑‍🎓 Average Score by Participation")

    participation_scores = df.groupby(
        "participation"
    )["final_score"].mean().sort_values(ascending=False)

    st.bar_chart(participation_scores)

# Download results
st.header("📥 Export Results")

output = BytesIO()

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Cleaned Data", index=False)

    if not numeric_df.empty:
        numeric_df.describe().T.to_excel(
            writer,
            sheet_name="Statistics"
        )
        numeric_df.corr().to_excel(
            writer,
            sheet_name="Correlation"
        )

st.download_button(
    label="Download Excel Report",
    data=output.getvalue(),
    file_name="Student_Performance_Analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Students who may need academic support
st.header("🎯 Student Support Analysis")

if "final_score" in df.columns:
    support_students = df[
        pd.to_numeric(df["final_score"], errors="coerce") < 50
    ].copy()

    col1, col2 = st.columns(2)

    col1.metric(
        "Students Needing Support",
        len(support_students)
    )

    col2.metric(
        "Support Rate",
        f"{len(support_students) / len(df) * 100:.1f}%"
        if len(df) > 0 else "0%"
    )

    if not support_students.empty:
        st.warning(
            "These students have scores below 50 "
            "and may benefit from additional academic support."
        )
        st.dataframe(
            support_students,
            use_container_width=True
        )

        if "student_id" in support_students.columns:
            selected_id = st.selectbox(
                "Search for a student needing support",
                support_students["student_id"].tolist()
            )

            selected_student = support_students[
                support_students["student_id"] == selected_id
            ]

            st.subheader("Selected Student")
            st.dataframe(
                selected_student,
                use_container_width=True
            )
    else:
        st.success("No students scored below 50.")
else:
    st.info(
        "The dataset needs a 'final_score' column "
        "to display this analysis."
    )

st.caption(
    "Student Performance Analysis Project | "
    "Educational use — dataset contains synthetic student records."
)
# =====================================
# 6. MACHINE LEARNING PREDICTION
# =====================================

import os
import joblib

st.header("🤖 Student Performance Prediction")

model_path = "student_performance_model.pkl"

if os.path.exists(model_path):

    st.write(
        "Enter student information to get a model prediction."
    )

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider(
            "Study Hours per Day",
            min_value=1.0,
            max_value=8.0,
            value=4.0,
            step=0.5
        )

        attendance = st.slider(
            "Attendance (%)",
            min_value=50.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

    with col2:
        previous_score = st.slider(
            "Previous Score",
            min_value=40.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        sleep_hours = st.slider(
            "Sleep Hours",
            min_value=4.0,
            max_value=10.0,
            value=7.0,
            step=0.5
        )

    if st.button("Predict Student Status"):

        model = joblib.load(model_path)

        student_data = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "sleep_hours": sleep_hours
        }])

        prediction = model.predict(student_data)[0]

        st.subheader("Prediction Result")
        st.write(f"Predicted Status: **{prediction}**")

        st.caption(
            "This is an educational prediction using synthetic data, "
            "not a reliable assessment of a real student's ability."
        )

else:
    st.warning(
        "Trained model not found. Run data_mining.py first "
        "to create student_performance_model.pkl."
    )
    # =====================================
# 7. MODEL EVALUATION
# =====================================

st.header("📊 Machine Learning Model Evaluation")

if os.path.exists(model_path):

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    from sklearn.tree import DecisionTreeClassifier

    if st.button("Evaluate Model"):

        features = [
            "study_hours",
            "attendance",
            "previous_score",
            "sleep_hours"
        ]

        X = df[features]
        y = df["status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        evaluation_model = DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        )

        evaluation_model.fit(X_train, y_train)
        predictions = evaluation_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions,
            pos_label="Needs Support",
            zero_division=0
        )
        recall = recall_score(
            y_test, predictions,
            pos_label="Needs Support",
            zero_division=0
        )
        f1 = f1_score(
            y_test, predictions,
            pos_label="Needs Support",
            zero_division=0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy * 100:.1f}%")
        c2.metric("Precision", f"{precision * 100:.1f}%")
        c3.metric("Recall", f"{recall * 100:.1f}%")
        c4.metric("F1 Score", f"{f1 * 100:.1f}%")

        st.subheader("Confusion Matrix")

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=["Pass", "Needs Support"]
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pass", "Needs Support"],
            yticklabels=["Pass", "Needs Support"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Evaluation uses a held-out test split. "
            "The dataset is synthetic, so these metrics do not "
            "establish real-world predictive performance."
        )