import math
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="CCA Staffing & Scheduling Helper",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("K-8 Staffing and Scheduling Helper")
st.markdown("### Cornerstone Christian Academy")

st.markdown(
    """
Use this tool to estimate teacher load and staffing needs for specials
like Gym, Art, or Music across two campuses.

The logic is based on:
- Period length = **50 minutes**
- Full time teaching load = **1000 minutes per week**
- Tipping point for adding a teacher = **12k-13k student-minutes**
"""
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("School Structure")

grades = st.sidebar.number_input(
    "Number of grades (K-8 is 9)", min_value=1, max_value=12, value=9, step=1
)

homerooms_per_grade = st.sidebar.number_input(
    "Homerooms per grade",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Use total homerooms per grade across both campuses.",
)

avg_class_size = st.sidebar.number_input(
    "Average students per homeroom",
    min_value=5,
    max_value=30,
    value=18,
    step=1,
)

st.sidebar.header("Instructional Model")

classes_per_week = st.sidebar.number_input(
    "Special classes per week (per homeroom)",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
    help="For example, Gym twice per week for each homeroom.",
)

period_length = st.sidebar.number_input(
    "Period length (minutes)",
    min_value=30,
    max_value=90,
    value=50,
    step=5,
)

full_time_load = st.sidebar.number_input(
    "Full time teaching load per week (minutes)",
    min_value=500,
    max_value=2000,
    value=1000,
    step=50,
)

tipping_min = st.sidebar.number_input(
    "Student-minute tipping point for new teacher",
    min_value=8000,
    max_value=20000,
    value=12000,
    step=500,
    help="Rule of thumb where total student-minutes suggest adding another teacher.",
)

st.sidebar.header("Campuses")

num_campuses = st.sidebar.selectbox(
    "Number of campuses", options=[1, 2], index=1
)

travel_time = st.sidebar.number_input(
    "Travel time between campuses (minutes)",
    min_value=0,
    max_value=60,
    value=10,
    step=5,
    help="Used for travel load estimates only.",
)

# -----------------------------
# Core Calculations
# -----------------------------
total_homerooms = grades * homerooms_per_grade
total_students_est = total_homerooms * avg_class_size

# How many class meetings of the special per week
total_sections_per_week = total_homerooms * classes_per_week

# Total teaching minutes required for that subject
total_teacher_minutes_required = total_sections_per_week * period_length

# Minimum teachers needed to keep each at or below full_time_load
min_teachers_needed = max(
    1, math.ceil(total_teacher_minutes_required / full_time_load)
)

avg_minutes_per_teacher = total_teacher_minutes_required / min_teachers_needed
avg_periods_per_teacher = avg_minutes_per_teacher / period_length

# Student-minute load (for the tipping rule)
student_minutes = total_students_est * classes_per_week * period_length
hits_tipping = student_minutes >= tipping_min

# Very rough travel estimate:
# assume one teacher splits time across campuses
# and travels once per day between them
if num_campuses == 2:
    weekly_travel_minutes = travel_time * 5  # 5 days
else:
    weekly_travel_minutes = 0

# -----------------------------
# Outputs: Metrics
# -----------------------------
st.subheader("School and Demand Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Grades", f"{grades}")
c2.metric("Estimated Students", f"{total_students_est:,}")
c3.metric("Homerooms", f"{total_homerooms}")
c4.metric("Classes per Week (this subject)", f"{total_sections_per_week}")

st.subheader("Teacher Load and Staffing")

t1, t2, t3, t4 = st.columns(4)
t1.metric(
    "Total Teacher Minutes Needed",
    f"{total_teacher_minutes_required:,.0f}",
    help="All class meetings of this special across both campuses.",
)
t2.metric(
    "Minimum Teachers Needed",
    f"{min_teachers_needed}",
    help="Based on the full time load constraint.",
)
t3.metric(
    "Average Load per Teacher (minutes/week)",
    f"{avg_minutes_per_teacher:,.0f}",
)
t4.metric(
    "Average Periods per Teacher (per week)",
    f"{avg_periods_per_teacher:,.1f}",
)

st.subheader("Student-Minute Check")

tip_col1, tip_col2 = st.columns([2, 1])
with tip_col1:
    st.markdown(
        f"""
- Total student-minutes for this subject: **{student_minutes:,.0f}**
- Tipping threshold: **{tipping_min:,.0f}** student-minutes
"""
    )

with tip_col2:
    if hits_tipping:
        st.error("Tipping point reached or exceeded. Consider adding a teacher.")
    else:
        st.success("Below tipping point. Staffing is within the target range.")

if num_campuses == 2:
    st.subheader("Travel Considerations")
    st.markdown(
        f"""
Assuming at least one teacher travels between campuses once per day:

- Travel time each way: **{travel_time} minutes**
- Estimated weekly travel time: **{weekly_travel_minutes} minutes**

You can treat this as unteachable time when planning a full load.
"""
    )

# -----------------------------
# Scenario Table
# -----------------------------
st.subheader("What If We Add More Teachers?")

scenario_teachers = st.slider(
    "Number of teachers to test scenarios",
    min_value=1,
    max_value=max(6, min_teachers_needed + 3),
    value=min_teachers_needed,
)

scenario_minutes_per_teacher = total_teacher_minutes_required / scenario_teachers
scenario_periods_per_teacher = scenario_minutes_per_teacher / period_length

st.markdown(
    f"""
With **{scenario_teachers}** teacher(s):

- Load per teacher: **{scenario_minutes_per_teacher:,.0f} minutes/week**
- Periods per teacher: **{scenario_periods_per_teacher:,.1f} per week**
- As a share of full time load: **{scenario_minutes_per_teacher / full_time_load:.1%}**
  of the target {full_time_load} minutes
"""
)

st.caption(
    "You can extend this app later with an OR-Tools timetable solver that assigns exact periods and campuses."
)
