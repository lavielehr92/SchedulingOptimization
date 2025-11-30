import math
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="CCA Staffing & Scheduling Helper",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Campus Information
CAMPUSES = {
    "58th Street": {
        "address": "1939 S. 58th St. Philadelphia, PA",
        "short": "58th St"
    },
    "Baltimore Ave": {
        "address": "4109 Baltimore Ave Philadelphia, PA", 
        "short": "Balt Ave"
    }
}

# -----------------------------
# Data Classes
# -----------------------------
@dataclass
class TeacherType:
    name: str
    is_traveling: bool  # Can travel between campuses
    max_periods_per_day: int
    max_consecutive_periods: int
    requires_special_room: bool
    room_type: str  # "classroom", "gym", "art", "music", "lab"
    min_break_between_travel: int  # periods needed after traveling

TEACHER_TYPES = {
    "Core (ELA/Math)": TeacherType("Core", False, 6, 3, False, "classroom", 0),
    "Science": TeacherType("Science", False, 6, 2, True, "lab", 0),
    "Gym/PE": TeacherType("Gym/PE", True, 8, 4, True, "gym", 1),
    "Art": TeacherType("Art", True, 6, 3, True, "art", 1),
    "Music": TeacherType("Music", True, 6, 3, True, "music", 1),
    "Library": TeacherType("Library", True, 6, 2, True, "library", 1),
    "Spanish/Language": TeacherType("Spanish", True, 6, 3, False, "classroom", 1),
    "Bible/Chapel": TeacherType("Bible", True, 5, 2, False, "classroom", 1),
    "STEM/Tech": TeacherType("STEM", True, 6, 2, True, "lab", 1),
}

# -----------------------------
# Session State Initialization
# -----------------------------
if 'teachers' not in st.session_state:
    st.session_state.teachers = []
if 'schedule' not in st.session_state:
    st.session_state.schedule = None
if 'teacher_schedules' not in st.session_state:
    st.session_state.teacher_schedules = {}
if 'campus_assignments' not in st.session_state:
    st.session_state.campus_assignments = {"58th Street": [], "Baltimore Ave": []}

# -----------------------------
# Helper Functions
# -----------------------------
def generate_time_slots(start_hour=8, periods=8, period_length=50, passing_time=5):
    """Generate time slots for the schedule."""
    slots = []
    current_minutes = start_hour * 60
    for i in range(periods):
        start = f"{current_minutes // 60}:{current_minutes % 60:02d}"
        end_minutes = current_minutes + period_length
        end = f"{end_minutes // 60}:{end_minutes % 60:02d}"
        slots.append({
            "period": i + 1,
            "start": start,
            "end": end,
            "label": f"Period {i+1}\n{start}-{end}"
        })
        current_minutes = end_minutes + passing_time
    return slots

def generate_schedule(grades_per_campus, homerooms_per_grade, teachers_config, 
                      periods_per_day, travel_time_periods, classes_per_week):
    """Generate a weekly schedule for both campuses."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    schedule = {
        "58th Street": {day: {p: [] for p in range(1, periods_per_day + 1)} for day in days},
        "Baltimore Ave": {day: {p: [] for p in range(1, periods_per_day + 1)} for day in days}
    }
    
    teacher_schedules = {}  # Track each teacher's schedule
    grade_schedules = {}  # Track each grade's schedule to avoid conflicts
    
    # Initialize teacher schedules
    for teacher in teachers_config:
        teacher_schedules[teacher['name']] = {
            day: {p: None for p in range(1, periods_per_day + 1)} for day in days
        }
    
    # Initialize grade schedules for each campus
    for campus in ["58th Street", "Baltimore Ave"]:
        grade_schedules[campus] = {}
        for grade in range(1, grades_per_campus + 1):
            for homeroom in range(1, homerooms_per_grade + 1):
                hr_key = f"{grade}-{homeroom}"
                grade_schedules[campus][hr_key] = {
                    day: {p: None for p in range(1, periods_per_day + 1)} for day in days
                }
    
    # Create list of all classes that need to be scheduled
    classes_to_schedule = []
    
    for campus in ["58th Street", "Baltimore Ave"]:
        for grade in range(1, grades_per_campus + 1):
            for homeroom in range(1, homerooms_per_grade + 1):
                for teacher in teachers_config:
                    teacher_type = TEACHER_TYPES.get(teacher['type'], TEACHER_TYPES["Core (ELA/Math)"])
                    
                    # Skip if teacher can't serve this campus
                    if not teacher_type.is_traveling and teacher['home_campus'] != campus and teacher['home_campus'] != "Both (Traveling)":
                        continue
                    
                    # Each homeroom needs X classes per week with this teacher type
                    for class_num in range(classes_per_week):
                        classes_to_schedule.append({
                            "campus": campus,
                            "grade": f"Grade {grade}",
                            "homeroom": f"{grade}-{homeroom}",
                            "teacher_type": teacher['type'],
                            "teacher_name": teacher['name'],
                            "priority": class_num  # Schedule first classes of week first
                        })
    
    # Sort to spread classes across the week
    classes_to_schedule.sort(key=lambda x: (x['priority'], x['homeroom']))
    
    # Simple greedy scheduling
    for class_info in classes_to_schedule:
        scheduled = False
        teacher_name = class_info['teacher_name']
        teacher_type = TEACHER_TYPES.get(class_info['teacher_type'], TEACHER_TYPES["Core (ELA/Math)"])
        homeroom = class_info['homeroom']
        campus = class_info['campus']
        
        # Try to spread across different days
        day_order = days.copy()
        
        for day in day_order:
            if scheduled:
                break
            
            # Count how many classes this homeroom already has today
            homeroom_today_count = sum(
                1 for p in range(1, periods_per_day + 1)
                if grade_schedules[campus][homeroom][day][p] is not None
            )
            
            for period in range(1, periods_per_day + 1):
                if scheduled:
                    break
                
                # Check if homeroom is already in class
                if grade_schedules[campus][homeroom][day][period] is not None:
                    continue
                    
                # Check if teacher is available
                if teacher_schedules[teacher_name][day][period] is not None:
                    continue
                
                # Check teacher daily limit
                teacher_today_count = sum(
                    1 for p in range(1, periods_per_day + 1)
                    if teacher_schedules[teacher_name][day][p] is not None
                )
                if teacher_today_count >= teacher_type.max_periods_per_day:
                    continue
                
                # Check if traveling teacher needs buffer
                if teacher_type.is_traveling and period > 1:
                    prev_assignment = teacher_schedules[teacher_name][day][period - 1]
                    if prev_assignment and prev_assignment['campus'] != campus:
                        # Need buffer period for travel
                        if teacher_type.min_break_between_travel > 0:
                            continue
                
                # Check max consecutive periods
                consecutive = 0
                for p in range(max(1, period - teacher_type.max_consecutive_periods), period):
                    if teacher_schedules[teacher_name][day][p] is not None:
                        consecutive += 1
                    else:
                        consecutive = 0
                
                if consecutive >= teacher_type.max_consecutive_periods:
                    continue
                
                # Schedule the class
                assignment = {
                    "campus": campus,
                    "grade": class_info['grade'],
                    "homeroom": homeroom,
                    "teacher": teacher_name,
                    "teacher_type": class_info['teacher_type']
                }
                
                schedule[campus][day][period].append(assignment)
                teacher_schedules[teacher_name][day][period] = assignment
                grade_schedules[campus][homeroom][day][period] = assignment
                scheduled = True
    
    return schedule, teacher_schedules

# -----------------------------
# Main UI
# -----------------------------
st.title("K-8 Staffing and Scheduling Helper")
st.markdown("### Cornerstone Christian Academy")

# Campus info
col1, col2 = st.columns(2)
with col1:
    st.info(f"**58th Street Campus**\n\n{CAMPUSES['58th Street']['address']}")
with col2:
    st.info(f"**Baltimore Ave Campus**\n\n{CAMPUSES['Baltimore Ave']['address']}")

st.markdown(
    """
Use this tool to estimate teacher load, staffing needs, and generate schedules
for specials like Gym, Art, or Music across both campuses.

**Key Parameters:**
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
    "Homerooms per grade (per campus)",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
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
)

period_length = st.sidebar.number_input(
    "Period length (minutes)",
    min_value=30,
    max_value=90,
    value=50,
    step=5,
)

periods_per_day = st.sidebar.number_input(
    "Periods per day",
    min_value=4,
    max_value=10,
    value=8,
    step=1,
)

full_time_load = st.sidebar.number_input(
    "Full time teaching load per week (minutes)",
    min_value=500,
    max_value=2000,
    value=1000,
    step=50,
)

tipping_min = st.sidebar.number_input(
    "Student-minute tipping point",
    min_value=8000,
    max_value=20000,
    value=12000,
    step=500,
)

st.sidebar.header("Travel")

travel_time = st.sidebar.number_input(
    "Travel time between campuses (minutes)",
    min_value=0,
    max_value=60,
    value=10,
    step=5,
)

travel_buffer_periods = st.sidebar.number_input(
    "Buffer periods needed after travel",
    min_value=0,
    max_value=2,
    value=1,
    step=1,
)

# -----------------------------
# Tabs for different sections
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Staffing Analysis", 
    "Teacher Configuration",
    "Schedule Generator",
    "View Schedules"
])

# -----------------------------
# TAB 1: Staffing Analysis
# -----------------------------
with tab1:
    st.subheader("School and Demand Summary")
    
    # Both campuses combined
    total_homerooms = grades * homerooms_per_grade * 2  # 2 campuses
    total_students_est = total_homerooms * avg_class_size
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grades", f"{grades}")
    c2.metric("Estimated Students (Both)", f"{total_students_est:,}")
    c3.metric("Total Homerooms", f"{total_homerooms}")
    c4.metric("Homerooms per Campus", f"{total_homerooms // 2}")
    
    st.divider()
    
    # Per-subject analysis
    st.subheader("Staffing Analysis by Subject")
    
    selected_subject = st.selectbox(
        "Select subject to analyze",
        options=list(TEACHER_TYPES.keys())
    )
    
    teacher_type = TEACHER_TYPES[selected_subject]
    
    # Show rules for this teacher type
    st.markdown(f"**Rules for {selected_subject}:**")
    rules_col1, rules_col2 = st.columns(2)
    with rules_col1:
        st.write(f"- Can travel between campuses: {'Yes' if teacher_type.is_traveling else 'No'}")
        st.write(f"- Max periods per day: {teacher_type.max_periods_per_day}")
        st.write(f"- Max consecutive periods: {teacher_type.max_consecutive_periods}")
    with rules_col2:
        st.write(f"- Requires special room: {'Yes' if teacher_type.requires_special_room else 'No'}")
        st.write(f"- Room type needed: {teacher_type.room_type}")
        st.write(f"- Buffer after travel: {teacher_type.min_break_between_travel} period(s)")
    
    st.divider()
    
    # Calculate staffing needs
    total_sections_per_week = total_homerooms * classes_per_week
    total_teacher_minutes_required = total_sections_per_week * period_length
    
    # Adjust for non-traveling teachers
    if not teacher_type.is_traveling:
        # Need separate teacher for each campus
        min_teachers_needed = max(2, math.ceil(total_teacher_minutes_required / full_time_load))
        st.warning(f"{selected_subject} teachers cannot travel - need at least one per campus")
    else:
        min_teachers_needed = max(1, math.ceil(total_teacher_minutes_required / full_time_load))
    
    avg_minutes_per_teacher = total_teacher_minutes_required / min_teachers_needed
    avg_periods_per_teacher = avg_minutes_per_teacher / period_length
    
    # Student-minute calculation
    student_minutes = total_students_est * classes_per_week * period_length
    hits_tipping = student_minutes >= tipping_min
    
    st.subheader("Teacher Load and Staffing")
    
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Minutes Needed", f"{total_teacher_minutes_required:,.0f}")
    t2.metric("Min Teachers Needed", f"{min_teachers_needed}")
    t3.metric("Avg Load/Teacher (min)", f"{avg_minutes_per_teacher:,.0f}")
    t4.metric("Avg Periods/Teacher", f"{avg_periods_per_teacher:,.1f}")
    
    tip_col1, tip_col2 = st.columns([2, 1])
    with tip_col1:
        st.markdown(f"""
- Total student-minutes: **{student_minutes:,.0f}**
- Tipping threshold: **{tipping_min:,.0f}**
""")
    with tip_col2:
        if hits_tipping:
            st.error("Consider adding a teacher")
        else:
            st.success("Staffing within range")

# -----------------------------
# TAB 2: Teacher Configuration
# -----------------------------
with tab2:
    st.subheader("Configure Teachers")
    
    st.markdown("Add teachers and assign them to subject areas. Rules will be applied based on teacher type.")
    
    # Show teacher type rules
    with st.expander("View Teacher Type Rules"):
        rules_data = []
        for name, tt in TEACHER_TYPES.items():
            rules_data.append({
                "Type": name,
                "Can Travel": "Yes" if tt.is_traveling else "No",
                "Max/Day": tt.max_periods_per_day,
                "Max Consecutive": tt.max_consecutive_periods,
                "Special Room": tt.room_type if tt.requires_special_room else "No",
                "Travel Buffer": tt.min_break_between_travel
            })
        st.dataframe(pd.DataFrame(rules_data), use_container_width=True, hide_index=True)
    
    with st.form("add_teacher"):
        col1, col2, col3 = st.columns(3)
        with col1:
            teacher_name = st.text_input("Teacher Name", placeholder="e.g., Mrs. Johnson")
        with col2:
            teacher_type_select = st.selectbox("Teacher Type", options=list(TEACHER_TYPES.keys()))
        with col3:
            home_campus = st.selectbox("Home Campus", options=["Both (Traveling)", "58th Street", "Baltimore Ave"])
        
        add_btn = st.form_submit_button("Add Teacher", use_container_width=True)
        
        if add_btn and teacher_name:
            st.session_state.teachers.append({
                "name": teacher_name,
                "type": teacher_type_select,
                "home_campus": home_campus,
                "rules": TEACHER_TYPES[teacher_type_select]
            })
            st.success(f"Added {teacher_name}")
            st.rerun()
    
    st.divider()
    
    if st.session_state.teachers:
        st.markdown("### Current Teachers")
        
        teacher_df = pd.DataFrame([
            {
                "Name": t['name'],
                "Type": t['type'],
                "Home Campus": t['home_campus'],
                "Can Travel": "Yes" if TEACHER_TYPES[t['type']].is_traveling else "No",
                "Max/Day": TEACHER_TYPES[t['type']].max_periods_per_day,
                "Room Needed": TEACHER_TYPES[t['type']].room_type
            }
            for t in st.session_state.teachers
        ])
        
        st.dataframe(teacher_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Teachers", type="secondary", use_container_width=True):
                st.session_state.teachers = []
                st.session_state.schedule = None
                st.session_state.teacher_schedules = {}
                st.rerun()
    else:
        st.info("No teachers configured. Add teachers above or use the quick-add below.")
    
    st.divider()
    
    if st.button("Load Sample Teachers", type="primary", use_container_width=True):
        st.session_state.teachers = [
            {"name": "Mr. Garcia", "type": "Gym/PE", "home_campus": "Both (Traveling)"},
            {"name": "Mrs. Chen", "type": "Art", "home_campus": "Both (Traveling)"},
            {"name": "Mr. Williams", "type": "Music", "home_campus": "Both (Traveling)"},
            {"name": "Mrs. Davis", "type": "Spanish/Language", "home_campus": "Both (Traveling)"},
            {"name": "Pastor Johnson", "type": "Bible/Chapel", "home_campus": "Both (Traveling)"},
            {"name": "Ms. Thompson", "type": "Library", "home_campus": "Both (Traveling)"},
        ]
        st.rerun()

# -----------------------------
# TAB 3: Schedule Generator
# -----------------------------
with tab3:
    st.subheader("Generate Weekly Schedule")
    
    if not st.session_state.teachers:
        st.warning("Please add teachers in the 'Teacher Configuration' tab first.")
    else:
        st.markdown(f"""
**Schedule Parameters:**
- Grades: {grades} | Homerooms per grade: {homerooms_per_grade} per campus
- Periods per day: {periods_per_day} | Period length: {period_length} minutes
- Specials per homeroom per week: {classes_per_week}
- Travel time between campuses: {travel_time} minutes
- Buffer periods after travel: {travel_buffer_periods}
""")
        
        st.markdown("**Teachers configured:**")
        for t in st.session_state.teachers:
            st.write(f"- {t['name']} ({t['type']}) - {t['home_campus']}")
        
        st.divider()
        
        if st.button("Generate Schedule", type="primary", use_container_width=True):
            with st.spinner("Generating optimized schedule..."):
                schedule, teacher_schedules = generate_schedule(
                    grades_per_campus=grades,
                    homerooms_per_grade=homerooms_per_grade,
                    teachers_config=st.session_state.teachers,
                    periods_per_day=periods_per_day,
                    travel_time_periods=travel_buffer_periods,
                    classes_per_week=classes_per_week
                )
                st.session_state.schedule = schedule
                st.session_state.teacher_schedules = teacher_schedules
                st.success("Schedule generated successfully!")
                st.rerun()
        
        if st.session_state.schedule:
            st.success("Schedule is ready! View it in the 'View Schedules' tab.")

# -----------------------------
# TAB 4: View Schedules
# -----------------------------
with tab4:
    st.subheader("Weekly Schedules")
    
    if not st.session_state.schedule:
        st.info("No schedule generated yet. Go to 'Schedule Generator' tab to create one.")
    else:
        time_slots = generate_time_slots(
            start_hour=8,
            periods=periods_per_day,
            period_length=period_length
        )
        
        view_option = st.radio(
            "View by:",
            ["Campus Schedule", "Teacher Schedule", "Side-by-Side Campuses"],
            horizontal=True
        )
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        
        if view_option == "Campus Schedule":
            col1, col2 = st.columns(2)
            with col1:
                selected_campus = st.selectbox("Select Campus", ["58th Street", "Baltimore Ave"])
            with col2:
                selected_day = st.selectbox("Select Day", days)
            
            st.markdown(f"### {selected_campus} - {selected_day}")
            st.caption(f"Address: {CAMPUSES[selected_campus]['address']}")
            
            # Create schedule table
            schedule_data = []
            for slot in time_slots:
                period = slot['period']
                classes = st.session_state.schedule[selected_campus][selected_day][period]
                
                if classes:
                    for c in classes:
                        schedule_data.append({
                            "Period": period,
                            "Time": f"{slot['start']}-{slot['end']}",
                            "Grade": c['grade'],
                            "Homeroom": c['homeroom'],
                            "Teacher": c['teacher'],
                            "Subject": c['teacher_type']
                        })
                else:
                    schedule_data.append({
                        "Period": period,
                        "Time": f"{slot['start']}-{slot['end']}",
                        "Grade": "-",
                        "Homeroom": "-",
                        "Teacher": "-",
                        "Subject": "-"
                    })
            
            df = pd.DataFrame(schedule_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Show full week view
            st.markdown("### Full Week Overview")
            
            week_data = []
            for slot in time_slots:
                row = {"Period": slot['period'], "Time": f"{slot['start']}-{slot['end']}"}
                for day in days:
                    classes = st.session_state.schedule[selected_campus][day][slot['period']]
                    if classes:
                        row[day] = ", ".join([f"{c['homeroom']}" for c in classes[:3]])
                        if len(classes) > 3:
                            row[day] += f" +{len(classes)-3}"
                    else:
                        row[day] = "-"
                week_data.append(row)
            
            week_df = pd.DataFrame(week_data)
            st.dataframe(week_df, use_container_width=True, hide_index=True)
        
        elif view_option == "Teacher Schedule":
            if st.session_state.teachers:
                selected_teacher = st.selectbox(
                    "Select Teacher",
                    options=[t['name'] for t in st.session_state.teachers]
                )
                
                st.markdown(f"### {selected_teacher}'s Weekly Schedule")
                
                # Find teacher info
                teacher_info = next((t for t in st.session_state.teachers if t['name'] == selected_teacher), None)
                if teacher_info:
                    teacher_type = TEACHER_TYPES[teacher_info['type']]
                    st.caption(f"Type: {teacher_info['type']} | Home: {teacher_info['home_campus']} | Can Travel: {'Yes' if teacher_type.is_traveling else 'No'}")
                
                # Build teacher schedule
                teacher_week = []
                for slot in time_slots:
                    row = {"Period": slot['period'], "Time": f"{slot['start']}-{slot['end']}"}
                    for day in days:
                        assignment = st.session_state.teacher_schedules.get(selected_teacher, {}).get(day, {}).get(slot['period'])
                        if assignment:
                            campus_short = "58th" if assignment['campus'] == "58th Street" else "Balt"
                            row[day] = f"{campus_short}: {assignment['homeroom']}"
                        else:
                            row[day] = "-"
                    teacher_week.append(row)
                
                teacher_df = pd.DataFrame(teacher_week)
                st.dataframe(teacher_df, use_container_width=True, hide_index=True)
                
                # Calculate teacher stats
                total_periods = sum(
                    1 for day in days 
                    for p in range(1, periods_per_day + 1)
                    if st.session_state.teacher_schedules.get(selected_teacher, {}).get(day, {}).get(p)
                )
                total_minutes = total_periods * period_length
                
                # Count campus switches
                campus_switches = 0
                for day in days:
                    prev_campus = None
                    for p in range(1, periods_per_day + 1):
                        assignment = st.session_state.teacher_schedules.get(selected_teacher, {}).get(day, {}).get(p)
                        if assignment:
                            if prev_campus and prev_campus != assignment['campus']:
                                campus_switches += 1
                            prev_campus = assignment['campus']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Periods/Week", total_periods)
                with col2:
                    st.metric("Teaching Minutes", total_minutes)
                with col3:
                    st.metric("Campus Switches/Week", campus_switches)
                
                load_pct = total_minutes / full_time_load * 100
                if load_pct > 100:
                    st.error(f"Load: {load_pct:.1f}% - OVERLOADED")
                elif load_pct > 90:
                    st.warning(f"Load: {load_pct:.1f}% - Near capacity")
                else:
                    st.success(f"Load: {load_pct:.1f}% - Within range")
        
        else:  # Side-by-Side Campuses
            selected_day = st.selectbox("Select Day", days)
            
            st.markdown(f"### Both Campuses - {selected_day}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**58th Street Campus**")
                st.caption(CAMPUSES["58th Street"]["address"])
                
                data_58 = []
                for slot in time_slots:
                    classes = st.session_state.schedule["58th Street"][selected_day][slot['period']]
                    if classes:
                        for c in classes:
                            data_58.append({
                                "Period": slot['period'],
                                "Time": f"{slot['start']}-{slot['end']}",
                                "Class": c['homeroom'],
                                "Teacher": c['teacher'].split()[0],  # First name only
                                "Subject": c['teacher_type'].split('/')[0]  # Short subject
                            })
                    else:
                        data_58.append({
                            "Period": slot['period'],
                            "Time": f"{slot['start']}-{slot['end']}",
                            "Class": "-",
                            "Teacher": "-",
                            "Subject": "-"
                        })
                st.dataframe(pd.DataFrame(data_58), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown(f"**Baltimore Ave Campus**")
                st.caption(CAMPUSES["Baltimore Ave"]["address"])
                
                data_balt = []
                for slot in time_slots:
                    classes = st.session_state.schedule["Baltimore Ave"][selected_day][slot['period']]
                    if classes:
                        for c in classes:
                            data_balt.append({
                                "Period": slot['period'],
                                "Time": f"{slot['start']}-{slot['end']}",
                                "Class": c['homeroom'],
                                "Teacher": c['teacher'].split()[0],
                                "Subject": c['teacher_type'].split('/')[0]
                            })
                    else:
                        data_balt.append({
                            "Period": slot['period'],
                            "Time": f"{slot['start']}-{slot['end']}",
                            "Class": "-",
                            "Teacher": "-",
                            "Subject": "-"
                        })
                st.dataframe(pd.DataFrame(data_balt), use_container_width=True, hide_index=True)
        
        # Export section
        st.divider()
        st.subheader("Export Schedule")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export all schedules to CSV
            all_data = []
            for campus in ["58th Street", "Baltimore Ave"]:
                for day in days:
                    for period in range(1, periods_per_day + 1):
                        for c in st.session_state.schedule[campus][day][period]:
                            slot = time_slots[period - 1]
                            all_data.append({
                                "Campus": campus,
                                "Day": day,
                                "Period": period,
                                "Time": f"{slot['start']}-{slot['end']}",
                                "Grade": c['grade'],
                                "Homeroom": c['homeroom'],
                                "Teacher": c['teacher'],
                                "Subject": c['teacher_type']
                            })
            
            if all_data:
                export_df = pd.DataFrame(all_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "Download Full Schedule (CSV)",
                    data=csv,
                    file_name="cca_schedule.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            # Export teacher schedules
            teacher_data = []
            for teacher in st.session_state.teachers:
                for day in days:
                    for period in range(1, periods_per_day + 1):
                        assignment = st.session_state.teacher_schedules.get(teacher['name'], {}).get(day, {}).get(period)
                        if assignment:
                            slot = time_slots[period - 1]
                            teacher_data.append({
                                "Teacher": teacher['name'],
                                "Type": teacher['type'],
                                "Day": day,
                                "Period": period,
                                "Time": f"{slot['start']}-{slot['end']}",
                                "Campus": assignment['campus'],
                                "Homeroom": assignment['homeroom']
                            })
            
            if teacher_data:
                teacher_export_df = pd.DataFrame(teacher_data)
                csv = teacher_export_df.to_csv(index=False)
                st.download_button(
                    "Download Teacher Schedules (CSV)",
                    data=csv,
                    file_name="cca_teacher_schedules.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Cornerstone Christian Academy - Scheduling Optimization Tool | 58th St & Baltimore Ave Campuses")
