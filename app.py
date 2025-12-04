import math
import streamlit as st
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
import copy
from datetime import datetime
import io

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
        "short": "58th St",
        "grade_start": 0,  # K
        "grade_end": 4,    # 4th grade
        "grade_label": "K-4"
    },
    "Baltimore Ave": {
        "address": "4109 Baltimore Ave Philadelphia, PA", 
        "short": "Balt Ave",
        "grade_start": 5,  # 5th grade
        "grade_end": 8,    # 8th grade
        "grade_label": "5-8"
    }
}

# Helper function to calculate grades per campus
def get_campus_grades(campus_name: str) -> int:
    """Returns the number of grades at a specific campus"""
    campus = CAMPUSES[campus_name]
    return campus["grade_end"] - campus["grade_start"] + 1

def get_total_grades() -> int:
    """Returns total number of unique grades across all campuses"""
    return sum(get_campus_grades(c) for c in CAMPUSES.keys())

def get_grade_name(grade_num: int) -> str:
    """Returns display name for a grade number (0=K, 1=1st, etc.)"""
    if grade_num == 0:
        return "K"
    return str(grade_num)

# -----------------------------
# Data Classes
# -----------------------------
@dataclass
class TeacherType:
    name: str
    is_traveling: bool
    max_periods_per_day: int
    max_consecutive_periods: int
    requires_special_room: bool
    room_type: str
    min_break_between_travel: int

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
# Base Estimate Data (CCA Default)
# -----------------------------
# With 18 homerooms × 2 classes/week = 36 classes per subject (1800 min)
# Load cap of 1000 min/week = 20 periods max per teacher
# Travel cooldown is soft constraint, so 2 teachers per subject should work
BASE_TEACHERS = [
    # Gym/PE - 2 teachers
    {"name": "Mr. Garcia", "type": "Gym/PE", "home_campus": "Both (Traveling)"},
    {"name": "Coach Martinez", "type": "Gym/PE", "home_campus": "Both (Traveling)"},
    # Art - 2 teachers
    {"name": "Mrs. Chen", "type": "Art", "home_campus": "Both (Traveling)"},
    {"name": "Ms. Rivera", "type": "Art", "home_campus": "Both (Traveling)"},
    # Music - 2 teachers
    {"name": "Mr. Williams", "type": "Music", "home_campus": "Both (Traveling)"},
    {"name": "Mrs. Anderson", "type": "Music", "home_campus": "Both (Traveling)"},
    # Spanish - 2 teachers
    {"name": "Mrs. Davis", "type": "Spanish/Language", "home_campus": "Both (Traveling)"},
    {"name": "Sr. Rodriguez", "type": "Spanish/Language", "home_campus": "Both (Traveling)"},
    # Bible/Chapel - 2 teachers
    {"name": "Pastor Johnson", "type": "Bible/Chapel", "home_campus": "Both (Traveling)"},
    {"name": "Pastor Smith", "type": "Bible/Chapel", "home_campus": "Both (Traveling)"},
    # Library - 2 teachers
    {"name": "Ms. Thompson", "type": "Library", "home_campus": "Both (Traveling)"},
    {"name": "Mrs. Baker", "type": "Library", "home_campus": "Both (Traveling)"},
]

BASE_SETTINGS = {
    "homerooms_per_grade": 2,
    "avg_class_size": 18,
    "classes_per_week": 2,
    "period_length": 50,
    "periods_per_day": 8,
    "full_time_load": 1250,  # 1200-1300 min = threshold for adding new teacher
    "tipping_min": 12000,
    "travel_time": 10,
    "travel_buffer_periods": 1,
    "max_switches_per_day": 2,
    "max_switches_per_week": 10
}

# -----------------------------
# Default Teacher Type Rules
# -----------------------------
DEFAULT_TEACHER_RULES = {
    "Core (ELA/Math)": {"is_traveling": False, "max_periods_per_day": 6, "max_consecutive_periods": 3, "requires_special_room": False, "room_type": "classroom", "min_break_between_travel": 0},
    "Science": {"is_traveling": False, "max_periods_per_day": 6, "max_consecutive_periods": 2, "requires_special_room": True, "room_type": "lab", "min_break_between_travel": 0},
    "Gym/PE": {"is_traveling": True, "max_periods_per_day": 8, "max_consecutive_periods": 4, "requires_special_room": True, "room_type": "gym", "min_break_between_travel": 1},
    "Art": {"is_traveling": True, "max_periods_per_day": 6, "max_consecutive_periods": 3, "requires_special_room": True, "room_type": "art", "min_break_between_travel": 1},
    "Music": {"is_traveling": True, "max_periods_per_day": 6, "max_consecutive_periods": 3, "requires_special_room": True, "room_type": "music", "min_break_between_travel": 1},
    "Library": {"is_traveling": True, "max_periods_per_day": 6, "max_consecutive_periods": 2, "requires_special_room": True, "room_type": "library", "min_break_between_travel": 1},
    "Spanish/Language": {"is_traveling": True, "max_periods_per_day": 6, "max_consecutive_periods": 3, "requires_special_room": False, "room_type": "classroom", "min_break_between_travel": 1},
    "Bible/Chapel": {"is_traveling": True, "max_periods_per_day": 5, "max_consecutive_periods": 2, "requires_special_room": False, "room_type": "classroom", "min_break_between_travel": 1},
    "STEM/Tech": {"is_traveling": True, "max_periods_per_day": 6, "max_consecutive_periods": 2, "requires_special_room": True, "room_type": "lab", "min_break_between_travel": 1},
}

def get_teacher_type_rules(subject_name):
    """Get teacher type rules from session state or defaults."""
    if 'custom_rules' in st.session_state and subject_name in st.session_state.custom_rules:
        rules = st.session_state.custom_rules[subject_name]
        return TeacherType(
            name=subject_name,
            is_traveling=rules['is_traveling'],
            max_periods_per_day=rules['max_periods_per_day'],
            max_consecutive_periods=rules['max_consecutive_periods'],
            requires_special_room=rules['requires_special_room'],
            room_type=rules['room_type'],
            min_break_between_travel=rules['min_break_between_travel']
        )
    return TEACHER_TYPES.get(subject_name, TEACHER_TYPES["Core (ELA/Math)"])

# -----------------------------
# Session State Initialization
# -----------------------------
if 'teachers' not in st.session_state:
    st.session_state.teachers = []
if 'schedule' not in st.session_state:
    st.session_state.schedule = None
if 'teacher_schedules' not in st.session_state:
    st.session_state.teacher_schedules = {}
if 'unscheduled_classes' not in st.session_state:
    st.session_state.unscheduled_classes = []
if 'saved_schedules' not in st.session_state:
    st.session_state.saved_schedules = {}
if 'settings' not in st.session_state:
    st.session_state.settings = BASE_SETTINGS.copy()
if 'custom_rules' not in st.session_state:
    st.session_state.custom_rules = {k: v.copy() for k, v in DEFAULT_TEACHER_RULES.items()}

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

def generate_schedule(campuses, homerooms_per_grade, teachers_config, 
                      periods_per_day, travel_time_periods, classes_per_week, custom_rules=None,
                      period_length=50, full_time_load=1000, travel_time=10,
                      max_switches_per_day=2, max_switches_per_week=6):
    """Generate a weekly schedule for both campuses with campus-specific grade ranges.
    
    Now enforces:
    - Travel time cooldown after campus switches
    - Weekly teacher load caps
    - Room availability (no double-booking special rooms)
    - Max consecutive periods (proper backward counting)
    - Campus switch limits per day and per week
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    # Calculate travel cooldown periods
    # ceil(travel_time / period_length) + travel_buffer_periods
    travel_cooldown_periods = math.ceil(travel_time / period_length) + travel_time_periods if travel_time > 0 else travel_time_periods
    
    # Helper to get teacher type with custom rules
    def get_type_rules(type_name):
        if custom_rules and type_name in custom_rules:
            rules = custom_rules[type_name]
            return TeacherType(
                name=type_name,
                is_traveling=rules['is_traveling'],
                max_periods_per_day=rules['max_periods_per_day'],
                max_consecutive_periods=rules['max_consecutive_periods'],
                requires_special_room=rules['requires_special_room'],
                room_type=rules['room_type'],
                min_break_between_travel=rules['min_break_between_travel']
            )
        return TEACHER_TYPES.get(type_name, TEACHER_TYPES["Core (ELA/Math)"])
    
    schedule = {
        "58th Street": {day: {p: [] for p in range(1, periods_per_day + 1)} for day in days},
        "Baltimore Ave": {day: {p: [] for p in range(1, periods_per_day + 1)} for day in days}
    }
    
    teacher_schedules = {}
    grade_schedules = {}
    
    # Room availability tracking: room_usage[campus][day][period][room_type] = teacher_name or None
    room_pools = {
        "gym": 1, "art": 1, "music": 1, "lab": 1, "library": 1, "chapel": 1,
        "theater": 1, "computer lab": 1, "classroom": 999  # unlimited classrooms
    }
    room_usage = {
        campus: {
            day: {
                p: {room: [] for room in room_pools.keys()}
                for p in range(1, periods_per_day + 1)
            } for day in days
        } for campus in ["58th Street", "Baltimore Ave"]
    }
    
    for teacher in teachers_config:
        teacher_schedules[teacher['name']] = {
            day: {p: None for p in range(1, periods_per_day + 1)} for day in days
        }
    
    # Initialize grade schedules using campus-specific grade ranges
    for campus_name, campus_info in campuses.items():
        grade_schedules[campus_name] = {}
        for grade in range(campus_info['grade_start'], campus_info['grade_end'] + 1):
            for homeroom in range(1, homerooms_per_grade + 1):
                grade_label = get_grade_name(grade)
                hr_key = f"{grade_label}-{homeroom}"
                grade_schedules[campus_name][hr_key] = {
                    day: {p: None for p in range(1, periods_per_day + 1)} for day in days
                }
    
    classes_to_schedule = []
    
    # Build class list using campus-specific grade ranges
    for campus_name, campus_info in campuses.items():
        for grade in range(campus_info['grade_start'], campus_info['grade_end'] + 1):
            for homeroom in range(1, homerooms_per_grade + 1):
                for teacher in teachers_config:
                    teacher_type = get_type_rules(teacher['type'])
                    
                    if not teacher_type.is_traveling and teacher['home_campus'] != campus_name and teacher['home_campus'] != "Both (Traveling)":
                        continue
                    
                    grade_label = get_grade_name(grade)
                    for class_num in range(classes_per_week):
                        classes_to_schedule.append({
                            "campus": campus_name,
                            "grade": f"Grade {grade_label}",
                            "homeroom": f"{grade_label}-{homeroom}",
                            "teacher_type": teacher['type'],
                            "teacher_name": teacher['name'],
                            "priority": class_num
                        })
    
    # Helper to get teacher's current weekly minutes
    def get_teacher_weekly_minutes(teacher_name):
        count = 0
        for day in days:
            for p in range(1, periods_per_day + 1):
                if teacher_schedules[teacher_name][day][p] is not None:
                    count += 1
        return count * period_length
    
    # Helper to count campus switches for a teacher on a given day
    def count_day_switches(teacher_name, day):
        switches = 0
        prev_campus = None
        for p in range(1, periods_per_day + 1):
            assignment = teacher_schedules[teacher_name][day][p]
            if assignment:
                if prev_campus and prev_campus != assignment['campus']:
                    switches += 1
                prev_campus = assignment['campus']
        return switches
    
    # Helper to count campus switches for a teacher for the whole week
    def count_week_switches(teacher_name):
        return sum(count_day_switches(teacher_name, day) for day in days)
    
    # Helper to find when teacher last switched to a different campus
    def get_last_switch_period(teacher_name, day, current_period, target_campus):
        """Returns the period of the most recent campus switch before current_period.
        Returns 0 if no switch occurred (teacher was at target campus or not scheduled).
        """
        for p in range(current_period - 1, 0, -1):
            assignment = teacher_schedules[teacher_name][day][p]
            if assignment:
                if assignment['campus'] != target_campus:
                    # Found a period where teacher was at different campus
                    return p
                else:
                    # Teacher was at target campus, keep looking back
                    continue
            # Empty period, keep looking
        return 0
    
    # Helper to check consecutive periods (proper backward counting)
    def count_consecutive_back(teacher_name, day, period):
        """Count consecutive assigned periods immediately before the given period."""
        consecutive = 0
        for p in range(period - 1, 0, -1):
            if teacher_schedules[teacher_name][day][p] is not None:
                consecutive += 1
            else:
                break  # Stop at first gap
        return consecutive
    
    # Helper function to check if a slot is valid for a teacher
    def is_valid_slot(teacher_name, teacher_type, campus, homeroom, day, period):
        # Check if homeroom is free
        if grade_schedules[campus][homeroom][day][period] is not None:
            return False
        
        # Check if teacher is free
        if teacher_schedules[teacher_name][day][period] is not None:
            return False
        
        # Check max periods per day
        teacher_today_count = sum(
            1 for p in range(1, periods_per_day + 1)
            if teacher_schedules[teacher_name][day][p] is not None
        )
        if teacher_today_count >= teacher_type.max_periods_per_day:
            return False
        
        # FIX #2: Check weekly load cap
        current_weekly_minutes = get_teacher_weekly_minutes(teacher_name)
        if current_weekly_minutes + period_length > full_time_load:
            return False
        
        # FIX #3: Check room availability for special rooms
        if teacher_type.requires_special_room:
            room = teacher_type.room_type
            rooms_in_use = len(room_usage[campus][day][period][room])
            max_rooms = room_pools.get(room, 1)
            if rooms_in_use >= max_rooms:
                return False
        
        # NOTE: Travel cooldown is now a SOFT constraint (handled in score_slot)
        # It's nice-to-have, not a hard requirement
        
        # FIX #5: Check campus switch limits (still a hard constraint)
        if teacher_type.is_traveling:
            # Check if this assignment would create a new switch
            would_switch = False
            # Look for the most recent assignment before this period
            for back_p in range(period - 1, 0, -1):
                prev_assignment = teacher_schedules[teacher_name][day][back_p]
                if prev_assignment:
                    if prev_assignment['campus'] != campus:
                        would_switch = True
                    break
            
            if would_switch:
                # Check day limit
                current_day_switches = count_day_switches(teacher_name, day)
                if current_day_switches >= max_switches_per_day:
                    return False
                
                # Check week limit
                current_week_switches = count_week_switches(teacher_name)
                if current_week_switches >= max_switches_per_week:
                    return False
        
        # FIX #4: Check consecutive periods (proper backward counting)
        consecutive = count_consecutive_back(teacher_name, day, period)
        if consecutive >= teacher_type.max_consecutive_periods:
            return False
        
        return True
    
    # Helper to score a slot (higher is better for optimization)
    def score_slot(teacher_name, teacher_type, campus, day, period):
        score = 100  # Base score
        
        # FIX #7: Enhanced scoring for better optimization
        
        # SOFT CONSTRAINT: Travel cooldown preference (nice-to-have, not required)
        if teacher_type.is_traveling and travel_cooldown_periods > 0:
            for back_p in range(period - 1, max(0, period - travel_cooldown_periods - 1), -1):
                if back_p < 1:
                    break
                prev_assignment = teacher_schedules[teacher_name][day][back_p]
                if prev_assignment:
                    if prev_assignment['campus'] != campus:
                        # Teacher switching campuses without full cooldown - penalize but allow
                        periods_since_switch = period - back_p
                        if periods_since_switch < travel_cooldown_periods:
                            # Penalty scales with how far under cooldown we are
                            score -= (travel_cooldown_periods - periods_since_switch) * 15
                    break
        
        # STRONG: Minimize campus switches (biggest penalty)
        would_switch = False
        for back_p in range(period - 1, 0, -1):
            prev_assignment = teacher_schedules[teacher_name][day][back_p]
            if prev_assignment:
                if prev_assignment['campus'] != campus:
                    would_switch = True
                    score -= 50  # Heavy penalty for switching
                else:
                    score += 20  # Reward for staying at same campus
                break
        
        # Also check if next period is already scheduled at same campus
        if period < periods_per_day:
            next_assignment = teacher_schedules[teacher_name][day][period + 1]
            if next_assignment:
                if next_assignment['campus'] == campus:
                    score += 15  # Reward for matching upcoming campus
                else:
                    score -= 30  # Penalty if would create sandwich switch
        
        # Spread load across week (avoid overloading days)
        teacher_today_count = sum(
            1 for p in range(1, periods_per_day + 1)
            if teacher_schedules[teacher_name][day][p] is not None
        )
        day_counts = []
        for d in days:
            day_counts.append(sum(
                1 for p in range(1, periods_per_day + 1)
                if teacher_schedules[teacher_name][d][p] is not None
            ))
        avg_per_day = sum(day_counts) / len(days) if day_counts else 0
        
        # Penalty for being above average
        if teacher_today_count > avg_per_day:
            score -= (teacher_today_count - avg_per_day) * 5
        
        # Avoid front/back loading (prefer middle periods)
        mid_period = periods_per_day / 2
        distance_from_mid = abs(period - mid_period)
        score -= distance_from_mid * 2  # Slight penalty for edge periods
        
        # Prefer contiguous blocks (adjacent to existing assignments)
        if period > 1 and teacher_schedules[teacher_name][day][period - 1] is not None:
            prev = teacher_schedules[teacher_name][day][period - 1]
            if prev['campus'] == campus:
                score += 25  # Strong bonus for contiguous same-campus
            else:
                score += 5  # Small bonus for any adjacency
        
        if period < periods_per_day and teacher_schedules[teacher_name][day][period + 1] is not None:
            next_slot = teacher_schedules[teacher_name][day][period + 1]
            if next_slot['campus'] == campus:
                score += 25
            else:
                score += 5
        
        # Slight preference for earlier days to ensure even spread
        day_index = days.index(day)
        score -= day_index * 1  # Very slight preference for earlier days
        
        return score
    
    # Sort classes to schedule teachers with most constraints first
    # Group by teacher, then by campus to minimize travel
    def class_sort_key(c):
        teacher_type = get_type_rules(c['teacher_type'])
        # Lower max_periods = more constrained = schedule first
        constraint_score = teacher_type.max_periods_per_day * 10 + teacher_type.max_consecutive_periods
        # Group by campus within teacher to minimize switches
        campus_order = 0 if c['campus'] == "58th Street" else 1
        return (constraint_score, c['teacher_name'], campus_order, c['homeroom'], c['priority'])
    
    classes_to_schedule.sort(key=class_sort_key)
    
    # Track unscheduled classes
    unscheduled_classes = []
    
    for class_info in classes_to_schedule:
        scheduled = False
        teacher_name = class_info['teacher_name']
        teacher_type = get_type_rules(class_info['teacher_type'])
        homeroom = class_info['homeroom']
        campus = class_info['campus']
        
        # Find all valid slots and pick the best one
        valid_slots = []
        for day in days:
            for period in range(1, periods_per_day + 1):
                if is_valid_slot(teacher_name, teacher_type, campus, homeroom, day, period):
                    slot_score = score_slot(teacher_name, teacher_type, campus, day, period)
                    valid_slots.append((day, period, slot_score))
        
        # Sort by score (descending) and pick best
        if valid_slots:
            valid_slots.sort(key=lambda x: -x[2])
            best_day, best_period, _ = valid_slots[0]
            
            assignment = {
                "campus": campus,
                "grade": class_info['grade'],
                "homeroom": homeroom,
                "teacher": teacher_name,
                "teacher_type": class_info['teacher_type']
            }
            
            schedule[campus][best_day][best_period].append(assignment)
            teacher_schedules[teacher_name][best_day][best_period] = assignment
            grade_schedules[campus][homeroom][best_day][best_period] = assignment
            
            # Update room usage
            if teacher_type.requires_special_room:
                room = teacher_type.room_type
                room_usage[campus][best_day][best_period][room].append(teacher_name)
            
            scheduled = True
        
        # Track if class couldn't be scheduled
        if not scheduled:
            unscheduled_classes.append(class_info)
    
    # Return schedule info along with any unscheduled classes
    return schedule, teacher_schedules, unscheduled_classes

def schedule_to_dataframe(schedule, teacher_schedules, periods_per_day, period_length):
    """Convert schedule to a DataFrame for export."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    time_slots = generate_time_slots(periods=periods_per_day, period_length=period_length)
    
    all_data = []
    for campus in ["58th Street", "Baltimore Ave"]:
        for day in days:
            for period in range(1, periods_per_day + 1):
                for c in schedule[campus][day][period]:
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
    
    return pd.DataFrame(all_data)

def save_schedule_state():
    """Save current schedule state to JSON-compatible format."""
    return {
        "teachers": st.session_state.teachers,
        "schedule": st.session_state.schedule,
        "teacher_schedules": st.session_state.teacher_schedules,
        "settings": st.session_state.settings,
        "custom_rules": st.session_state.get('custom_rules', {}),
        "unscheduled_classes": st.session_state.get('unscheduled_classes', []),
        "timestamp": datetime.now().isoformat()
    }

def load_schedule_state(state_data):
    """Load schedule state from saved data."""
    st.session_state.teachers = state_data.get("teachers", [])
    st.session_state.schedule = state_data.get("schedule")
    st.session_state.teacher_schedules = state_data.get("teacher_schedules", {})
    st.session_state.settings = state_data.get("settings", BASE_SETTINGS.copy())
    # FIX #6: Also restore custom_rules and unscheduled_classes
    if "custom_rules" in state_data:
        st.session_state.custom_rules = state_data["custom_rules"]
    if "unscheduled_classes" in state_data:
        st.session_state.unscheduled_classes = state_data["unscheduled_classes"]

def teachers_to_csv():
    """Convert teachers to CSV string."""
    if not st.session_state.teachers:
        return ""
    df = pd.DataFrame(st.session_state.teachers)
    return df.to_csv(index=False)

def csv_to_teachers(csv_content):
    """Parse CSV content to teachers list."""
    df = pd.read_csv(io.StringIO(csv_content))
    teachers = []
    for _, row in df.iterrows():
        teachers.append({
            "name": row.get('name', ''),
            "type": row.get('type', 'Core (ELA/Math)'),
            "home_campus": row.get('home_campus', 'Both (Traveling)')
        })
    return teachers

# -----------------------------
# Main UI
# -----------------------------
st.title("K-8 Staffing and Scheduling Helper")
st.markdown("### Cornerstone Christian Academy")

# Campus info
col1, col2 = st.columns(2)
with col1:
    campus_58 = CAMPUSES['58th Street']
    grades_58 = get_campus_grades('58th Street')
    st.info(f"**58th Street Campus ({campus_58['grade_label']})**\n\n{campus_58['address']}\n\n*{grades_58} grades*")
with col2:
    campus_balt = CAMPUSES['Baltimore Ave']
    grades_balt = get_campus_grades('Baltimore Ave')
    st.info(f"**Baltimore Ave Campus ({campus_balt['grade_label']})**\n\n{campus_balt['address']}\n\n*{grades_balt} grades*")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("School Structure")

# Display total grades (read-only since it's campus-dependent)
total_grades = get_total_grades()
st.sidebar.markdown(f"**Total Grades:** {total_grades} (K-8)")
st.sidebar.caption("58th St: K-4 (5 grades) | Balt Ave: 5-8 (4 grades)")

# Check if we need to load base estimate (flag set before widgets)
if st.session_state.get('_load_base', False):
    st.session_state._load_base = False
    st.session_state.settings = BASE_SETTINGS.copy()

if st.session_state.get('_reset_all', False):
    st.session_state._reset_all = False
    st.session_state.settings = BASE_SETTINGS.copy()

homerooms_per_grade = st.sidebar.number_input(
    "Homerooms per grade",
    min_value=1, max_value=10,
    value=st.session_state.settings.get('homerooms_per_grade', 2),
    step=1,
)
st.session_state.settings['homerooms_per_grade'] = homerooms_per_grade

avg_class_size = st.sidebar.number_input(
    "Average students per homeroom",
    min_value=5, max_value=30,
    value=st.session_state.settings.get('avg_class_size', 18),
    step=1,
)
st.session_state.settings['avg_class_size'] = avg_class_size

st.sidebar.header("Instructional Model")

classes_per_week = st.sidebar.number_input(
    "Special classes per week (per homeroom)",
    min_value=1, max_value=5,
    value=st.session_state.settings.get('classes_per_week', 2),
    step=1,
)
st.session_state.settings['classes_per_week'] = classes_per_week

period_length = st.sidebar.number_input(
    "Period length (minutes)",
    min_value=30, max_value=90,
    value=st.session_state.settings.get('period_length', 50),
    step=5,
)
st.session_state.settings['period_length'] = period_length

periods_per_day = st.sidebar.number_input(
    "Periods per day",
    min_value=4, max_value=10,
    value=st.session_state.settings.get('periods_per_day', 8),
    step=1,
)
st.session_state.settings['periods_per_day'] = periods_per_day

full_time_load = st.sidebar.number_input(
    "Full time teaching load per week (minutes)",
    min_value=500, max_value=2000,
    value=st.session_state.settings.get('full_time_load', 1000),
    step=50,
)
st.session_state.settings['full_time_load'] = full_time_load

tipping_min = st.sidebar.number_input(
    "Student-minute tipping point",
    min_value=8000, max_value=20000,
    value=st.session_state.settings.get('tipping_min', 12000),
    step=500,
)
st.session_state.settings['tipping_min'] = tipping_min

st.sidebar.header("Travel")

travel_time = st.sidebar.number_input(
    "Travel time between campuses (minutes)",
    min_value=0, max_value=60,
    value=st.session_state.settings.get('travel_time', 10),
    step=5,
)
st.session_state.settings['travel_time'] = travel_time

travel_buffer_periods = st.sidebar.number_input(
    "Buffer periods needed after travel",
    min_value=0, max_value=2,
    value=st.session_state.settings.get('travel_buffer_periods', 1),
    step=1,
)
st.session_state.settings['travel_buffer_periods'] = travel_buffer_periods

max_switches_per_day = st.sidebar.number_input(
    "Max campus switches per day",
    min_value=0, max_value=6,
    value=st.session_state.settings.get('max_switches_per_day', 2),
    step=1,
)
st.session_state.settings['max_switches_per_day'] = max_switches_per_day

max_switches_per_week = st.sidebar.number_input(
    "Max campus switches per week",
    min_value=0, max_value=20,
    value=st.session_state.settings.get('max_switches_per_week', 6),
    step=1,
)
st.session_state.settings['max_switches_per_week'] = max_switches_per_week

# Quick actions in sidebar
st.sidebar.divider()
st.sidebar.header("Quick Actions")

if st.sidebar.button("Load Base Estimate", use_container_width=True, type="primary"):
    st.session_state._load_base = True
    st.session_state.teachers = copy.deepcopy(BASE_TEACHERS)
    st.session_state.schedule = None
    st.session_state.teacher_schedules = {}
    st.session_state.unscheduled_classes = []
    st.rerun()

if st.sidebar.button("Reset All", use_container_width=True):
    st.session_state._reset_all = True
    st.session_state.teachers = []
    st.session_state.schedule = None
    st.session_state.teacher_schedules = {}
    st.session_state.unscheduled_classes = []
    st.rerun()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Staffing Analysis", 
    "Teacher Configuration",
    "Subject Rules",
    "Schedule Generator",
    "View Schedules",
    "Save/Load"
])

# -----------------------------
# TAB 1: Staffing Analysis
# -----------------------------
with tab1:
    st.subheader("School and Demand Summary")
    
    # Calculate totals using campus-specific grade counts
    homerooms_58th = get_campus_grades('58th Street') * homerooms_per_grade
    homerooms_balt = get_campus_grades('Baltimore Ave') * homerooms_per_grade
    total_homerooms = homerooms_58th + homerooms_balt
    total_students_est = total_homerooms * avg_class_size
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Grades", f"{total_grades}")
    c2.metric("Estimated Students (Both)", f"{total_students_est:,}")
    c3.metric("Total Homerooms", f"{total_homerooms}")
    c4.metric("58th St / Balt Ave", f"{homerooms_58th} / {homerooms_balt}")
    
    st.divider()
    
    st.subheader("Staffing Analysis by Subject")
    
    selected_subject = st.selectbox(
        "Select subject to analyze",
        options=list(st.session_state.custom_rules.keys())
    )
    
    teacher_type = get_teacher_type_rules(selected_subject)
    
    # Editable rules section
    st.markdown(f"**Rules for {selected_subject}:** *(Click to edit)*")
    
    with st.expander("Edit Rules", expanded=False):
        edit_col1, edit_col2 = st.columns(2)
        
        with edit_col1:
            new_traveling = st.checkbox(
                "Can travel between campuses",
                value=teacher_type.is_traveling,
                key=f"travel_{selected_subject}"
            )
            new_max_periods = st.number_input(
                "Max periods per day",
                min_value=1, max_value=10,
                value=teacher_type.max_periods_per_day,
                key=f"max_periods_{selected_subject}"
            )
            new_max_consecutive = st.number_input(
                "Max consecutive periods",
                min_value=1, max_value=6,
                value=teacher_type.max_consecutive_periods,
                key=f"max_consec_{selected_subject}"
            )
        
        with edit_col2:
            new_special_room = st.checkbox(
                "Requires special room",
                value=teacher_type.requires_special_room,
                key=f"special_room_{selected_subject}"
            )
            new_room_type = st.selectbox(
                "Room type needed",
                options=["classroom", "gym", "art", "music", "lab", "library", "chapel"],
                index=["classroom", "gym", "art", "music", "lab", "library", "chapel"].index(teacher_type.room_type) if teacher_type.room_type in ["classroom", "gym", "art", "music", "lab", "library", "chapel"] else 0,
                key=f"room_type_{selected_subject}"
            )
            new_travel_buffer = st.number_input(
                "Buffer periods after travel",
                min_value=0, max_value=3,
                value=teacher_type.min_break_between_travel,
                key=f"travel_buffer_{selected_subject}"
            )
        
        if st.button("Save Rule Changes", key=f"save_rules_{selected_subject}", type="primary"):
            st.session_state.custom_rules[selected_subject] = {
                "is_traveling": new_traveling,
                "max_periods_per_day": new_max_periods,
                "max_consecutive_periods": new_max_consecutive,
                "requires_special_room": new_special_room,
                "room_type": new_room_type,
                "min_break_between_travel": new_travel_buffer
            }
            st.success(f"Rules for {selected_subject} updated!")
            st.rerun()
        
        if st.button("Reset to Default", key=f"reset_rules_{selected_subject}"):
            st.session_state.custom_rules[selected_subject] = DEFAULT_TEACHER_RULES[selected_subject].copy()
            st.success(f"Rules for {selected_subject} reset to defaults!")
            st.rerun()
    
    # Display current rules
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
    
    total_sections_per_week = total_homerooms * classes_per_week
    total_teacher_minutes_required = total_sections_per_week * period_length
    
    if not teacher_type.is_traveling:
        min_teachers_needed = max(2, math.ceil(total_teacher_minutes_required / full_time_load))
        st.warning(f"{selected_subject} teachers cannot travel - need at least one per campus")
    else:
        min_teachers_needed = max(1, math.ceil(total_teacher_minutes_required / full_time_load))
    
    avg_minutes_per_teacher = total_teacher_minutes_required / min_teachers_needed
    avg_periods_per_teacher = avg_minutes_per_teacher / period_length
    
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
    
    # Import from CSV
    st.markdown("#### Import Teachers from CSV")
    uploaded_file = st.file_uploader("Upload teacher CSV", type=['csv'], key="teacher_upload")
    if uploaded_file:
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            imported_teachers = csv_to_teachers(content)
            if st.button("Import Teachers from CSV"):
                st.session_state.teachers = imported_teachers
                st.success(f"Imported {len(imported_teachers)} teachers!")
                st.rerun()
            st.write("Preview:")
            st.dataframe(pd.DataFrame(imported_teachers), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    
    st.divider()
    
    with st.expander("View Current Subject Rules"):
        rules_data = []
        for name, rules in st.session_state.custom_rules.items():
            rules_data.append({
                "Type": name,
                "Can Travel": "Yes" if rules['is_traveling'] else "No",
                "Max/Day": rules['max_periods_per_day'],
                "Max Consecutive": rules['max_consecutive_periods'],
                "Special Room": rules['room_type'] if rules['requires_special_room'] else "No",
                "Travel Buffer": rules['min_break_between_travel']
            })
        st.dataframe(pd.DataFrame(rules_data), use_container_width=True, hide_index=True)
        st.caption("To add or edit subjects, go to the 'Subject Rules' tab")
    
    with st.form("add_teacher"):
        col1, col2, col3 = st.columns(3)
        with col1:
            teacher_name = st.text_input("Teacher Name", placeholder="e.g., Mrs. Johnson")
        with col2:
            # Use dynamic subject list from custom_rules
            teacher_type_select = st.selectbox("Teacher Type", options=list(st.session_state.custom_rules.keys()))
        with col3:
            home_campus = st.selectbox("Home Campus", options=["Both (Traveling)", "58th Street", "Baltimore Ave"])
        
        add_btn = st.form_submit_button("Add Teacher", use_container_width=True)
        
        if add_btn and teacher_name:
            st.session_state.teachers.append({
                "name": teacher_name,
                "type": teacher_type_select,
                "home_campus": home_campus
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Clear All Teachers", type="secondary", use_container_width=True):
                st.session_state.teachers = []
                st.session_state.schedule = None
                st.session_state.teacher_schedules = {}
                st.rerun()
        with col2:
            csv_data = teachers_to_csv()
            st.download_button(
                "Download Teachers CSV",
                data=csv_data,
                file_name="cca_teachers.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col3:
            if st.button("Load Base Teachers", use_container_width=True):
                st.session_state.teachers = BASE_TEACHERS.copy()
                st.rerun()
    else:
        st.info("No teachers configured. Add teachers above or load the base estimate.")
        
        if st.button("Load Base Estimate Teachers", type="primary", use_container_width=True):
            st.session_state.teachers = BASE_TEACHERS.copy()
            st.rerun()

# -----------------------------
# TAB 3: Subject Rules
# -----------------------------
with tab3:
    st.subheader("Edit Subject/Teacher Type Rules")
    
    st.markdown("""
    Customize the scheduling rules for each subject type. These rules control how teachers 
    of each type are scheduled across both campuses. You can also add new subjects or remove existing ones.
    """)
    
    # Display all rules in editable table format
    st.markdown("### Current Subjects")
    
    # Create a dataframe of current rules
    rules_display = []
    for subject, rules in st.session_state.custom_rules.items():
        rules_display.append({
            "Subject": subject,
            "Can Travel": "Yes" if rules['is_traveling'] else "No",
            "Max/Day": rules['max_periods_per_day'],
            "Max Consecutive": rules['max_consecutive_periods'],
            "Special Room": "Yes" if rules['requires_special_room'] else "No",
            "Room Type": rules['room_type'],
            "Travel Buffer": rules['min_break_between_travel']
        })
    
    st.dataframe(pd.DataFrame(rules_display), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Add new subject section
    st.markdown("### Add New Subject")
    
    with st.form("add_subject_form"):
        new_subject_name = st.text_input("Subject Name", placeholder="e.g., Drama, Computer Science, Health")
        
        add_col1, add_col2 = st.columns(2)
        
        with add_col1:
            new_traveling = st.checkbox("Can travel between campuses", value=True)
            new_max_periods = st.number_input("Max periods per day", min_value=1, max_value=10, value=6)
            new_max_consecutive = st.number_input("Max consecutive periods", min_value=1, max_value=6, value=3)
        
        with add_col2:
            new_special_room = st.checkbox("Requires special room", value=False)
            new_room_type = st.selectbox(
                "Room type needed",
                options=["classroom", "gym", "art", "music", "lab", "library", "chapel", "theater", "computer lab"]
            )
            new_travel_buffer = st.number_input("Buffer periods after travel", min_value=0, max_value=3, value=1)
        
        if st.form_submit_button("Add Subject", type="primary", use_container_width=True):
            if new_subject_name:
                if new_subject_name in st.session_state.custom_rules:
                    st.error(f"Subject '{new_subject_name}' already exists!")
                else:
                    st.session_state.custom_rules[new_subject_name] = {
                        "is_traveling": new_traveling,
                        "max_periods_per_day": new_max_periods,
                        "max_consecutive_periods": new_max_consecutive,
                        "requires_special_room": new_special_room,
                        "room_type": new_room_type,
                        "min_break_between_travel": new_travel_buffer
                    }
                    st.success(f"Subject '{new_subject_name}' added!")
                    st.rerun()
            else:
                st.error("Please enter a subject name")
    
    st.divider()
    
    # Remove subject section
    st.markdown("### Remove Subject")
    
    remove_col1, remove_col2 = st.columns([3, 1])
    with remove_col1:
        subject_to_remove = st.selectbox(
            "Select subject to remove",
            options=list(st.session_state.custom_rules.keys()),
            key="remove_subject_select"
        )
    with remove_col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Remove Subject", type="secondary", use_container_width=True):
            # Check if any teachers are using this subject
            teachers_using = [t['name'] for t in st.session_state.teachers if t.get('type') == subject_to_remove]
            if teachers_using:
                st.error(f"Cannot remove '{subject_to_remove}' - it's assigned to: {', '.join(teachers_using)}")
            else:
                del st.session_state.custom_rules[subject_to_remove]
                st.success(f"Subject '{subject_to_remove}' removed!")
                st.rerun()
    
    st.divider()
    
    # Edit individual subject rules
    st.markdown("### Edit Rules for a Subject")
    
    edit_subject = st.selectbox(
        "Select subject to edit",
        options=list(st.session_state.custom_rules.keys()),
        key="edit_subject_select"
    )
    
    current_rules = st.session_state.custom_rules[edit_subject]
    
    with st.form(f"edit_rules_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            edit_traveling = st.checkbox(
                "Can travel between campuses",
                value=current_rules['is_traveling']
            )
            edit_max_periods = st.number_input(
                "Max periods per day",
                min_value=1, max_value=10,
                value=current_rules['max_periods_per_day']
            )
            edit_max_consecutive = st.number_input(
                "Max consecutive periods",
                min_value=1, max_value=6,
                value=current_rules['max_consecutive_periods']
            )
        
        with col2:
            edit_special_room = st.checkbox(
                "Requires special room",
                value=current_rules['requires_special_room']
            )
            room_options = ["classroom", "gym", "art", "music", "lab", "library", "chapel", "theater", "computer lab"]
            edit_room_type = st.selectbox(
                "Room type needed",
                options=room_options,
                index=room_options.index(current_rules['room_type']) if current_rules['room_type'] in room_options else 0
            )
            edit_travel_buffer = st.number_input(
                "Buffer periods after travel",
                min_value=0, max_value=3,
                value=current_rules['min_break_between_travel']
            )
        
        submit_col1, submit_col2 = st.columns(2)
        with submit_col1:
            if st.form_submit_button("Save Changes", type="primary", use_container_width=True):
                st.session_state.custom_rules[edit_subject] = {
                    "is_traveling": edit_traveling,
                    "max_periods_per_day": edit_max_periods,
                    "max_consecutive_periods": edit_max_consecutive,
                    "requires_special_room": edit_special_room,
                    "room_type": edit_room_type,
                    "min_break_between_travel": edit_travel_buffer
                }
                st.success(f"Rules for {edit_subject} saved!")
                st.rerun()
        
        with submit_col2:
            if st.form_submit_button("Reset to Default", use_container_width=True):
                if edit_subject in DEFAULT_TEACHER_RULES:
                    st.session_state.custom_rules[edit_subject] = DEFAULT_TEACHER_RULES[edit_subject].copy()
                    st.success(f"Rules for {edit_subject} reset to defaults!")
                    st.rerun()
                else:
                    st.warning(f"No default rules for '{edit_subject}' - this is a custom subject")
    
    st.divider()
    
    # Reset and export section
    st.markdown("### Manage All Subjects")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reset to Default Subjects", type="secondary", use_container_width=True):
            st.session_state.custom_rules = {k: v.copy() for k, v in DEFAULT_TEACHER_RULES.items()}
            st.success("All subjects reset to defaults!")
            st.rerun()
    
    with col2:
        # Export rules as JSON
        rules_json = json.dumps(st.session_state.custom_rules, indent=2)
        st.download_button(
            "Download Subjects (JSON)",
            data=rules_json,
            file_name="cca_subject_rules.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Export as CSV
        rules_csv_data = []
        for subject, rules in st.session_state.custom_rules.items():
            rules_csv_data.append({
                "subject": subject,
                "is_traveling": rules['is_traveling'],
                "max_periods_per_day": rules['max_periods_per_day'],
                "max_consecutive_periods": rules['max_consecutive_periods'],
                "requires_special_room": rules['requires_special_room'],
                "room_type": rules['room_type'],
                "min_break_between_travel": rules['min_break_between_travel']
            })
        rules_csv = pd.DataFrame(rules_csv_data).to_csv(index=False)
        st.download_button(
            "Download Subjects (CSV)",
            data=rules_csv,
            file_name="cca_subject_rules.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Import rules
    st.markdown("### Import Subjects")
    
    import_col1, import_col2 = st.columns(2)
    
    with import_col1:
        uploaded_rules = st.file_uploader("Upload subjects JSON", type=['json'], key="rules_upload")
        if uploaded_rules:
            try:
                content = uploaded_rules.getvalue().decode('utf-8')
                imported_rules = json.loads(content)
                st.write("Preview:")
                st.json(imported_rules)
                if st.button("Apply Imported Rules (JSON)"):
                    st.session_state.custom_rules.update(imported_rules)
                    st.success("Subjects imported!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading JSON: {e}")
    
    with import_col2:
        uploaded_csv = st.file_uploader("Upload subjects CSV", type=['csv'], key="rules_csv_upload")
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                st.write("Preview:")
                st.dataframe(df.head(), use_container_width=True, hide_index=True)
                if st.button("Apply Imported Rules (CSV)"):
                    for _, row in df.iterrows():
                        subject_name = row['subject']
                        st.session_state.custom_rules[subject_name] = {
                            "is_traveling": bool(row.get('is_traveling', True)),
                            "max_periods_per_day": int(row.get('max_periods_per_day', 6)),
                            "max_consecutive_periods": int(row.get('max_consecutive_periods', 3)),
                            "requires_special_room": bool(row.get('requires_special_room', False)),
                            "room_type": str(row.get('room_type', 'classroom')),
                            "min_break_between_travel": int(row.get('min_break_between_travel', 1))
                        }
                    st.success(f"Imported {len(df)} subjects!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

# -----------------------------
# TAB 4: Schedule Generator
# -----------------------------
with tab4:
    st.subheader("Generate Weekly Schedule")
    
    if not st.session_state.teachers:
        st.warning("Please add teachers in the 'Teacher Configuration' tab first.")
    else:
        st.markdown(f"""
**Schedule Parameters:**
- Total Grades: {total_grades} (58th St: K-4, Balt Ave: 5-8) | Homerooms per grade: {homerooms_per_grade}
- Periods per day: {periods_per_day} | Period length: {period_length} minutes
- Specials per homeroom per week: {classes_per_week}
- Full-time load cap: {full_time_load} minutes/week
- Travel time: {travel_time} min | Buffer: {travel_buffer_periods} periods | Max switches/day: {max_switches_per_day} | Max switches/week: {max_switches_per_week}
""")
        
        st.markdown("**Teachers configured:**")
        for t in st.session_state.teachers:
            st.write(f"- {t['name']} ({t['type']}) - {t['home_campus']}")
        
        st.divider()
        
        if st.button("Generate Schedule", type="primary", use_container_width=True):
            with st.spinner("Generating optimized schedule..."):
                schedule, teacher_schedules, unscheduled = generate_schedule(
                    campuses=CAMPUSES,
                    homerooms_per_grade=homerooms_per_grade,
                    teachers_config=st.session_state.teachers,
                    periods_per_day=periods_per_day,
                    travel_time_periods=travel_buffer_periods,
                    classes_per_week=classes_per_week,
                    custom_rules=st.session_state.get('custom_rules'),
                    period_length=period_length,
                    full_time_load=full_time_load,
                    travel_time=travel_time,
                    max_switches_per_day=max_switches_per_day,
                    max_switches_per_week=max_switches_per_week
                )
                st.session_state.schedule = schedule
                st.session_state.teacher_schedules = teacher_schedules
                st.session_state.unscheduled_classes = unscheduled
                
                if not unscheduled:
                    st.success("✅ Schedule generated successfully! All classes scheduled.")
                else:
                    st.warning(f"⚠️ Schedule generated with conflicts: {len(unscheduled)} class(es) could not be scheduled.")
                st.rerun()
        
        if st.session_state.schedule:
            unscheduled = st.session_state.get('unscheduled_classes', [])
            if unscheduled:
                st.warning(f"⚠️ **{len(unscheduled)} class(es) could not be scheduled.** Consider:")
                st.markdown("""
- Adding more teachers
- Increasing periods per day
- Reducing classes per week per homeroom
- Adjusting teacher rules (max periods, travel constraints)
""")
                with st.expander("View Unscheduled Classes", expanded=False):
                    unscheduled_df = pd.DataFrame(unscheduled)
                    if not unscheduled_df.empty:
                        unscheduled_df = unscheduled_df[['campus', 'grade', 'homeroom', 'teacher_name', 'teacher_type']]
                        unscheduled_df.columns = ['Campus', 'Grade', 'Homeroom', 'Teacher', 'Subject']
                        st.dataframe(unscheduled_df, use_container_width=True)
            else:
                st.success("✅ Schedule is ready! All classes scheduled. View it in the 'View Schedules' tab.")

# -----------------------------
# TAB 5: View Schedules
# -----------------------------
with tab5:
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
                
                teacher_info = next((t for t in st.session_state.teachers if t['name'] == selected_teacher), None)
                if teacher_info:
                    teacher_type = TEACHER_TYPES[teacher_info['type']]
                    st.caption(f"Type: {teacher_info['type']} | Home: {teacher_info['home_campus']} | Can Travel: {'Yes' if teacher_type.is_traveling else 'No'}")
                
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
                
                total_periods = sum(
                    1 for day in days 
                    for p in range(1, periods_per_day + 1)
                    if st.session_state.teacher_schedules.get(selected_teacher, {}).get(day, {}).get(p)
                )
                total_minutes = total_periods * period_length
                
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
                                "Teacher": c['teacher'].split()[0],
                                "Subject": c['teacher_type'].split('/')[0]
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            schedule_df = schedule_to_dataframe(
                st.session_state.schedule, 
                st.session_state.teacher_schedules,
                periods_per_day, 
                period_length
            )
            if not schedule_df.empty:
                csv = schedule_df.to_csv(index=False)
                st.download_button(
                    "Download Full Schedule (CSV)",
                    data=csv,
                    file_name="cca_schedule.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
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
        
        with col3:
            # Excel export with multiple sheets
            if not schedule_df.empty:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    schedule_df.to_excel(writer, index=False, sheet_name='Full Schedule')
                    if teacher_data:
                        pd.DataFrame(teacher_data).to_excel(writer, index=False, sheet_name='Teacher Schedules')
                    pd.DataFrame(st.session_state.teachers).to_excel(writer, index=False, sheet_name='Teachers')
                
                st.download_button(
                    "Download All (Excel)",
                    data=buffer.getvalue(),
                    file_name="cca_complete_schedule.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# -----------------------------
# TAB 6: Save/Load
# -----------------------------
with tab6:
    st.subheader("Save and Load Schedules")
    
    st.markdown("""
    Save your current schedule configuration to a file, or load a previously saved schedule.
    This includes all teachers, settings, and the generated schedule.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Save Current Schedule")
        
        schedule_name = st.text_input("Schedule Name", value=f"CCA_Schedule_{datetime.now().strftime('%Y%m%d')}")
        
        if st.button("Save Schedule to Memory", use_container_width=True):
            if st.session_state.schedule:
                st.session_state.saved_schedules[schedule_name] = save_schedule_state()
                st.success(f"Saved '{schedule_name}' to memory!")
            else:
                st.warning("No schedule to save. Generate one first.")
        
        st.divider()
        
        # Download as JSON
        if st.session_state.schedule or st.session_state.teachers:
            state_data = save_schedule_state()
            json_str = json.dumps(state_data, indent=2, default=str)
            st.download_button(
                "Download Schedule (JSON)",
                data=json_str,
                file_name=f"{schedule_name}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        st.markdown("### Load Schedule")
        
        # Load from memory
        if st.session_state.saved_schedules:
            selected_saved = st.selectbox(
                "Load from saved schedules",
                options=list(st.session_state.saved_schedules.keys())
            )
            
            if st.button("Load Selected Schedule", use_container_width=True):
                load_schedule_state(st.session_state.saved_schedules[selected_saved])
                st.success(f"Loaded '{selected_saved}'!")
                st.rerun()
        else:
            st.info("No schedules saved in memory yet.")
        
        st.divider()
        
        # Upload JSON
        uploaded_json = st.file_uploader("Upload schedule JSON", type=['json'])
        if uploaded_json:
            try:
                content = uploaded_json.getvalue().decode('utf-8')
                state_data = json.loads(content)
                
                st.write("Preview:")
                st.write(f"- Teachers: {len(state_data.get('teachers', []))}")
                st.write(f"- Has Schedule: {'Yes' if state_data.get('schedule') else 'No'}")
                st.write(f"- Saved: {state_data.get('timestamp', 'Unknown')}")
                
                if st.button("Load Uploaded Schedule", use_container_width=True, type="primary"):
                    load_schedule_state(state_data)
                    st.success("Schedule loaded!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading JSON: {e}")
    
    st.divider()
    
    # Import schedule from CSV
    st.markdown("### Import Schedule from CSV")
    st.markdown("Upload a previously exported schedule CSV to view or modify it.")
    
    uploaded_schedule_csv = st.file_uploader("Upload schedule CSV", type=['csv'], key="schedule_csv_upload")
    if uploaded_schedule_csv:
        try:
            df = pd.read_csv(uploaded_schedule_csv)
            st.write("Imported Schedule Preview:")
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
            
            st.info(f"Total rows: {len(df)} | Columns: {', '.join(df.columns)}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    
    st.divider()
    
    # Base estimate info
    st.markdown("### Base Estimate (Default CCA Configuration)")
    
    st.markdown("**Default Teachers:**")
    base_df = pd.DataFrame(BASE_TEACHERS)
    st.dataframe(base_df, use_container_width=True, hide_index=True)
    
    st.markdown("**Default Settings:**")
    settings_df = pd.DataFrame([BASE_SETTINGS])
    st.dataframe(settings_df, use_container_width=True, hide_index=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Cornerstone Christian Academy - Scheduling Optimization Tool | 58th St & Baltimore Ave Campuses")
