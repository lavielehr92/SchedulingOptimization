# CCA Staffing & Scheduling Helper

A Streamlit application for Cornerstone Christian Academy to estimate teacher load and staffing needs across two campuses.

## Features

- **Flexible Inputs**: Adjust grades, homerooms, class sizes, and more
- **Teacher Load Calculation**: Based on 1000 teaching minutes/week optimal load
- **Staffing Recommendations**: Automatic calculation of minimum teachers needed
- **Student-Minute Tipping Point**: 12k-13k threshold for adding staff
- **Multi-Campus Support**: Travel time considerations for two campuses
- **Scenario Testing**: "What-if" slider to test different staffing levels

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Period Length | 50 min | Duration of each class period |
| Full-Time Load | 1000 min/week | Optimal teaching minutes per teacher |
| Tipping Point | 12,000 student-min | Threshold for hiring additional teacher |
| Travel Time | 10 min | Time between campuses |

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This app is deployed on Streamlit Cloud. Visit the live app at:
https://schedulingoptimization.streamlit.app

## License

MIT License - Cornerstone Christian Academy
