import random
from datetime import datetime, timedelta
import pandas as pd
# ==============================================================================
# DATASET CREATION: Generating a synthetic dataset of 3000 samples
# ==============================================================================
print("\n--- Generating Synthetic Dataset ---")

task_templates = [
    "Develop the new user authentication module using OAuth 2.0",
    "Fix the critical bug in the payment processing gateway API",
    "Design the UI/UX mockups for the new dashboard homepage",
    "Write comprehensive API documentation for the new inventory management endpoints",
    "Create and execute end-to-end test cases for the user profile feature",
    "Refactor the legacy database schema to improve performance",
    "Implement a real-time notification system using WebSockets",
    "Optimize the front-end asset loading time for mobile devices",
    "Set up a new CI/CD pipeline for the staging environment",
    "Analyze user feedback from the latest survey and create a summary report",
    "Update the privacy policy to comply with new GDPR regulations",
    "Resolve the CSS issues causing layout breaks on Safari browsers",
    "Build a data visualization component for the analytics dashboard",
    "Conduct a security audit of the entire application stack",
    "Migrate the user data from the old server to the new cloud infrastructure"
]

categories = ["Bug", "Feature", "Documentation", "Testing", "DevOps", "Design", "Research"]
priorities = ["Low", "Medium", "High"]
users = [f"user_{i}" for i in range(1, 21)]

data = []
for i in range(3000):
    user = random.choice(users)
    deadline = datetime.now() + timedelta(days=random.randint(1, 60))
    workload = random.randint(1, 15)
    category = random.choice(categories)

    if category in ["Bug", "DevOps"]:
        priority = "High"
    elif workload > 10:
        priority = random.choice(["Medium", "High"])
    else:
        priority = random.choice(priorities)

    task_description = f"{random.choice(task_templates)} for project '{random.choice(['Phoenix', 'Viper', 'Omega', 'Eagle'])}'. Assigned to {user}."
    data.append({
        "task_id": f"TSK-{i+1000}",
        "task_description": task_description,
        "category": category,
        "priority": priority,
        "assigned_user": user,
        "deadline": deadline.strftime("%Y-%m-%d"),
        "user_workload": workload
    })

df_generated = pd.DataFrame(data) # Renamed DataFrame to avoid conflict
df_generated.to_csv("synthetic_task_dataset.csv", index=False)
print("Synthetic dataset created and saved as 'synthetic_task_dataset.csv'.")