import random
from datetime import datetime, timedelta
import pandas as pd

print("\n--- Generating Improved Synthetic Dataset ---")

# Define categories, priorities, and users
categories = ["Bug", "Feature", "Design", "Testing", "Documentation", "DevOps", "Research"]
priorities = ["Low", "Medium", "High"]
users = [f"user_{i}" for i in range(1, 21)]
projects = ["Eagle", "Viper", "Phoenix", "Omega"]

# Improved task templates with keywords for rule-based labeling
task_templates = [
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
    "Migrate the user data from the old server to the new cloud infrastructure",
    "Develop the new user authentication module using OAuth 2.0"
]

# Rule-based category assignment
def assign_category(description):
    desc = description.lower()
    if "bug" in desc or "fix" in desc or "issue" in desc:
        return "Bug"
    elif "design" in desc or "ui" in desc or "ux" in desc or "mockup" in desc:
        return "Design"
    elif "test" in desc or "case" in desc or "execute" in desc:
        return "Testing"
    elif "doc" in desc or "documentation" in desc or "policy" in desc:
        return "Documentation"
    elif "deploy" in desc or "ci/cd" in desc or "devops" in desc or "pipeline" in desc:
        return "DevOps"
    elif "research" in desc or "analyze" in desc or "audit" in desc:
        return "Research"
    elif "feature" in desc or "implement" in desc or "develop" in desc or "build" in desc or "optimize" in desc:
        return "Feature"
    else:
        return random.choice(categories)

# Rule-based priority assignment
def assign_priority(description, workload):
    desc = description.lower()
    # High priority for critical, bug, fix, security, urgent, or high workload
    if any(word in desc for word in ["critical", "urgent", "security", "bug", "fix"]) or workload > 12:
        return "High"
    # Medium for optimize, improve, update, migrate, or moderate workload
    elif any(word in desc for word in ["optimize", "improve", "update", "migrate", "refactor"]) or 7 < workload <= 12:
        return "Medium"
    # Low for documentation, design, or low workload
    elif any(word in desc for word in ["doc", "documentation", "design", "mockup", "analyze", "summary"]) or workload <= 7:
        return "Low"
    else:
        return random.choices(priorities, weights=[1,2,2])[0]

# Generate synthetic dataset
data = []
for i in range(1, 3001):
    template = random.choice(task_templates)
    project = random.choice(projects)
    assigned_user = random.choice(users)
    days_offset = random.randint(0, 60)
    due_date = (datetime.now() + timedelta(days=days_offset)).strftime("%Y-%m-%d")
    workload = random.randint(1, 15)
    description = f"{template} for project '{project}'. Assigned to {assigned_user}."
    category = assign_category(template)
    priority = assign_priority(template, workload)
    data.append([
        f"TSK-{i:04d}", description, category, priority, assigned_user, due_date, workload
    ])

df = pd.DataFrame(data, columns=[
    "task_id", "task_description", "category", "priority", "assigned_user", "due_date", "user_workload"
])

# Balance classes (optional: upsample minority classes)
# You can add code here to balance if needed

# Save to CSV
df.to_csv("synthetic_task_dataset.csv", index=False)
print("Improved synthetic_task_dataset.csv created with 3000 samples.")