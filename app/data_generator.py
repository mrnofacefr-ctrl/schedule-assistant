"""
data_generator.py
------------------
Generates realistic sample schedule data (meetings, workshops, tasks,
appointments) for the next 30 days and loads it into the SQLite store.
Run directly to (re)seed the database:  python -m app.data_generator
"""

import random
from datetime import datetime, timedelta

from app import db

random.seed(42)  # reproducible sample data

MEETING_TITLES = [
    "Weekly Team Sync", "Client Check-in Call", "1:1 with Manager",
    "Sprint Planning", "Budget Review Meeting", "Product Roadmap Discussion",
    "Marketing Strategy Sync", "Vendor Negotiation Call", "Board Update Meeting",
    "Cross-team Standup",
]
WORKSHOP_TITLES = [
    "Advanced Excel Workshop", "Public Speaking Workshop", "AI Tools Bootcamp",
    "Design Thinking Workshop", "Negotiation Skills Training", "Leadership Workshop",
    "Cold Email Outreach Masterclass", "SEO Fundamentals Workshop",
]
TASK_TITLES = [
    "Finish quarterly report", "Review pull requests", "Prepare client proposal",
    "Update project documentation", "Send invoices", "Research competitor pricing",
    "Draft blog post outline", "Organize expense receipts", "Follow up on leads",
    "Plan next sprint backlog",
]
APPOINTMENT_TITLES = [
    "Dentist Appointment", "Annual Health Checkup", "Haircut Appointment",
    "Car Servicing", "Bank Appointment", "Eye Checkup", "Physiotherapy Session",
]

LOCATIONS = ["Zoom", "Google Meet", "Conference Room A", "Conference Room B",
             "Office HQ", "Client Office", "Home", "Downtown Clinic", "Online"]
ATTENDEES_POOL = ["Alex", "Priya", "Sam", "Jordan", "Maria", "Client Team",
                   "Manager", "Design Team", "Marketing Team"]


def _rand_time_slot():
    start_hour = random.choice([9, 10, 11, 13, 14, 15, 16])
    start_min = random.choice([0, 30])
    duration_min = random.choice([30, 60, 90])
    start = datetime(2000, 1, 1, start_hour, start_min)
    end = start + timedelta(minutes=duration_min)
    return start.strftime("%H:%M"), end.strftime("%H:%M")


def generate_sample_schedule(start_date: datetime, num_days: int = 30) -> list[dict]:
    entries = []
    for day_offset in range(num_days):
        day = start_date + timedelta(days=day_offset)
        weekday = day.weekday()  # 0=Mon ... 6=Sun
        date_str = day.strftime("%Y-%m-%d")

        # Weekends: lighter schedule (mostly personal appointments/tasks)
        if weekday >= 5:
            if random.random() < 0.4:
                start, end = _rand_time_slot()
                kind = random.choice(["appointment", "task"])
                title = random.choice(APPOINTMENT_TITLES if kind == "appointment" else TASK_TITLES)
                entries.append({
                    "title": title,
                    "type": kind,
                    "date": date_str, "start_time": start, "end_time": end,
                    "location": random.choice(LOCATIONS),
                    "description": "Personal weekend item.",
                    "attendees": "",
                })
            continue

        # Weekdays: 2-4 items mixing meetings/workshops/tasks/appointments
        num_items = random.randint(2, 4)
        used_slots = []
        for _ in range(num_items):
            kind = random.choices(
                ["meeting", "workshop", "task", "appointment"],
                weights=[0.45, 0.15, 0.3, 0.1],
            )[0]
            start, end = _rand_time_slot()
            # avoid exact duplicate start times on the same day
            attempts = 0
            while start in used_slots and attempts < 5:
                start, end = _rand_time_slot()
                attempts += 1
            used_slots.append(start)

            if kind == "meeting":
                title = random.choice(MEETING_TITLES)
                attendees = ", ".join(random.sample(ATTENDEES_POOL, k=random.randint(1, 3)))
                location = random.choice(["Zoom", "Google Meet", "Conference Room A", "Conference Room B"])
                desc = f"Discuss updates related to {title.lower()}."
            elif kind == "workshop":
                title = random.choice(WORKSHOP_TITLES)
                attendees = ", ".join(random.sample(ATTENDEES_POOL, k=random.randint(2, 4)))
                location = random.choice(["Conference Room A", "Online", "Office HQ"])
                desc = f"Skill-building session: {title}."
            elif kind == "task":
                title = random.choice(TASK_TITLES)
                attendees = ""
                location = "N/A"
                desc = f"Deliverable: {title}."
            else:
                title = random.choice(APPOINTMENT_TITLES)
                attendees = ""
                location = random.choice(["Downtown Clinic", "Home", "Office HQ"])
                desc = f"Personal appointment: {title}."

            entries.append({
                "title": title, "type": kind, "date": date_str,
                "start_time": start, "end_time": end,
                "location": location, "description": desc, "attendees": attendees,
            })
    return entries


def seed_database(reset: bool = True):
    db.init_db()
    if reset:
        db.clear_all()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    sample = generate_sample_schedule(today, num_days=30)
    for entry in sample:
        db.insert_entry(entry)
    return len(sample)


if __name__ == "__main__":
    count = seed_database(reset=True)
    print(f"Seeded {count} schedule entries for the next 30 days into {db.DB_PATH}")
