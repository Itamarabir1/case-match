# מבנה הפרויקט

קבצי הפרויקט המרכזיים **בשורש** (מקובל בתעשייה):

| קובץ / תיקייה | תפקיד |
|----------------|--------|
| **README.md** | תיעוד לכל הפרויקט (לא רק לבקאנד) |
| **requirements.txt** | תלויות Python לכל הפרויקט |
| **pyproject.toml** | הגדרת הפרויקט Python (שם, גרסה, תלויות) |
| **.dockerignore** | מה לא לשלוח ל-Docker build (פרויקט כולו) |
| **.gitignore** | מה לא לעשות commit |
| **docker-compose.yml** | הרצת שירותים (כרגע backend). |
| **.env.example** | דוגמה למשתני סביבה בשורש (להעתקה ל-.env ליד docker-compose). |

---

## תיקיות – כל שירות עם Dockerfile ו-.env.example משלו

| תיקייה | תוכן |
|--------|--------|
| **backend/** | קוד ה-API: `src/` (כולל `src/config/` – הגדרות מ-.env). **Dockerfile** + **requirements.txt** + **.env.example** + **.dockerignore**. |
| **frontend/** | `index.html` + README. **.env.example** (כרגע סטטי; אם יהיה build – משתנים כאן). אין Dockerfile כרגע (ממשק סטטי). |
| **scripts/** | סקריפטים (אינדוקס, RAG, CLI) – רצים משורש, מייבאים מ-`backend`. |
| **examples/** | קבצי דוגמה. |
| **exports/** | ייצוא טקסטים (לא ב-Git). |
| **chroma_db/** | DB של Chroma (לא ב-Git). |
| **tests/** | בדיקות. |

---

## הרצה

- **מקומי:** משורש `uv sync` (או `pip install -r requirements.txt`), אז `cd backend && uvicorn src.app:app --reload --port 8000`.
- **Docker:** משורש `docker compose up --build` → API ב-http://localhost:8000. לפתוח `frontend/index.html` בדפדפן.
