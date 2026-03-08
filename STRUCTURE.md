# מבנה הפרויקט

## חוקים: .gitignore, .dockerignore, .env / .env.example

- **.gitignore** – **אחד לכל הפרויקט** (בשורש). קובע מה לא נכנס ל-commit בכל התיקיות.
- **.dockerignore** – **אחד לבקאנד ואחד לפרונט**: `backend/.dockerignore`, `frontend/.dockerignore`. כל שירות קובע מה לא נשלח ל-build של ה-Docker שלו.
- **.env.example** ו-**.env** (הקובץ האמיתי):
  - **משותף** (בקאנד + פרונט או compose) → **בשורש**: `.env.example` / `.env`
  - **ייחודי לבקאנד** → **backend/.env.example** / **backend/.env**
  - **ייחודי לפרונט** → **frontend/.env.example** / **frontend/.env**  
  docker-compose טוען: `env_file: [.env, backend/.env]` (שורש + בקאנד).

---

## קבצים בשורש

| קובץ | תפקיד |
|------|--------|
| **README.md** | תיעוד לכל הפרויקט |
| **.gitignore** | אחד לכל הפרויקט – מה לא ב-commit |
| **.dockerignore** | (לשימוש ב-build משורש, אם יש) |
| **.env.example** | משתנים **משותפים** בלבד (להעתקה ל-.env) |
| **requirements.txt**, **pyproject.toml**, **docker-compose.yml** | |

---

## תיקיות – כל שירות עם .dockerignore ו-.env.example משלו

| תיקייה | תוכן |
|--------|--------|
| **backend/** | **Dockerfile**, **.dockerignore**, **.env.example** (משתנים ייחודיים לבקאנד). קוד: `src/`. |
| **frontend/** | **.dockerignore**, **.env.example** (משתנים ייחודיים לפרונט). `index.html`, README. אין Dockerfile כרגע (סטטי). |
| **scripts/** | סקריפטים (אינדוקס, RAG, CLI) – רצים משורש, מייבאים מ-`backend`. |
| **examples/** | קבצי דוגמה. |
| **exports/** | ייצוא טקסטים (לא ב-Git). |
| **chroma_db/** | DB של Chroma (לא ב-Git). |
| **tests/** | בדיקות. |

---

## הרצה

- **מקומי:** משורש `uv sync` (או `pip install -r requirements.txt`), אז `cd backend && uvicorn src.app:app --reload --port 8000`.
- **Docker:** משורש `docker compose up --build` → API ב-http://localhost:8000. לפתוח `frontend/index.html` בדפדפן.
