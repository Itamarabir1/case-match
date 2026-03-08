# Law Retrieval (case-match)

**זה קובץ ה-README הראשי של הפרויקט** – תיעוד, מבנה והרצה (backend, frontend, scripts).  
**Repository:** [github.com/Itamarabir1/case-match](https://github.com/Itamarabir1/case-match)

## 1. סיכום השיחה

### הרעיון המרכזי

- **מטרה:** מערכת שמקבלת **בעיה משפטית** (טקסט) ומחזירה **תיקים/פסקי דין דומים** ואת מה שנפסק שם – **לא** חיזוי "תזכה/תפסיד".
- **טכניקה:** חיפוש סמנטי (וקטורי) על מאגר תיקים אנגלי, עם אופציה לחיפוש היברידי (וקטור + BM25) ולאחר מכן הצגת תוצאות (ואם תרצה – סיכום עם LLM).

### בחירות טכניות

| נושא | ההחלטה |
|------|---------|
| **מאגר נתונים** | **CourtListener API** – תיקים/פסקי דין באנגלית; אינדוקס דרך `scripts/rebuild_first_courtlistener_cases.py`. |
| **מה נשמר אצלך** | **וקטורים + טקסט ה-chunks** ב-Chroma; טקסט מלא ווקטורים גם ב-`exports/courtlistener_first_cases/`. |
| **טעינת נתונים** | **Streaming** מ-CourtListener API (עם checkpoint ו-`--resume` להמשך). |
| **מגבלת 512 טוקנים** | תיקים ארוכים → **chunking**. לא לחתוך בעיוור; להשתמש ב-**RecursiveCharacterTextSplitter** עם separators: `["\n\n", "\n", ". ", " "]` כדי לחתוך בין פסקאות/משפטים. |
| **גודל chunk** | ~400–450 טוקנים (או ~1,200–1,600 תווים), עם **overlap** 50–100 טוקנים. |
| **מודל embedding** | **sentence-transformers**, אנגלית + CPU: למשל `all-MiniLM-L6-v2` (קל) או `all-mpnet-base-v2` (קצת יותר איכותי). |
| **מאגר וקטורים** | **Chroma** להתחלה; אם יהיו הרבה מאוד chunks – לשקול **FAISS**. |
| **סדר פיתוח** | קודם **retrieval פשוט** (chunk → embed → store → query → aggregate). רק אחרי שזה עובד – להוסיף **Small-to-Big** (אם צריך LLM עם הקשר גדול) ו-**Hybrid + RRF**. |
| **Aggregation** | כשמקבצים לפי תיק (doc_id): **לא** רק "כמה chunks מתוך ה-top" – לשקלל **ממוצע ציוני הדמיון** של ה-chunks של כל תיק. |
| **Validation** | אחרי שליפה: לסנן chunks ריקים או **קצרים מדי** (למשל פחות מ-5 מילים או 50 תווים). |
| **Hybrid Search** | **BM25** (מילות מפתח: מספר חוק, שם שופט) + **חיפוש וקטורי**; מיזוג תוצאות עם **RRF** (Reciprocal Rank Fusion). |
| **Parallel** | בבניית האינדקס – **multiprocessing** (או workers) כדי לנצל CPU; לשים לב ל-RAM ולבלעדיות כתיבה ל-DB. |
| **Small-to-Big** | אופציונלי: chunks קטנים (למשל 200 מילים) לאינדקס, ו-**parent** גדול (למשל 1000 מילים) לשליחה ל-LLM – דיוק בחיפוש + הקשר מלא. |

---

## 2. עקרונות ארכיטקטורה ועיצוב

- **Pydantic:** מודלים ל-Document, Chunk, Query, SearchResult וכו' – ולידציה על קלט/פלט.
- **Pipeline ברור:** שלבי אינדקס ושלבי query מוגדרים (פונקציות/מודולים נפרדים).
- **System prompt:** אם יש LLM – פרומפט קבוע (תפקיד, פורמט, מגבלות) בקובץ/משתנה נפרד.
- **JSON Schema:** תיאור ה-API (בקשה/תגובה) – לתיעוד וולידציה.
- **Config:** קובץ (YAML/JSON/env) למודל, גודל chunk, RRF, נתיבי DB – בלי magic numbers בקוד.
- **לוגינג:** לוגים מובנים (למשל JSON) עם request_id, שלב, זמנים.
- **אי-אמון:** ולידציה וסניטציה על קלט; בדיקת chunks שחזרו לפני הצגה.

---

## 3. איך הפרויקט נראה (תצוגה מלמעלה)

### 3.1 מבנה הפרויקט (שורש = מקום אחד לכל ההגדרות)

```
Law/
├── README.md              ← תיעוד הפרויקט (קובץ אחד לכל הפרויקט)
├── .env.example
├── .gitignore
├── .dockerignore          ← מה לא להעתיק ל-Docker (שורש)
├── requirements.txt       ← תלויות Python (פרויקט – מקומי)
├── pyproject.toml         ← הגדרת הפרויקט Python (פרויקט)
├── docker-compose.yml
├── render.yaml
├── backend/               ← שירות API: Dockerfile + .env.example + קוד
│   ├── src/
│   │   ├── api/, controllers/, middlewares/, services/, domain/,
│   │   ├── repositories/, schemas/, infrastructure/, config/, utils/, pipeline/
│   │   └── app.py
│   ├── Dockerfile         ← בניית image ה-backend
│   ├── requirements.txt
│   ├── .env.example       ← משתני סביבה של ה-backend
│   ├── .dockerignore
│   └── README.md
├── frontend/
│   ├── index.html
│   ├── .env.example       ← משתני סביבה של הפרונט (אופציונלי)
│   └── README.md
├── scripts/
│   ├── rebuild_first_courtlistener_cases.py
│   ├── query_cli.py, match_case_file.py, find_best_match.py
│   ├── decision_support.py, rag_analysis.py, show_index.py, reset_index_and_data.py
│   └── ...
├── examples/
├── exports/
├── chroma_db/
└── tests/
```

### 3.2 זרימת בניית האינדקס (CourtListener)

1. טעינת תיקים מ-**CourtListener API** (streaming, עם checkpoint להמשך).
2. לכל תיק: חילוץ טקסט → **chunking** (RecursiveCharacterTextSplitter, separators, גודל/overlap מתוך config).
3. **Embedding** לכל chunk (batch).
4. שמירה ב-**Chroma**: id, embedding, metadata (doc_id, chunk_index, text, title, citation, court, date_filed וכו').
5. ייצוא טקסט מלא ווקטורים ל-`exports/courtlistener_first_cases/`.

האינדקס נשמר ב-`chroma_db/`; טקסטים מלאים ב-`exports/`.

### 3.3 זרימת חיפוש (בכל שאילתה)

1. **קלט:** טקסט בעיה משפטית (ולידציה עם Pydantic – לא ריק, אורך מקסימלי).
2. **Embedding** לשאילתה.
3. **חיפוש:** וקטורי: top-k chunks מ-Chroma; אם hybrid: גם BM25 → מיזוג עם **RRF**.
4. **Validation:** השלכת chunks קצרים מדי/ריקים.
5. **Aggregation:** קיבוץ לפי `doc_id`, ציון תיק = **ממוצע ציוני** ה-chunks של אותו תיק.
6. **פלט:** רשימת תיקים ממוינת (doc_id, ציון, אולי טקסט/ציטוטים מה-chunks).
7. (אופציונלי) שליחה ל-**LLM** עם system prompt – רק להצגה/סיכום, לא לחיזוי תוצאה.

### 3.4 רכיבים עיקריים בקוד

- **schemas (Pydantic):** `Document`, `Chunk`, `SearchQuery`, `SearchResult`, `RankedCase`.
- **chunking:** פונקציה שמקבלת טקסט ומחזירה רשימת `Chunk` (עם doc_id, index, text).
- **embedding:** טעינת sentence-transformers פעם אחת; פונקציה `embed(texts)`.
- **store:** init Chroma, `add_chunks(chunks)`, `search(query_embedding, k)`.
- **retrieval:** `search(query_text)` – embed → search → validate → aggregate → `list[RankedCase]`.
- **scripts/rebuild_first_courtlistener_cases.py:** stream מ-CourtListener → chunking → embed → store + export.
- **pipeline/query:** חיבור query → retrieval → תצוגה/API.

---

## 4. סיכום בכמה שורות

- **מה בונים:** מערכת retrieval על תיקים משפטיים באנגלית – "הכנס בעיה משפטית, קבל תיקים דומים ומה נפסק".
- **איפה הנתונים:** CourtListener API; אצלך Chroma (וקטורים + metadata) + ייצוא ל-`exports/`.
- **איך:** chunking חכם (RecursiveCharacterTextSplitter), embedding (sentence-transformers, CPU), Chroma, aggregation לפי ממוצע ציונים, validation על chunks; אופציונלי: Hybrid + RRF, Small-to-Big, LLM עם system prompt.
- **איך הפרויקט נראה:** תיקיית config, src (api, services, domain, repositories, schemas, infrastructure, utils, pipeline), scripts לאינדקס ו-CLI, ו-chroma_db – עם pipeline ברור לאינדקס ול-query ומודלים ב-Pydantic לכל הממשקים.

---

## 5. הרצה

**התקנת uv (אם עדיין לא):**  
[https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) או `pip install uv`

```bash
# מהתיקייה Law (שורש הפרויקט)
cd Law

# יצירת סביבה וירטואלית + התקנת תלויות (משורש הפרויקט)
uv venv
uv sync

# (Windows) הפעלת הסביבה
.venv\Scripts\activate

# (Linux/macOS) הפעלת הסביבה
# source .venv/bin/activate

# העתקת .env (לאחר עריכה + COURTLISTENER_API_TOKEN)
cp .env.example .env   # Linux/macOS
# copy .env.example .env   # Windows CMD

# בניית אינדקס מ-CourtListener (ראה למטה: CourtListener – אינדוקס והמשך)

# חיפוש מ-CLI
uv run python scripts/query_cli.py "breach of contract Section 12"

# התאמת תיק מקובץ טקסט לתיקים דומים באינדקס
uv run python scripts/match_case_file.py examples/sample_case.txt
uv run python scripts/match_case_file.py examples/sample_case.txt --top-k 5

# CourtListener: בדיקה (5 תיקים) או הורדת הכל (ללא --max-docs); --resume להמשך
uv run python scripts/rebuild_first_courtlistener_cases.py --max-docs 5
uv run python scripts/rebuild_first_courtlistener_cases.py
uv run python scripts/rebuild_first_courtlistener_cases.py --resume
uv run python scripts/match_case_file.py examples/sample_case.txt --top-k 3

# הפעלת API (משורש – הקוד ב-backend/)
cd backend && uv run uvicorn src.app:app --reload --port 8000
# POST http://localhost:8000/search – תיקים דומים (retrieval)

# או עם Docker
docker compose up --build
# API: http://localhost:8000 – לפתוח frontend/index.html בדפדפן
```

**חלופה עם pip:**  
משורש: `pip install -r requirements.txt`. הרץ סקריפטים: `python scripts/query_cli.py "..."`.

---

### Troubleshooting (Windows) – שגיאת cache של Hugging Face

אם מופיעה השגיאה:
```text
ERROR: Invalid disk cache: your machine does not support long paths.
```
זה קורה ב-Windows כשנתיב ה-cache של Hugging Face ארוך מדי (מגבלת אורך נתיב). **פתרון:** להגדיר משתנה סביבה `HF_HOME` לנתיב **קצר**, למשל:
- `C:\hf_cache`
- או `c:\Users\user\Desktop\Law\.cache\hf`

ב-PowerShell לפני ההרצה:
```powershell
$env:HF_HOME = "C:\hf_cache"
uv run python scripts/rebuild_first_courtlistener_cases.py
```
או להגדיר `HF_HOME` במערכת (מערכת → משתני סביבה) או ב-.env אם הכלי שלך מעביר משתנים אלה לתהליך.

---

### CourtListener – אינדוקס והמשך (checkpoint)

הוסף ב-`.env` את הטוקן מ-[Developer Tools](https://www.courtlistener.com/help/api/rest/) (Your API Token):

```bash
COURTLISTENER_API_TOKEN=your_token_here
COURTLISTENER_BASE_URL=https://www.courtlistener.com/api/rest/v4
```

- **חפיפה בין chunks:** נקבעת ב-config (`CHUNK_OVERLAP`, ברירת מחדל 150 תווים) – מילים משותפות בין chunks סמוכים באותו תיק.
- **מטא-דאטה שנשמרת:** חובה: `id`, טקסט מלא, `title` (שם תיק – נמשך מ-cluster אם צריך). מומלץ: `citation`, `court`, `date_filed`, `docket_number`, `disposition` (לתמיכה ב-Decision Support). התיק המלא נשמר גם ב-`exports/.../texts`.
- **הרצה בלי הגבלה:** ברירת מחדל – מוריד את כל התיקים. אפשר לעצור (Ctrl+C) בכל רגע; הנקודה נשמרת ב-checkpoint.
- **המשך מהנקודה:** אחרי עצירה, הרץ עם `--resume` כדי להמשיך מהעמוד הבא (בלי למחוק את האינדקס).

```bash
# בדיקה מהירה (5 תיקים)
uv run python scripts/rebuild_first_courtlistener_cases.py --max-docs 5

# הורדת הכל (ללא הגבלה); לעצור ידנית – checkpoint נשמר אוטומטית
uv run python scripts/rebuild_first_courtlistener_cases.py

# המשך מאותה נקודה אחרי עצירה
uv run python scripts/rebuild_first_courtlistener_cases.py --resume

uv run python scripts/match_case_file.py examples/sample_case.txt --top-k 3
```

הטקסטים והווקטורים יישמרו ב-`exports/courtlistener_first_cases/texts` ו-`exports/courtlistener_first_cases/vectors`. קובץ ה-checkpoint: `exports/courtlistener_first_cases/courtlistener_checkpoint.json`.

### 5 התיקים הכי דומים (CLI)

`query_cli` (שאילתה) או `match_case_file` (קובץ תיק) – מחזירים 5 תיקים דומים (ברירת מחדל `TOP_K=5`).

### Decision Support – סטטיסטיקה על תיקים דומים

**מתוך 5 תיקים דומים** (ברירת מחדל): התובע ניצח / הנתבע ניצח, והסתברות הצלחה לתובע. דורש `disposition` באינדקס.

```bash
uv run python scripts/decision_support.py "בעיה משפטית"
uv run python scripts/decision_support.py examples/sample_case.txt
```

### RAG – ניתוח תיק חדש מול תיקים דומים (LLM)

מרכיב **context** מחמשת התיקים הכי דומים (טקסט מלא מ-`exports/.../texts` או snippets), מוסיף את **התיק החדש** שהמשתמש הכניס, ושולח ל-**Groq API** (LLM בענן) לניתוח מובנה: Similarity Analysis, Legal Pattern, Common Outcome.

```bash
uv run python scripts/rag_analysis.py "טקסט התיק החדש"
uv run python scripts/rag_analysis.py examples/sample_case.txt
```

הגדרות ב-`.env`: `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL`, `EXPORTS_TEXTS_DIR`.
