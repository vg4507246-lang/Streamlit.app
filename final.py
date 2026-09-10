import streamlit as st
import sqlite3
import pandas as pd
import asyncio
import nest_asyncio
nest_asyncio.apply()
import os
import sys
import json
import io
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))

PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="Job Hunt Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.path.abspath("./job_hunt.db")

# ── DB helpers ────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def run_query(sql, params=()):
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)

def run_update(sql, params=()):
    with get_conn() as conn:
        conn.execute(sql, params)
        conn.commit()

# ── Profile helpers ───────────────────────────────────
def load_profile():
    p = os.path.join(PROJECT_ROOT, "user_profile.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

def save_profile(profile: dict):
    p = os.path.join(PROJECT_ROOT, "user_profile.json")
    _PASSWORD_KEYS = {"linkedin_password", "naukri_password", "internshala_password", "gmail_password"}
    safe = {k: v for k, v in profile.items() if k not in _PASSWORD_KEYS}
    safe["has_password"] = bool(profile.get("linkedin_password") or profile.get("gmail_password"))
    with open(p, "w") as f:
        json.dump(safe, f, indent=2)

def update_env(key: str, value: str):
    env_path = os.path.join(PROJECT_ROOT, ".env")
    lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        st.error(f"Could not extract PDF text: {e}")
        return ""

def generate_pdf_bytes(resume_text: str, cover_letter: str = "",
                        job_title: str = "", company: str = "") -> bytes:
    """Generate a PDF from text using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_LEFT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        normal = ParagraphStyle("normal", parent=styles["Normal"],
                                 fontSize=10, leading=14, spaceAfter=6)
        heading = ParagraphStyle("heading", parent=styles["Heading2"],
                                  fontSize=12, leading=16, spaceBefore=10, spaceAfter=4)
        story = []

        if job_title and company:
            story.append(Paragraph(f"Resume — {job_title} @ {company}", styles["Title"]))
            story.append(Spacer(1, 0.3*cm))

        for line in resume_text.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.2*cm))
            elif line.isupper() and len(line) < 40:
                story.append(Paragraph(line, heading))
            else:
                story.append(Paragraph(line.replace("&","&amp;").replace("<","&lt;"), normal))

        if cover_letter:
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph("Cover Letter", styles["Heading1"]))
            story.append(Spacer(1, 0.3*cm))
            for line in cover_letter.split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.2*cm))
                else:
                    story.append(Paragraph(line.replace("&","&amp;").replace("<","&lt;"), normal))

        doc.build(story)
        return buf.getvalue()
    except ImportError:
        st.error("reportlab not installed — run: pip install reportlab")
        return b""

def generate_screening_answers(jd_text: str, questions: list[str],
                                resume_text: str, model: str = "llama3.1",
                                provider: str = "ollama") -> dict[str, str]:
    try:
        from app.llm import chat
        answers = {}
        for q in questions:
            prompt = f"""You are helping a job applicant answer a screening question.
Resume:
{resume_text[:2000]}

Job description excerpt:
{jd_text[:1000]}

Question: {q}

Write a concise, professional answer (2-4 sentences max). Be specific and reference the resume."""
            answers[q] = chat(prompt, model=model, provider=provider)
        return answers
    except Exception as e:
        return {q: f"[Error generating answer: {e}]" for q in questions}
    
# ✅ Fixed version
def extract_screening_questions(jd_text: str, model: str = "llama3.1",
                                 provider: str = "ollama") -> list[str]:
    try:
        from app.llm import chat
        prompt = f"""Extract any screening questions from this job description.
Return ONLY a JSON array of question strings. If no questions found, return [].
Example: ["Do you have 3+ years of Python experience?", "Are you willing to relocate?"]

Job description:
{jd_text[:3000]}"""
        text = chat(prompt, model=model, provider=provider)
        import re
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except Exception:
        return []


# ── Session state ─────────────────────────────────────
for key, default in [
    ("profile",              load_profile()),
    ("linkedin_password",    ""),
    ("naukri_password",      ""),
    ("internshala_password", ""),
    ("logged_in",            False),
    ("ats_result",           None),
    ("ats_job_id",           None),
    ("rewritten_resume",     None),
    ("edited_resume",        None),
    ("edited_cover",         None),
    ("screening_qa",         {}),
    ("apply_job_id",         None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

profile       = st.session_state.profile
is_setup_done = bool(profile.get("gmail_email") and
                     profile.get("has_password") and
                     profile.get("resume_text"))


# ══════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════
if not st.session_state.logged_in:
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown("## 🤖 Job Hunt Agent")
        st.markdown("#### Sign in with your Gmail")
        st.divider()
        with st.form("login_form"):
            email    = st.text_input("Gmail address", placeholder="your@gmail.com")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if submit:
            if not email or not password:
                st.error("Please enter both email and password")
            else:
                st.session_state.profile["gmail_email"]  = email
                st.session_state.profile["gmail_password"] = password
                st.session_state.profile["has_password"] = True
                # FIX #4: Also store in a dedicated session key so autofill
                # can always retrieve it (profile dict may be reloaded from disk
                # which strips the password for security).
                st.session_state["gmail_password_session"] = password
                st.session_state.logged_in = True
                save_profile(st.session_state.profile)
                st.rerun()
        st.divider()
        st.caption("Credentials saved only to your local profile file.")
    st.stop()


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.title("🤖 Job Hunt Agent")
    st.success(f"👤 {profile.get('full_name', profile.get('gmail_email', 'User'))}")
    st.caption(profile.get("gmail_email", ""))
    st.divider()

    pages = [
        "👤 Profile",
        "💼 Jobs",
        "📄 Resumes",
        "📬 Apply",
        "📨 Applications",
        "📊 Feedback",
        "🚀 Run Agent",
        "⚙️ Settings",
    ]
    page = st.radio("Navigate", pages, label_visibility="collapsed")
    st.divider()

    try:
        total_jobs = run_query("SELECT COUNT(*) as n FROM jobs").iloc[0]["n"]
        total_apps = run_query("SELECT COUNT(*) as n FROM applications").iloc[0]["n"]
        st.metric("Total jobs",         total_jobs)
        st.metric("Total applications", total_apps)
    except Exception:
        st.warning("DB not ready")

    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in","profile","linkedin_password","ats_result",
                  "rewritten_resume","edited_resume","edited_cover","screening_qa"]:
            st.session_state[k] = {} if k == "profile" else None if "result" in k or "resume" in k or "cover" in k else {} if k == "screening_qa" else False if k == "logged_in" else ""
        st.rerun()


# ══════════════════════════════════════════════════════
# PAGE: PROFILE
# ══════════════════════════════════════════════════════
if page == "👤 Profile":
    st.title("Profile")
    st.caption("Your personal details and job hunt overview")

    st.subheader("Personal details")
    col1, col2, col3 = st.columns(3)
    with col1:
        full_name   = st.text_input("Full name",         value=profile.get("full_name",   ""))
    with col2:
        location    = st.text_input("Preferred location",value=profile.get("location",    ""))
    with col3:
        target_role = st.text_input("Target role",       value=profile.get("target_role", ""),
                                    help="e.g. ML Engineer, Data Scientist")

    if st.button("💾 Save details", type="primary"):
        updated = {**profile, "full_name": full_name, "location": location, "target_role": target_role}
        save_profile(updated)
        st.session_state.profile = updated
        st.success("✅ Details saved!")
        st.rerun()

    st.divider()
    if profile.get("resume_text"):
        st.success("✅ Resume on file — go to 📄 Resumes to update or check ATS score")
    else:
        st.warning("⚠️ No resume uploaded yet — go to 📄 Resumes to upload your PDF")

    st.divider()
    st.subheader("Dashboard")

    try:
        stats = run_query("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status='shortlisted' THEN 1 ELSE 0 END) as shortlisted,
                   SUM(CASE WHEN status='applied'     THEN 1 ELSE 0 END) as applied,
                   SUM(CASE WHEN status='interview'   THEN 1 ELSE 0 END) as interviews,
                   ROUND(AVG(match_score), 1) as avg_score
            FROM jobs
        """).iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total jobs",  int(stats["total"]       or 0))
        c2.metric("Shortlisted", int(stats["shortlisted"] or 0))
        c3.metric("Applied",     int(stats["applied"]     or 0))
        c4.metric("Interviews",  int(stats["interviews"]  or 0))
        c5.metric("Avg score",   f"{stats['avg_score'] or 0}%")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Score distribution")
            scores = run_query("SELECT match_score FROM jobs WHERE match_score > 0")
            if not scores.empty:
                st.bar_chart(scores["match_score"].value_counts(bins=10).sort_index().rename("count"),
                             use_container_width=True)
            else:
                st.info("No scored jobs yet")
        with col2:
            st.subheader("Application funnel")
            funnel = run_query("SELECT status, COUNT(*) as count FROM applications GROUP BY status ORDER BY count DESC")
            if not funnel.empty:
                st.bar_chart(funnel.set_index("status"), use_container_width=True)
            else:
                st.info("No applications yet")

        recent = run_query("SELECT title, company, location, match_score, status, source, scraped_at FROM jobs ORDER BY scraped_at DESC LIMIT 10")
        if not recent.empty:
            st.subheader("Recent jobs")
            st.dataframe(recent, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"DB error: {e}")


# ══════════════════════════════════════════════════════
# PAGE: JOBS
# ══════════════════════════════════════════════════════
elif page == "💼 Jobs":
    st.title("Jobs")
 
    st.subheader("🔎 Quick search")
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    with col1:
        search_query = st.text_input("Job title", value=profile.get("target_role",""),
                                     placeholder="e.g. ML Engineer", label_visibility="collapsed")
    with col2:
        search_location = st.text_input("Location", value=profile.get("location","Hyderabad, India"),
                                        label_visibility="collapsed")
    with col3:
        search_source = st.selectbox(
            "Source",
            ["All", "LinkedIn", "Indeed", "Naukri", "Internshala"],
            label_visibility="collapsed",
        )
    with col4:
        run_search = st.button("🔍 Search", use_container_width=True, type="primary")

    if run_search and search_query:
        _src_map = {
            "All":         ["linkedin", "indeed", "naukri", "internshala"],
            "LinkedIn":    ["linkedin"],
            "Indeed":      ["indeed"],
            "Naukri":      ["naukri"],
            "Internshala": ["internshala"],
        }
        selected_sources = _src_map.get(search_source, ["indeed"])

        with st.status(f"Searching '{search_query}' on {search_source}...", expanded=True) as ss:
            try:
                os.environ["LINKEDIN_EMAIL"]    = profile.get("linkedin_email", "")
                os.environ["LINKEDIN_PASSWORD"] = st.session_state.linkedin_password or ""
                from app.tools.search_jobs import run_search_jobs

                # FIX 1: Safe int conversion — blank or non-numeric max_jobs won't crash
                try:
                    max_jobs = int(profile.get("max_jobs", 10) or 10)
                except (ValueError, TypeError):
                    max_jobs = 10

                # FIX 2: Run in a separate thread with its own event loop to avoid
                # Streamlit's event loop conflicts
                import concurrent.futures
                def _run_search():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(run_search_jobs(
                            query=search_query,
                            location=search_location,
                            max_per_source=max_jobs,
                            sources=selected_sources,
                        ))
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(_run_search).result(timeout=120)

                # FIX 3: Handle "error" key and surface per-scraper failures
                if result.get("error"):
                    ss.update(label="Search failed", state="error")
                    st.error(f"Search error: {result['error']}")
                elif result.get("status") == "ok":
                    ss.update(label="✅ Search complete!", state="complete")
                    saved = result.get("saved_new", 0)
                    if saved > 0:
                        st.success(f"✅ {saved} new jobs saved")
                    else:
                        st.warning("No new jobs — all already in DB")
                    # Show any partial scraper failures even on overall success
                    for src, err in result.get("scraper_errors", {}).items():
                        st.warning(f"⚠️ {src.capitalize()} scraper failed: {err}")
                elif result.get("status") == "no_results":
                    ss.update(label="No jobs found", state="error")
                    st.warning("No jobs found — try a broader title or different location")
                    # Show exactly which scraper failed and why
                    for src, err in result.get("scraper_errors", {}).items():
                        st.error(f"❌ {src.capitalize()} error: {err}")
                else:
                    ss.update(label="Search failed", state="error")
                    st.error(f"Unexpected response: {result}")
            except ImportError as e:
                ss.update(label="Import error", state="error")
                st.error(f"Could not import search tool: {e}")
            except Exception as e:
                ss.update(label="Error", state="error")
                st.error(f"Search error: {e}")
        st.rerun()
    elif run_search and not search_query:
        st.warning("Enter a job title to search")
 
    st.divider()
 
    try:
        total_in_db = run_query("SELECT COUNT(*) as n FROM jobs").iloc[0]["n"]
        if total_in_db == 0:
            st.info("No jobs yet — use the search bar above.")
        else:
            # ── Filters ──────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                sf = st.selectbox("Status", ["All","new","shortlisted","applied","rejected","interview","offer"])
            with col2:
                mf = st.slider("Min score", 0, 100, 0)
 
            where, params = ["1=1"], []
            if mf > 0:      where.append("match_score >= ?"); params.append(mf)
            if sf != "All": where.append("status = ?");      params.append(sf)
 
            base_sql = f"""
                SELECT id, title, company, location, salary_raw,
                       match_score, status, source, url, scraped_at
                FROM jobs WHERE {' AND '.join(where)}
                ORDER BY match_score DESC NULLS LAST
            """
            all_jobs = run_query(base_sql, params)
 
            def sc_col(s):
                if s is None: return "—"
                return f"🟢 {s:.0f}" if s >= 75 else f"🟡 {s:.0f}" if s >= 50 else f"🔴 {s:.0f}"
 
            display_cols = ["id","score","title","company","location","salary_raw","status","scraped_at"]
 
            # ── Four-column job lists by platform ────────────────
            li_jobs     = all_jobs[all_jobs["source"] == "linkedin"].copy()
            indeed_jobs = all_jobs[all_jobs["source"] == "indeed"].copy()
            naukri_jobs = all_jobs[all_jobs["source"] == "naukri"].copy()
            is_jobs     = all_jobs[all_jobs["source"] == "internshala"].copy()

            col_li, col_in = st.columns(2)
            col_nk, col_is = st.columns(2)

            with col_li:
                st.subheader(f"🔵 LinkedIn  ({len(li_jobs)})")
                if not li_jobs.empty:
                    li_jobs["score"] = li_jobs["match_score"].apply(sc_col)
                    st.dataframe(li_jobs[display_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("No LinkedIn jobs match filters.")

            with col_in:
                st.subheader(f"🟦 Indeed  ({len(indeed_jobs)})")
                if not indeed_jobs.empty:
                    indeed_jobs["score"] = indeed_jobs["match_score"].apply(sc_col)
                    st.dataframe(indeed_jobs[display_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("No Indeed jobs match filters.")

            with col_nk:
                st.subheader(f"🟧 Naukri  ({len(naukri_jobs)})")
                if not naukri_jobs.empty:
                    naukri_jobs["score"] = naukri_jobs["match_score"].apply(sc_col)
                    st.dataframe(naukri_jobs[display_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("No Naukri jobs match filters.")

            with col_is:
                st.subheader(f"🟢 Internshala  ({len(is_jobs)})")
                if not is_jobs.empty:
                    is_jobs["score"] = is_jobs["match_score"].apply(sc_col)
                    st.dataframe(is_jobs[display_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("No Internshala jobs match filters.")
            
            # ── Scraped jobs summary ──────────────────────────
            st.divider()
            sc = run_query("SELECT status, COUNT(*) as count FROM jobs GROUP BY status")
            sd = dict(zip(sc["status"], sc["count"]))
            last = run_query("SELECT MAX(scraped_at) as t FROM jobs").iloc[0]["t"]
            st.subheader("📊 Scraped Jobs")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total",       total_in_db)
            c2.metric("New",         sd.get("new", 0))
            c3.metric("Shortlisted", sd.get("shortlisted", 0))
            c4.metric("Applied",     sd.get("applied", 0))
            if last: st.caption(f"Last updated: {last[:19]}")
 
    except Exception as e:
        st.error(f"DB error: {e}")
# ══════════════════════════════════════════════════════
# PAGE: RESUMES — upload + ATS + rewrite + edit + download
# ══════════════════════════════════════════════════════

elif page == "📄 Resumes":
    st.title("📄 Resumes")
    st.caption("Upload, version, and select your resume for ATS optimization and tailoring.")

    # ── 1. Upload Section ─────────────────────────────
    st.subheader("1. Upload New Resume")
    uploaded_pdf = st.file_uploader("Choose a PDF file to add a new version", type=["pdf"])

    extracted_text = None
    if uploaded_pdf:
        with st.spinner("Extracting text..."):
            # Seek to 0 to ensure we read from the start if the buffer was moved
            pdf_content = uploaded_pdf.read()
            extracted_text = extract_text_from_pdf(pdf_content)
        
        if extracted_text:
            st.success(f"✅ Extracted {len(extracted_text)} characters")
            with st.expander("Preview extracted text"):
                st.text_area("Extracted Content", extracted_text, height=150, disabled=True)
        else:
            st.error("Could not extract text. Ensure the PDF is not a scanned image.")

    if st.button("💾 Save as New Version", type="primary", use_container_width=True):
        if not extracted_text:
            st.error("Nothing to save. Please upload a valid PDF first.")
        else:
            try:
                # Calculate next version for a base resume
                res = run_query("SELECT MAX(version) as v FROM resumes WHERE is_base=1")
                next_v = int(res.iloc[0]["v"] or 0) + 1
                
                run_update(
                    "INSERT INTO resumes (is_base, version, target_role, resume_text, created_at) VALUES (1,?,?,?,?)",
                    (next_v, profile.get("target_role", "ML Engineer"), extracted_text, datetime.utcnow().isoformat())
                )
                
                # Update session profile to the latest text
                upd = {**profile, "resume_text": extracted_text}
                save_profile(upd)
                st.session_state.profile = upd
                
                st.success(f"✅ Version {next_v} saved to database!")
                st.rerun()  # Refresh to update the selection dropdown below
            except Exception as e:
                st.error(f"Database save failed: {e}")

    st.divider()

    # ── 2. Selection Section (The Source of Truth) ──────
    st.subheader("2. Select Working Resume")
    
    # Fetch all resumes (base and tailored) to let the user choose
    resumes_df = run_query("""
        SELECT id, version, target_role, is_base, created_at 
        FROM resumes 
        ORDER BY created_at DESC
    """)

    if resumes_df.empty:
        st.warning("⚠️ No resumes found in database. Please upload one above.")
        active_resume_text = None
    else:
        # Create descriptive labels for the dropdown
        resume_options = {}
        for _, r in resumes_df.iterrows():
            r_type = "Base" if r['is_base'] else "Tailored"
            label = f"ID {r['id']}: v{r['version']} - {r['target_role']} ({r_type}) - {r['created_at'][:10]}"
            resume_options[label] = r['id']
        
        selected_label = st.selectbox(
            "Select the resume version to use for ATS and Tailoring:",
            options=list(resume_options.keys()),
            help="The selected resume text will be passed to the AI tools below."
        )
        
        selected_id = resume_options[selected_label]
        
        # Load the specific text for this selection
        active_data = run_query("SELECT resume_text FROM resumes WHERE id=?", (selected_id,))
        active_resume_text = active_data.iloc[0]["resume_text"]
        
        # Store in session state for cross-tool consistency
        st.session_state["active_resume_id"] = selected_id
        st.session_state["active_resume_text"] = active_resume_text

        st.info(f"🎯 **Currently Active:** {selected_label}")
        with st.expander("View Selected Resume Content"):
            st.text_area("Content", active_resume_text, height=200, disabled=True)

    st.divider()   

   
    # ── 3. ATS Score Checker ──────────────────────────
    st.subheader("3. ATS Score Checker")
    
    jobs_df = run_query("SELECT id, title, company, description FROM jobs ORDER BY scraped_at DESC LIMIT 50")
    job_opts = {}
    
    if not active_resume_text:
        st.info("Upload or select a resume above to enable auditing.")
    elif jobs_df.empty:
        st.info("Search for jobs first to enable market analysis.")
    else:
        job_opts = {f"ID {r['id']} — {r['title']} @ {r['company']}": r['id'] for _, r in jobs_df.iterrows()}
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_job_label = st.selectbox("Select Specific Job for Audit", list(job_opts.keys()), key="ats_job_sel")
            target_job_id = job_opts[target_job_label]
        
        # TWO BUTTONS: One for specific, one for general
        with col2:
            st.write("") # Spacer
            spec_btn = st.button("🎯 Specific Audit", use_container_width=True, type="primary")
            gen_btn = st.button("📊 Market Audit", use_container_width=True)

        # LOGIC FOR SPECIFIC AUDIT
        if spec_btn:
            with st.spinner("Analyzing specific job match..."):
                job_desc = jobs_df[jobs_df['id'] == target_job_id].iloc[0]["description"]
                from app.services.ats_scorer import score_ats
                st.session_state.ats_result = score_ats(resume_text=active_resume_text, jd_text=job_desc)
                st.session_state.ats_mode = "Specific"

        # LOGIC FOR GENERAL MARKET AUDIT
        if gen_btn:
            with st.spinner("Analyzing industry standards..."):
                # Filter jobs that match your target role to create a 'Standard' JD
                role_query = f"%{profile.get('target_role', 'Engineer')}%"
                market_data = run_query("SELECT description FROM jobs WHERE title LIKE ? LIMIT 10", (role_query,))
                
                if market_data.empty:
                    st.error("Not enough similar jobs found. Scrape more jobs for your role first.")
                else:
                    combined_jd = "\n---\n".join(market_data['description'].tolist())
                    from app.services.ats_scorer import score_ats
                    st.session_state.ats_result = score_ats(resume_text=active_resume_text, jd_text=combined_jd)
                    st.session_state.ats_mode = "General Market"

        # DISPLAY RESULTS
        if st.session_state.get("ats_result"):
            res = st.session_state.ats_result
            mode = st.session_state.get("ats_mode", "Audit")
            st.markdown(f"### {mode} Result: {res['ats_score']}%")
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("✅ **Matches Found:**")
                st.write(", ".join(res.get("matched_keywords", [])))
            with c2:
                st.write("❌ **Industry Gaps:**")
                st.write(", ".join(res.get("missing_keywords", [])))
            
            st.info(f"💡 **Strategy:** {res.get('feedback', '')}")

    st.divider()

    # ── 4. Rewrite & Tailor ───────────────────────────
    st.subheader("4. Rewrite for Specific Job")
    if not active_resume_text:
        st.info("Upload or select a resume to enable tailoring.")
    elif not job_opts:
        st.info("No jobs available for tailoring.")
    else:
        # job_opts is now guaranteed to exist because it was defined above
        rw_job_label = st.selectbox("Select Target Job", list(job_opts.keys()), key="rw_job_sel")
        rw_job_id = job_opts[rw_job_label]
        
        if st.button("✍️ Generate Tailored Version", type="primary", use_container_width=True):
            with st.status("Tailoring resume...", expanded=True) as status:
                try:
                    from app.tools.rewrite_resume import run_rewrite_resume
                    # Pass the source_resume_id to the rewrite tool
                    
                    rw_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(rw_loop)
                    rw_res = rw_loop.run_until_complete(run_rewrite_resume(job_id=rw_job_id, base_resume_text=st.session_state.get("active_resume_text", "")))
                    
                    if "error" in rw_res:
                        status.update(label="❌ Failed", state="error")
                        st.error(rw_res["error"])
                    else:
                        st.session_state.rewritten_resume = rw_res
                        # Pre-load editable fields from result
                        st.session_state.edited_resume = rw_res.get("resume_text", "")
                        st.session_state.edited_cover  = rw_res.get("cover_letter_preview", "")
                        status.update(label="✅ Success!", state="complete")
                except Exception as e:
                    st.error(f"Tailoring error: {e}")

    # ── Edit + Download section ────────────────
    if st.session_state.rewritten_resume:
        rw = st.session_state.rewritten_resume
        st.divider()
        st.subheader("📝 Edit and download your tailored resume")

        col1,col2,col3 = st.columns(3)
        col1.metric("Job",     rw.get("title","—"))
        col2.metric("Company", rw.get("company","—"))
        col3.metric("Version", f"v{rw.get('version',1)}")

        if rw.get("keywords_added"):
            kws = rw["keywords_added"]
            st.info(f"💡 Keywords added: {', '.join(kws) if isinstance(kws,list) else kws}")
        if rw.get("changes_summary"):
            st.write(f"**Changes:** {rw['changes_summary']}")

        # Editable resume text area
        st.write("**Resume** — edit freely, then generate PDF")
        edited_resume = st.text_area(
            "Tailored resume",
            value=st.session_state.edited_resume or "",
            height=400,
            key="resume_edit_area",
            label_visibility="collapsed",
        )
        st.session_state.edited_resume = edited_resume

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("📄 Generate Resume PDF", use_container_width=True, type="primary"):
                with st.spinner("Generating resume PDF..."):
                    pdf_bytes = generate_pdf_bytes(
                        resume_text=edited_resume,
                        job_title=rw.get("title",""),
                        company=rw.get("company",""),
                    )
                if pdf_bytes:
                    st.session_state["resume_pdf_ready"] = pdf_bytes
        with col_r2:
            if st.session_state.get("resume_pdf_ready"):
                fname_r = f"resume_{rw.get('title','job').replace(' ','_')}.pdf"
                st.download_button(
                    "⬇️ Download Resume PDF",
                    data=st.session_state["resume_pdf_ready"],
                    file_name=fname_r,
                    mime="application/pdf",
                    use_container_width=True,
                )
 
        st.divider()        

        # Editable cover letter
        st.write("**Cover letter** — edit freely")
        edited_cover = st.text_area(
            "Cover letter",
            value=st.session_state.edited_cover or "",
            height=250,
            key="cover_edit_area",
            label_visibility="collapsed",
        )
        st.session_state.edited_cover = edited_cover

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("📄 Generate Cover Letter PDF", use_container_width=True, type="primary"):
                with st.spinner("Generating cover letter PDF..."):
                    pdf_bytes_cl = generate_pdf_bytes(
                        resume_text=edited_cover,
                        job_title=rw.get("title",""),
                        company=rw.get("company",""),
                    )
                if pdf_bytes_cl:
                    st.session_state["cover_pdf_ready"] = pdf_bytes_cl
        with col_c2:
            if st.session_state.get("cover_pdf_ready"):
                fname_c = f"cover_letter_{rw.get('title','job').replace(' ','_')}.pdf"
                st.download_button(
                    "⬇️ Download Cover Letter PDF",
                    data=st.session_state["cover_pdf_ready"],
                    file_name=fname_c,
                    mime="application/pdf",
                    use_container_width=True,
                )
 
        st.divider()

        # Save edits to DB
        if st.button("💾 Save edits to DB", use_container_width=True):
            try:
                run_update(
                    "UPDATE resumes SET resume_text=?, cover_letter=? WHERE id=?",
                    (edited_resume, edited_cover, rw.get("resume_id")),
                )
                st.success("✅ Edits saved to DB")
            except Exception as e:
                st.error(f"Save failed: {e}")

        # Quick-jump to Apply page
        st.divider()
        st.info("✅ Resume ready — go to **📬 Apply** to select this job and apply")        

        
    # ── 5. Manage All Resumes ─────────────────────────
    st.divider()
    st.subheader("5. Database Management")
    if not resumes_df.empty:
        st.dataframe(resumes_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Delete Tailored Resumes", help="Clears only generated versions, keeps base resumes"):
            run_update("DELETE FROM resumes WHERE is_base=0")
            st.success("Cleaned tailored resumes."); st.rerun()

# ══════════════════════════════════════════════════════
# PAGE: APPLY — select jobs, screening Q&A, apply
# ══════════════════════════════════════════════════════
elif page == "📬 Apply":
    st.title("Apply")
    st.caption("Select jobs to apply to, review AI-generated screening answers, edit and submit")

    # ── Manual selection ──────────────────────────────
    st.subheader("Select jobs to apply to")
    st.caption("Tick the jobs you want to apply for. Each one opens a review panel.")

    try:
        jobs_apply = run_query("""
            SELECT id, title, company, location, match_score, status, url, description, source
            FROM jobs
            WHERE status NOT IN ('applied','rejected','offer','skipped')
            ORDER BY match_score DESC NULLS LAST
        """)
    except Exception as e:
        st.error(f"DB error: {e}")
        jobs_apply = pd.DataFrame()

    if jobs_apply.empty:
        st.info("No jobs available to apply to. Search for jobs first in 💼 Jobs.")
    else:
        st.caption(f"{len(jobs_apply)} jobs available")

        # Checkbox per job with platform badge
        selected_ids = []
        for _, job in jobs_apply.iterrows():
            score_str = f"🟢 {job['match_score']:.0f}%" if job['match_score'] and job['match_score']>=75 \
                        else f"🟡 {job['match_score']:.0f}%" if job['match_score'] and job['match_score']>=50 \
                        else f"🔴 {job['match_score']:.0f}%" if job['match_score'] else "—"
            src = job.get("source","")
            platform_badge = (
                "🟦 Indeed"      if src == "indeed"      else
                "🔵 LinkedIn"    if src == "linkedin"    else
                "🟧 Naukri"      if src == "naukri"      else
                "🟢 Internshala" if src == "internshala" else
                f"🔘 {src}"
            )
            label = f"**{job['title']}** @ {job['company']}   {score_str}   {platform_badge}   {job['location'] or ''}"
            if st.checkbox(label, key=f"chk_{job['id']}"):
                selected_ids.append(int(job['id']))

        if not selected_ids:
            st.info("Tick one or more jobs above to review and apply")
        else:
            st.success(f"✅ {len(selected_ids)} job(s) selected")
            st.divider()

            # Review each selected job
            for job_id in selected_ids:
                job_row = jobs_apply[jobs_apply["id"] == job_id].iloc[0]
                job_source    = job_row.get("source","")
                is_indeed     = job_source == "indeed"
                is_linkedin   = job_source == "linkedin"
                is_naukri     = job_source == "naukri"
                is_internshala = job_source == "internshala"

                with st.expander(f"📋 Review: {job_row['title']} @ {job_row['company']}", expanded=True):

                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.write(f"**Company:** {job_row['company']}")
                        st.write(f"**Location:** {job_row['location'] or '—'}")
                        st.write(f"**Platform:** {'🟦 Indeed' if is_indeed else '🔵 LinkedIn' if is_linkedin else '🟧 Naukri' if is_naukri else '🟢 Internshala' if is_internshala else job_source}")
                        st.write(f"**Match score:** {job_row['match_score']:.0f}%" if job_row['match_score'] else "**Match score:** —")
                        if job_row.get("url"):
                            st.markdown(f"[View job posting ↗]({job_row['url']})")
                    with col2:
                        tr = run_query("SELECT id, version FROM resumes WHERE job_id=? AND is_base=0 ORDER BY version DESC LIMIT 1", (job_id,))
                        if not tr.empty:
                            st.success(f"✅ Tailored resume ready (v{tr.iloc[0]['version']})")
                        else:
                            st.warning("⚠️ No tailored resume — go to 📄 Resumes to rewrite first")

                    # ── Screening questions ────────────────────
                    st.write("---")
                    st.write("**📝 Screening questions**")
                    jd = job_row.get("description","") or ""
                    qa_key = f"qa_{job_id}"

                    col1, col2 = st.columns([1,1])
                    with col1:
                        if st.button("🤖 Extract & answer questions", key=f"extract_{job_id}"):
                            with st.spinner("Extracting questions and generating answers..."):
                                questions = extract_screening_questions(jd, model=profile.get("ollama_model","llama3.1"), provider=os.getenv("LLM_PROVIDER", "ollama"),)
                                if questions:
                                    answers = generate_screening_answers(
                                        jd_text=jd, questions=questions,
                                        resume_text=profile.get("resume_text",""),
                                        model=profile.get("ollama_model","llama3.1"),
                                        provider=os.getenv("LLM_PROVIDER", "ollama"),
                                    )
                                    st.session_state.screening_qa[qa_key] = answers
                                    st.success(f"✅ Generated answers for {len(questions)} questions")
                                else:
                                    st.info("No screening questions found in this JD")
                                    st.session_state.screening_qa[qa_key] = {}
                    with col2:
                        st.caption("JD available ✅" if jd else "No JD on file ⚠️")

                    # Editable Q&A
                    qa = st.session_state.screening_qa.get(qa_key, {})
                    edited_qa = {}
                    if qa:
                        st.write(f"**{len(qa)} question(s) — edit answers below:**")
                        for i, (question, answer) in enumerate(qa.items()):
                            st.write(f"**Q{i+1}: {question}**")
                            edited_qa[question] = st.text_area(
                                f"Answer {i+1}", value=answer, height=100,
                                key=f"ans_{job_id}_{i}", label_visibility="collapsed")
                        st.session_state.screening_qa[qa_key] = edited_qa
                    else:
                        st.caption("Click 'Extract & answer questions' to auto-fill screening answers")

                    # ── Single Apply button ────────────────────
                    st.write("---")
                    if st.button("🚀 Apply", key=f"apply_btn_{job_id}", type="primary", use_container_width=True):
                        with st.spinner(f"Applying to {job_row['title']} @ {job_row['company']}..."):
                            try:
                                # Load resume PDF
                                resume_pdf = None
                                r = run_query("SELECT resume_text, cover_letter FROM resumes WHERE job_id=? AND is_base=0 ORDER BY version DESC LIMIT 1", (job_id,))
                                if r.empty:
                                    r = run_query("SELECT resume_text, cover_letter FROM resumes WHERE is_base=1 ORDER BY created_at DESC LIMIT 1")
                                if not r.empty:
                                    resume_pdf = generate_pdf_bytes(
                                        resume_text=r.iloc[0]["resume_text"] or "",
                                        cover_letter=r.iloc[0].get("cover_letter") or "",
                                    )

                                if is_indeed:
                                    from app.scrapers.indeed_autofill import autofill_indeed_job, AutofillConfig
                                    gmail_password = (
                                        st.session_state.profile.get("gmail_password", "")
                                        or st.session_state.get("gmail_password_session", "")
                                    )
                                    cfg = AutofillConfig(
                                        email=profile.get("gmail_email",""),
                                        password=gmail_password,
                                        full_name=profile.get("full_name",""),
                                        phone=profile.get("phone",""),
                                        resume_pdf=resume_pdf,
                                        cover_letter="",
                                        screening_answers=edited_qa,
                                        dry_run=False,
                                    )
                                    res = autofill_indeed_job(job_url=job_row.get("url",""), config=cfg)
                                elif is_linkedin:
                                    from app.scrapers.linkedin_autofill import autofill_linkedin_job, LinkedInAutofillConfig
                                    linkedin_password = (
                                        st.session_state.get("linkedin_password", "")
                                        or st.session_state.profile.get("linkedin_password", "")
                                    )
                                    cfg = LinkedInAutofillConfig(
                                        email=profile.get("linkedin_email",""),
                                        password=linkedin_password,
                                        full_name=profile.get("full_name",""),
                                        phone=profile.get("phone",""),
                                        resume_pdf=resume_pdf,
                                        cover_letter="",
                                        screening_answers=edited_qa,
                                        dry_run=False,
                                    )
                                    res = autofill_linkedin_job(job_url=job_row.get("url",""), config=cfg)
                                elif is_naukri:
                                    from app.scrapers.naukri_autofill import autofill_naukri_job, NaukriAutofillConfig
                                    naukri_password = (
                                        st.session_state.get("naukri_password", "")
                                        or st.session_state.profile.get("naukri_password", "")
                                    )
                                    cfg = NaukriAutofillConfig(
                                        email=profile.get("naukri_email",""),
                                        password=naukri_password,
                                        full_name=profile.get("full_name",""),
                                        phone=profile.get("phone",""),
                                        resume_pdf=resume_pdf,
                                        cover_letter=edited_qa.get("cover_letter",""),
                                        screening_answers=edited_qa,
                                        dry_run=False,
                                    )
                                    res = autofill_naukri_job(job_url=job_row.get("url",""), config=cfg)
                                elif is_internshala:
                                    from app.scrapers.internshala_autofill import autofill_internshala_job, InternshalaAutofillConfig
                                    internshala_password = (
                                        st.session_state.get("internshala_password", "")
                                        or st.session_state.profile.get("internshala_password", "")
                                    )
                                    cfg = InternshalaAutofillConfig(
                                        email=profile.get("internshala_email",""),
                                        password=internshala_password,
                                        full_name=profile.get("full_name",""),
                                        phone=profile.get("phone",""),
                                        cover_letter=edited_qa.get("cover_letter", ""),
                                        screening_answers=edited_qa,
                                        dry_run=False,
                                    )
                                    res = autofill_internshala_job(job_url=job_row.get("url",""), config=cfg)
                                else:
                                    res = {"success": False, "status": "unsupported", "error": f"Platform '{job_source}' not supported for auto-fill yet."}

                                if res.get("success") or res.get("status") == "submitted":
                                    run_update("UPDATE jobs SET status='applied' WHERE id=?", (job_id,))
                                    # Log to applications table
                                    from app.tools.apply_job import run_apply_job
                                    a_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(a_loop)
                                    a_loop.run_until_complete(run_apply_job(
                                        job_id=job_id, apply_method="form",
                                        applied_to=job_row.get("url",""),
                                        screening_answers=edited_qa, dry_run=False,
                                    ))
                                    st.success(f"✅ Applied successfully to {job_row['title']} @ {job_row['company']}!")
                                    st.balloons()
                                else:
                                    st.error(f"❌ {res.get('error', 'Apply failed')}")
                            except Exception as e:
                                st.error(f"Apply error: {e}")



# ══════════════════════════════════════════════════════
# PAGE: APPLICATIONS
# ══════════════════════════════════════════════════════
elif page == "📨 Applications":
    st.title("Applications")

    # ── Helper: run autofill for one job (Indeed or LinkedIn) ──
    def _autofill_row(row) -> dict:
        resume_pdf = None
        try:
            r = run_query(
                "SELECT resume_text, cover_letter FROM resumes WHERE job_id=? AND is_base=0 ORDER BY version DESC LIMIT 1",
                (row["job_id"],)
            )
            if r.empty:
                r = run_query("SELECT resume_text, cover_letter FROM resumes WHERE is_base=1 ORDER BY created_at DESC LIMIT 1")
            if not r.empty:
                resume_pdf = generate_pdf_bytes(
                    resume_text=r.iloc[0]["resume_text"] or "",
                    cover_letter=r.iloc[0].get("cover_letter") or "",
                )
        except Exception as e:
            st.warning(f"Could not load resume PDF: {e}")

        screening_answers = {}
        try:
            sa_row = run_query("SELECT outcome_notes FROM applications WHERE id=?", (row["id"],))
            if not sa_row.empty and sa_row.iloc[0].get("outcome_notes"):
                screening_answers = json.loads(sa_row.iloc[0]["outcome_notes"] or "{}")
        except Exception:
            pass

        job_source = (row.get("source") or "").lower()
        job_url    = (row.get("applied_to") or "").lower()
        is_linkedin    = "linkedin.com"    in job_url or job_source == "linkedin"
        is_naukri      = "naukri.com"      in job_url or job_source == "naukri"
        is_internshala = "internshala.com" in job_url or job_source == "internshala"
        # Indeed is the fallback

        if is_linkedin:
            from app.scrapers.linkedin_autofill import autofill_linkedin_job, LinkedInAutofillConfig
            linkedin_password = (
                st.session_state.get("linkedin_password", "")
                or st.session_state.profile.get("linkedin_password", "")
            )
            cfg = LinkedInAutofillConfig(
                email=profile.get("linkedin_email",""),
                password=linkedin_password,
                full_name=profile.get("full_name",""),
                phone=profile.get("phone",""),
                resume_pdf=resume_pdf,
                cover_letter="",
                screening_answers=screening_answers,
                dry_run=profile.get("dry_run", True),
            )
            return autofill_linkedin_job(job_url=row["applied_to"] or "", config=cfg)
        elif is_naukri:
            from app.scrapers.naukri_autofill import autofill_naukri_job, NaukriAutofillConfig
            naukri_password = (
                st.session_state.get("naukri_password", "")
                or st.session_state.profile.get("naukri_password", "")
            )
            cfg = NaukriAutofillConfig(
                email=profile.get("naukri_email",""),
                password=naukri_password,
                full_name=profile.get("full_name",""),
                phone=profile.get("phone",""),
                resume_pdf=resume_pdf,
                cover_letter=screening_answers.get("cover_letter",""),
                screening_answers=screening_answers,
                dry_run=profile.get("dry_run", True),
            )
            return autofill_naukri_job(job_url=row["applied_to"] or "", config=cfg)
        elif is_internshala:
            from app.scrapers.internshala_autofill import autofill_internshala_job, InternshalaAutofillConfig
            internshala_password = (
                st.session_state.get("internshala_password", "")
                or st.session_state.profile.get("internshala_password", "")
            )
            cfg = InternshalaAutofillConfig(
                email=profile.get("internshala_email",""),
                password=internshala_password,
                full_name=profile.get("full_name",""),
                phone=profile.get("phone",""),
                cover_letter=screening_answers.get("cover_letter",""),
                screening_answers=screening_answers,
                dry_run=profile.get("dry_run", True),
            )
            return autofill_internshala_job(job_url=row["applied_to"] or "", config=cfg)
        else:
            from app.scrapers.indeed_autofill import autofill_indeed_job, AutofillConfig
            gmail_password = (
                st.session_state.profile.get("gmail_password", "")
                or st.session_state.get("gmail_password_session", "")
            )
            cfg = AutofillConfig(
                email=profile.get("gmail_email",""),
                password=gmail_password,
                full_name=profile.get("full_name",""),
                phone=profile.get("phone",""),
                resume_pdf=resume_pdf,
                cover_letter="",
                screening_answers=screening_answers,
                dry_run=profile.get("dry_run", True),
            )
            return autofill_indeed_job(job_url=row["applied_to"] or "", config=cfg)

    try:
        apps = run_query("""
            SELECT a.id, a.job_id, j.title, j.company, j.source,
                   a.apply_method, a.status,
                   a.applied_to, a.outcome_notes, a.applied_at
            FROM applications a
            LEFT JOIN jobs j ON j.id = a.job_id
            ORDER BY a.applied_at DESC
        """)
        if apps.empty:
            st.info("No applications yet. Use 📬 Apply to start applying.")
        else:
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total",          len(apps))
            c2.metric("Pending/Queued", len(apps[apps["status"]=="pending_approval"]))
            c3.metric("Sent",           len(apps[apps["status"]=="sent"]))
            c4.metric("Rejected",       len(apps[apps["status"]=="rejected"]))
            c5.metric("Interview",      len(apps[apps["status"]=="interview"]))
            st.dataframe(apps, use_container_width=True, hide_index=True)

            # ── Pending Applications — Open & Apply ───────────
            pending = apps[apps["status"] == "pending_approval"]
            if not pending.empty:
                st.divider()
                st.subheader("🔗 Pending Applications — Open & Apply")

                # ── ✅ Apply All button (Change 2) ────────────
                indeed_pending = pending[
                    pending["applied_to"].str.contains("indeed.com", na=False)
                ]
                linkedin_pending = pending[
                    pending["applied_to"].str.contains("linkedin.com", na=False)
                ]
                naukri_pending = pending[
                    pending["applied_to"].str.contains("naukri.com", na=False)
                ]
                internshala_pending = pending[
                    pending["applied_to"].str.contains("internshala.com", na=False)
                ]
                autofillable = pd.concat([
                    indeed_pending, linkedin_pending, naukri_pending, internshala_pending
                ]).drop_duplicates()
                non_autofillable = pending[~pending.index.isin(autofillable.index)]

                dry_run_flag = profile.get("dry_run", True)
                col_hdr1, col_hdr2 = st.columns([4, 1])
                with col_hdr1:
                    if dry_run_flag:
                        st.caption("🟡 Dry run is ON — Selenium will navigate but NOT submit")
                    else:
                        st.caption("🔴 Live mode — Selenium WILL submit real applications")
                with col_hdr2:
                    apply_all_btn = st.button(
                        "✅ Apply All",
                        use_container_width=True,
                        type="primary",
                        disabled=autofillable.empty,
                        help="Auto-fills and submits all pending Indeed + LinkedIn + Naukri + Internshala applications via Selenium",
                    )

                if apply_all_btn and not autofillable.empty:
                    progress = st.progress(0, text="Starting Selenium autofill…")
                    aa_results = {"submitted": 0, "dry_run": 0, "failed": 0}
                    for i, (_, row) in enumerate(autofillable.iterrows()):
                        progress.progress(
                            (i + 1) / len(autofillable),
                            text=f"Applying to {row['title']} @ {row['company']}…",
                        )
                        res = _autofill_row(row)
                        if res.get("status") == "submitted":
                            run_update("UPDATE applications SET status='sent' WHERE id=?", (row["id"],))
                            run_update("UPDATE jobs SET status='applied' WHERE id=?", (row["job_id"],))
                            aa_results["submitted"] += 1
                            st.success(f"✅ Submitted: {row['title']} @ {row['company']}")
                        elif res.get("status") == "dry_run":
                            aa_results["dry_run"] += 1
                            st.info(f"🟡 Dry-run OK: {row['title']} @ {row['company']}")
                        else:
                            aa_results["failed"] += 1
                            st.error(f"❌ Failed: {row['title']} @ {row['company']} — {res.get('error','unknown error')}")
                    progress.empty()
                    st.info(
                        f"Apply All complete — "
                        f"submitted: {aa_results['submitted']}, "
                        f"dry-run: {aa_results['dry_run']}, "
                        f"failed: {aa_results['failed']}"
                    )
                    if aa_results["submitted"] > 0:
                        st.balloons()
                    st.rerun()

                st.caption(
                    f"{len(pending)} pending job(s) — "
                    f"{len(autofillable)} auto-fillable (Indeed + LinkedIn), "
                    f"{len(non_autofillable)} other"
                )
                for _, row in pending.iterrows():
                    url = (row.get("applied_to") or "").lower()
                    is_autofillable = any(d in url for d in ("indeed.com", "linkedin.com", "naukri.com", "internshala.com"))
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    with col1:
                        if "indeed.com" in url:
                            src_badge = "🟦 Indeed"
                        elif "linkedin.com" in url:
                            src_badge = "🔵 LinkedIn"
                        elif "naukri.com" in url:
                            src_badge = "🟧 Naukri"
                        elif "internshala.com" in url:
                            src_badge = "🟢 Internshala"
                        else:
                            src_badge = "🔘 Other"
                        st.write(f"**{row['title']}** @ {row['company']}  {src_badge}")
                    with col2:
                        if row.get("applied_to"):
                            st.markdown(f"[🔗 Open job ↗]({row['applied_to']})")
                        else:
                            st.caption("No URL saved")
                    with col3:
                        btn_label = "🤖 Auto-fill" if is_autofillable else "✅ Mark sent"
                        btn_tip   = "Launch Selenium to fill & submit" if is_autofillable else "Mark as manually sent"
                        if st.button(btn_label, key=f"mark_sent_{row['id']}", help=btn_tip):
                            if is_autofillable:
                                with st.spinner(f"Selenium filling {row['title']}…"):
                                    res = _autofill_row(row)
                                if res.get("status") == "submitted":
                                    run_update("UPDATE applications SET status='sent' WHERE id=?", (row["id"],))
                                    run_update("UPDATE jobs SET status='applied' WHERE id=?", (row["job_id"],))
                                    st.success(f"✅ Submitted: {row['title']} @ {row['company']}")
                                    st.balloons()
                                elif res.get("status") == "dry_run":
                                    st.info("🟡 Dry-run complete — no real submission made")
                                else:
                                    st.error(f"❌ {res.get('error','Autofill failed')}")
                            else:
                                run_update("UPDATE applications SET status='sent' WHERE id=?", (row["id"],))
                                run_update("UPDATE jobs SET status='applied' WHERE id=?", (row["job_id"],))
                                st.success("Marked as sent!")
                            st.rerun()
                    with col4:
                        if is_autofillable and st.button("✅ Mark sent", key=f"manual_sent_{row['id']}",
                                                          help="Skip Selenium and mark as sent manually"):
                            run_update("UPDATE applications SET status='sent' WHERE id=?", (row["id"],))
                            run_update("UPDATE jobs SET status='applied' WHERE id=?", (row["job_id"],))
                            st.success("Marked as sent!"); st.rerun()

            st.divider()
            st.subheader("Log outcome")
            col1,col2,col3 = st.columns([1,2,1])
            with col1: app_id = st.number_input("Application ID", min_value=1, step=1)
            with col2: outcome = st.selectbox("Outcome", ["rejected","interview","offer","no_response"])
            with col3: outcome_notes = st.text_input("Notes (optional)")
            if st.button("Save outcome"):
                run_update("UPDATE applications SET status=?, outcome_notes=? WHERE id=?",
                           (outcome, outcome_notes or None, app_id))
                st.success(f"Application {app_id} → {outcome}"); st.rerun()
    except Exception as e:
        st.error(f"DB error: {e}")


# ══════════════════════════════════════════════════════
# PAGE: FEEDBACK
# ══════════════════════════════════════════════════════
elif page == "📊 Feedback":
    st.title("Weekly feedback")
    try:
        fb = run_query("""
            SELECT week_number, year, jobs_scraped, jobs_applied,
                   rejections, interviews, avg_match_score,
                   suggested_keywords, resume_improvements,
                   top_rejection_reasons, created_at
            FROM feedback ORDER BY year DESC, week_number DESC
        """)
        if fb.empty:
            st.info("No feedback yet. Run the agent for a full cycle first.")
        else:
            latest = fb.iloc[0]
            st.subheader(f"Week {int(latest['week_number'])} / {int(latest['year'])}")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Scraped",   int(latest["jobs_scraped"]  or 0))
            c2.metric("Applied",   int(latest["jobs_applied"]  or 0))
            c3.metric("Rejected",  int(latest["rejections"]    or 0))
            c4.metric("Avg score", f"{latest['avg_match_score'] or 0}%")

            if latest["resume_improvements"]:
                st.info(f"💡 {latest['resume_improvements']}")
            if latest["suggested_keywords"]:
                kws  = latest["suggested_keywords"].split(",")
                cols = st.columns(min(len(kws),6))
                for i,kw in enumerate(kws[:6]): cols[i].success(kw.strip())
            if latest["top_rejection_reasons"]:
                try:
                    for r in json.loads(latest["top_rejection_reasons"]): st.error(f"• {r}")
                except Exception:
                    st.write(latest["top_rejection_reasons"])

            st.divider()
            if len(fb) > 1:
                cd = fb[["week_number","jobs_applied","rejections","interviews"]].copy()
                cd["week"] = cd["week_number"].astype(str)
                st.line_chart(cd.set_index("week")[["jobs_applied","rejections","interviews"]],
                              use_container_width=True)
            st.dataframe(fb, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"DB error: {e}")


# ══════════════════════════════════════════════════════
# PAGE: RUN AGENT
# ══════════════════════════════════════════════════════
elif page == "🚀 Run Agent":
    st.title("Run Agent")
    st.caption("Full automated pipeline — search → score → shortlist → rewrite → cover letter → apply")
 
    if not is_setup_done:
        st.error("⚠️ Upload your resume in 📄 Resumes before running the agent.")
        st.stop()
 
    try:
        ready = run_query("SELECT COUNT(*) as n FROM jobs WHERE status IN ('new','shortlisted')").iloc[0]["n"]
        if ready > 0: st.success(f"✅ {ready} jobs ready to be processed")
        else:         st.warning("No new jobs — the scheduler will search automatically, or go to 💼 Jobs → Quick Search.")
    except Exception:
        pass
 
    st.divider()
    
    # ── Model Settings ─────────────────────────────────
    st.subheader("🧠 Model Settings")
    model_provider = st.selectbox(
        "LLM Provider",
        ["Ollama (local)", "Google Gemini", "Qwen (Ollama)", "Qwen (API)"],
        index=0
    )

    if model_provider == "Ollama (local)":
        model_name = st.selectbox("Model", ["llama3.1", "mistral", "phi3"])
    elif model_provider == "Google Gemini":
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"])
        gemini_key = st.text_input("Google API Key", type="password",
                                   value=os.getenv("GOOGLE_API_KEY", ""))
        if gemini_key:
            update_env("GOOGLE_API_KEY", gemini_key)
    elif model_provider == "Qwen (Ollama)":
        model_name = st.selectbox("Model", ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5-coder:7b"])
    elif model_provider == "Qwen (API)":
        model_name = "qwen-plus"
        qwen_key = st.text_input("Qwen API Key", type="password",
                                  value=os.getenv("QWEN_API_KEY", ""))


    # ── Search settings ───────────────────────────────
    st.subheader("🔍 Search Settings")
    col1, col2 = st.columns(2)
    with col1:
        query    = st.text_input("Job query",  value=profile.get("target_role","ML Engineer"))
        location = st.text_input("Location",   value=profile.get("location","Hyderabad, India"))
        max_jobs = st.slider("Max jobs", 5, 100, int(profile.get("max_jobs",20)))
    with col2:
        sources    = st.multiselect("Sources", ["linkedin","indeed","naukri","internshala"], default=profile.get("sources",["indeed"]))
        auto_apply = st.toggle("Auto apply", value=profile.get("auto_apply",False))
        dry_run    = st.toggle("Dry run",    value=profile.get("dry_run",True))
 
    if dry_run:    st.warning("Dry run is ON — no real applications will be sent")
    if auto_apply and not dry_run: st.error("⚠️ Auto apply ON + dry run OFF — REAL applications WILL be sent!")
 
    st.divider()
 
    # ── Shortlisting filters ──────────────────────────
    st.subheader("🎯 Shortlisting Filters")
    st.caption("Jobs that don't pass these filters will be skipped before resume rewriting.")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider(
            "Minimum match score (%)", 0, 100, int(profile.get("min_score", 60)),
            help="Jobs below this ATS score are dropped before rewriting"
        )
        blacklisted_companies = st.text_input(
            "Blacklisted companies (comma-separated)",
            value=profile.get("blacklisted_companies", ""),
            placeholder="e.g. TCS, Infosys, Wipro",
            help="Jobs from these companies will be skipped"
        )
    with col2:
        required_keywords = st.text_input(
            "Required keywords (comma-separated)",
            value=profile.get("required_keywords", ""),
            placeholder="e.g. Python, remote, senior",
            help="At least one of these must appear in the job description"
        )
        exclude_keywords = st.text_input(
            "Exclude keywords (comma-separated)",
            value=profile.get("exclude_keywords", ""),
            placeholder="e.g. unpaid, internship, onsite-only",
            help="Jobs containing any of these words are dropped"
        )
    with col3:
        shortlist_max = st.number_input(
            "Max jobs to process after shortlisting",
            min_value=1, max_value=50, value=int(profile.get("shortlist_max", 10)),
            help="Cap how many shortlisted jobs get resume rewrites and cover letters"
        )
        cover_letter_toggle = st.toggle(
            "Generate cover letter", value=profile.get("cover_letter_enabled", True),
            help="Write a tailored cover letter for each shortlisted job"
        )
 
    st.divider()
 
    # ── Auto-scheduler ────────────────────────────────
    st.subheader("⏰ Auto-Scheduler")
    st.caption("Automatically search every hour and apply only to newly found jobs.")
 
    if "scheduler_running" not in st.session_state:
        st.session_state.scheduler_running = False
    if "scheduler_last_run" not in st.session_state:
        st.session_state.scheduler_last_run = None
    if "scheduler_next_run" not in st.session_state:
        st.session_state.scheduler_next_run = None
 
    sched_col1, sched_col2, sched_col3 = st.columns(3)
    with sched_col1:
        scheduler_interval = st.number_input(
            "Scrape interval (minutes)", min_value=10, max_value=1440,
            value=int(profile.get("scheduler_interval_mins", 60)),
            help="How often to auto-search for new jobs"
        )
    with sched_col2:
        if st.session_state.scheduler_last_run:
            st.metric("Last run", st.session_state.scheduler_last_run.strftime("%H:%M:%S"))
        else:
            st.metric("Last run", "Never")
    with sched_col3:
        if st.session_state.scheduler_next_run:
            st.metric("Next run", st.session_state.scheduler_next_run.strftime("%H:%M:%S"))
        else:
            st.metric("Next run", "—")
 
    sched_btn_col1, sched_btn_col2 = st.columns(2)
    with sched_btn_col1:
        if st.button(
            "▶️ Start Scheduler" if not st.session_state.scheduler_running else "⏸️ Scheduler Running...",
            use_container_width=True,
            type="primary" if not st.session_state.scheduler_running else "secondary",
            disabled=st.session_state.scheduler_running,
        ):
            st.session_state.scheduler_running = True
            st.session_state.scheduler_next_run = datetime.now(IST)
            st.rerun()
    with sched_btn_col2:
        if st.button("⏹️ Stop Scheduler", use_container_width=True,
                     disabled=not st.session_state.scheduler_running):
            st.session_state.scheduler_running = False
            st.session_state.scheduler_next_run = None
            st.success("Scheduler stopped.")
            st.rerun()
 
    # ── Scheduler tick ────────────────────────────────
    # If scheduler is active and it's time to run, fire the agent automatically
    if st.session_state.scheduler_running and st.session_state.scheduler_next_run:
        now = datetime.now(IST)
        if now >= st.session_state.scheduler_next_run:
            st.info(f"⏰ Scheduled run triggered at {now.strftime('%H:%M:%S')} IST")
 
            # Collect job IDs that already exist BEFORE this scrape
            try:
                existing_ids = set(
                    run_query("SELECT id FROM jobs")["id"].tolist()
                )
            except Exception:
                existing_ids = set()
 
            with st.status("⏰ Scheduler: Agent running...", expanded=True) as sched_sb:
                try:
                    os.environ["LINKEDIN_EMAIL"]    = profile.get("linkedin_email","")
                    os.environ["LINKEDIN_PASSWORD"] = st.session_state.linkedin_password or ""
                    os.environ["AGENT_DRY_RUN"]     = str(dry_run).lower()
                    os.environ["AGENT_AUTO_APPLY"]  = str(auto_apply).lower()
                    os.environ["AGENT_SOURCES"]     = ",".join(sources)
 
                    blacklist_note = (
                        f"Skip jobs from these companies: {blacklisted_companies}. "
                        if blacklisted_companies.strip() else ""
                    )
                    req_kw_note = (
                        f"Only keep jobs that mention at least one of: {required_keywords}. "
                        if required_keywords.strip() else ""
                    )
                    excl_kw_note = (
                        f"Drop jobs containing any of these words: {exclude_keywords}. "
                        if exclude_keywords.strip() else ""
                    )
                    cover_note = (
                        "For each shortlisted job, write a concise, tailored cover letter "
                        "referencing the job title, company name, and top 2–3 matching skills from the resume. "
                        "Save it alongside the tailored resume. "
                        if cover_letter_toggle else ""
                    )
 
                    
                    sched_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(sched_loop)
                    result = sched_loop.run_until_complete(run_agent(
                        instruction=(
                            f"Run a job hunt cycle: "
                            f"1) Search for '{query}' jobs in {location} using {', '.join(sources)} (max {max_jobs} jobs). "
                            f"2) Score each job against the user resume using ATS matching. "
                            f"3) Shortlist jobs scoring >= {min_score}%. {blacklist_note}{req_kw_note}{excl_kw_note}"
                            f"Take the top {shortlist_max} shortlisted jobs. Mark dropped jobs as 'skipped'. "
                            f"4) For each shortlisted job, rewrite the resume. {cover_note}"
                            f"5) Give a summary: jobs scraped, shortlisted, skipped."
                        ),
                    ))
                    
                    # ── Step 2: direct apply (bypass LLM for reliability) ──
                    if auto_apply:
                        try:
                            from app.tools.apply_job import run_apply_job
                            new_ids = set(run_query("SELECT id FROM jobs WHERE status='shortlisted'")["id"].tolist())
                            if new_ids:
                                st.write(f"📨 Applying to {len(new_ids)} shortlisted job(s)...")
                                applied, skipped, errors = 0, 0, 0
                                for job_id in sorted(new_ids):
                                    a_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(a_loop)
                                    res = a_loop.run_until_complete(run_apply_job(
                                        job_id=int(job_id),
                                        apply_method="manual",
                                        dry_run=dry_run,
                                    ))
                                    if res.get("error"):
                                        errors += 1
                                        st.warning(f"Job {job_id}: {res['error']}")
                                    elif res.get("status") == "already_queued":
                                        skipped += 1
                                    else:
                                        applied += 1
                                        st.success(f"✅ Queued: {res.get('job_title')} @ {res.get('company')}")
                                st.info(f"Apply complete — queued: {applied}, skipped: {skipped}, errors: {errors}")
                            else:
                                st.info("No new shortlisted jobs to apply to.")
                        except Exception as apply_err:
                            st.warning(f"Apply error: {apply_err}")
 
                    st.session_state.scheduler_last_run = now
                    st.session_state.scheduler_next_run = datetime.now(IST) + timedelta(minutes=scheduler_interval)
                    sched_sb.update(label="✅ Scheduled run complete!", state="complete")
                    st.markdown(result.get("output","No output"))
 
                except ImportError as e:
                    sched_sb.update(label="Import error", state="error")
                    st.error(f"Import error: {e}")
                except Exception as e:
                    sched_sb.update(label="Error", state="error")
                    st.error(f"Scheduler agent error: {e}")
 
            # Auto-rerun after scheduler_interval to keep the loop alive
            import time
            time.sleep(1)
            st.rerun()
 
        else:
            # Not yet time — show countdown and rerun after a short pause
            remaining = int((st.session_state.scheduler_next_run - now).total_seconds())
            st.info(f"🕐 Next scheduled run in **{remaining // 60}m {remaining % 60}s**")
            import time
            time.sleep(10)
            st.rerun()
 
    st.divider()
 
    # ── Manual run ────────────────────────────────────
    st.subheader("▶️ Manual Run")
    if st.button("🚀 Run Agent Now", use_container_width=True, type="primary"):
        with st.status("Agent running...", expanded=True) as sb:
            try:
                os.environ["LINKEDIN_EMAIL"]    = profile.get("linkedin_email","")
                os.environ["LINKEDIN_PASSWORD"] = st.session_state.linkedin_password or ""
                os.environ["AGENT_DRY_RUN"]     = str(dry_run).lower()
                os.environ["AGENT_AUTO_APPLY"]  = str(auto_apply).lower()
                os.environ["AGENT_SOURCES"]     = ",".join(sources)
                st.write(f"🔍 Searching '{query}' in {location}...")
 
                # Snapshot existing IDs before this run so we don't re-apply
                try:
                    existing_ids = set(run_query("SELECT id FROM jobs")["id"].tolist())
                except Exception:
                    existing_ids = set()
 
                from app.agent.agent import run_agent
 
                blacklist_note = (
                    f"Skip jobs from these companies: {blacklisted_companies}. "
                    if blacklisted_companies.strip() else ""
                )
                req_kw_note = (
                    f"Only keep jobs that mention at least one of: {required_keywords}. "
                    if required_keywords.strip() else ""
                )
                excl_kw_note = (
                    f"Drop jobs containing any of these words: {exclude_keywords}. "
                    if exclude_keywords.strip() else ""
                )
                cover_note = (
                    "For each shortlisted job, write a concise, tailored cover letter "
                    "referencing the job title, company name, and top 2–3 matching skills from the resume. "
                    "Save it alongside the tailored resume. "
                    if cover_letter_toggle else ""
                )
 
                
                manual_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(manual_loop)
                result = manual_loop.run_until_complete(run_agent(
                    instruction=(
                        f"Run a job hunt cycle: "
                        f"1) Search for '{query}' jobs in {location} using {', '.join(sources)} (max {max_jobs} jobs). "
                        f"2) Score each job against the user resume using ATS matching. "
                        f"3) Shortlist jobs scoring >= {min_score}%. {blacklist_note}{req_kw_note}{excl_kw_note}"
                        f"Take the top {shortlist_max} shortlisted jobs. Mark dropped jobs as 'skipped'. "
                        f"4) For each shortlisted job, rewrite the resume. {cover_note}"
                        f"5) Give a summary: jobs scraped, shortlisted, skipped."
                    ),
                ))

                # ── Step 2: direct apply (bypass LLM for reliability) ──
                if auto_apply:
                    try:
                        from app.tools.apply_job import run_apply_job
                        new_ids = set(run_query("SELECT id FROM jobs WHERE status='shortlisted'")["id"].tolist())
                        if new_ids:
                            st.write(f"📨 Applying to {len(new_ids)} shortlisted job(s)...")
                            applied, skipped, errors = 0, 0, 0
                            for job_id in sorted(new_ids):
                                a_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(a_loop)
                                res = a_loop.run_until_complete(run_apply_job(
                                    job_id=int(job_id),
                                    apply_method="manual",
                                    dry_run=dry_run,
                                ))
                                if res.get("error"):
                                    errors += 1
                                    st.warning(f"Job {job_id}: {res['error']}")
                                elif res.get("status") == "already_queued":
                                    skipped += 1
                                else:
                                    applied += 1
                                    st.success(f"✅ Queued: {res.get('job_title')} @ {res.get('company')}")
                            st.info(f"Apply complete — queued: {applied}, skipped: {skipped}, errors: {errors}")
                        else:
                            st.info("No new shortlisted jobs to apply to.")
                    except Exception as apply_err:
                        st.warning(f"Apply error: {apply_err}")
                
                sb.update(label="✅ Agent finished!", state="complete")
                st.success(f"Completed in {result.get('steps','?')} steps")
                st.markdown(result.get("output","No output"))
            except ImportError as e:
                sb.update(label="Import error", state="error"); st.error(f"Import error: {e}")
            except Exception as e:
                sb.update(label="Error", state="error"); st.error(f"Agent error: {e}")
 

# ══════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("Settings")
    st.subheader("Account")
    st.info(f"Signed in as **{profile.get('gmail_email','Unknown')}**")

    st.divider()
    st.subheader("🔐 Platform Credentials")
    st.caption("Used for auto-applying via Selenium. Passwords are never saved to disk.")

    # ── LinkedIn ──────────────────────────────────────
    with st.expander("🔵 LinkedIn", expanded=True):
        col_li1, col_li2 = st.columns(2)
        with col_li1:
            li_email = st.text_input("LinkedIn email",
                value=profile.get("linkedin_email",""),
                placeholder="your@linkedin.com")
        with col_li2:
            li_pass = st.text_input("LinkedIn password",
                type="password",
                placeholder="LinkedIn password",
                value="")
        if st.button("💾 Save LinkedIn credentials", use_container_width=True):
            upd = {**profile, "linkedin_email": li_email}
            if li_pass:
                upd["linkedin_password"] = li_pass
                st.session_state.linkedin_password = li_pass
                update_env("LINKEDIN_EMAIL",    li_email)
                update_env("LINKEDIN_PASSWORD", li_pass)
            save_profile(upd)
            st.session_state.profile = upd
            st.success("✅ LinkedIn credentials saved!")

    # ── Naukri ────────────────────────────────────────
    with st.expander("🟧 Naukri"):
        col_nk1, col_nk2 = st.columns(2)
        with col_nk1:
            nk_email = st.text_input("Naukri email",
                value=profile.get("naukri_email",""),
                placeholder="your@email.com",
                key="naukri_email_input")
        with col_nk2:
            nk_pass = st.text_input("Naukri password",
                type="password",
                placeholder="Naukri password",
                value="",
                key="naukri_pass_input")
        if st.button("💾 Save Naukri credentials", use_container_width=True, key="save_naukri"):
            upd = {**profile, "naukri_email": nk_email}
            if nk_pass:
                upd["naukri_password"] = nk_pass
                st.session_state.naukri_password = nk_pass
                update_env("NAUKRI_EMAIL",    nk_email)
                update_env("NAUKRI_PASSWORD", nk_pass)
            save_profile(upd)
            st.session_state.profile = upd
            st.success("✅ Naukri credentials saved!")

    # ── Internshala ───────────────────────────────────
    with st.expander("🟢 Internshala"):
        col_is1, col_is2 = st.columns(2)
        with col_is1:
            is_email = st.text_input("Internshala email",
                value=profile.get("internshala_email",""),
                placeholder="your@email.com",
                key="internshala_email_input")
        with col_is2:
            is_pass = st.text_input("Internshala password",
                type="password",
                placeholder="Internshala password",
                value="",
                key="internshala_pass_input")
        if st.button("💾 Save Internshala credentials", use_container_width=True, key="save_internshala"):
            upd = {**profile, "internshala_email": is_email}
            if is_pass:
                upd["internshala_password"] = is_pass
                st.session_state.internshala_password = is_pass
                update_env("INTERNSHALA_EMAIL",    is_email)
                update_env("INTERNSHALA_PASSWORD", is_pass)
            save_profile(upd)
            st.session_state.profile = upd
            st.success("✅ Internshala credentials saved!")

    st.divider()
    st.subheader("Clear data")
    st.warning("This permanently deletes data from your local database.")
    col1,col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear all jobs",         use_container_width=True):
            run_update("DELETE FROM jobs");         st.success("Jobs cleared"); st.rerun()
        if st.button("🗑️ Clear all applications", use_container_width=True):
            run_update("DELETE FROM applications"); st.success("Applications cleared"); st.rerun()
        if st.button("🗑️ Clear base resume",       use_container_width=True):
            run_update("DELETE FROM resumes WHERE is_base=1"); st.success("Base resume cleared"); st.rerun()
    with col2:
        if st.button("🗑️ Clear feedback history", use_container_width=True):
            run_update("DELETE FROM feedback");     st.success("Feedback cleared"); st.rerun()
        if st.button("🗑️ Clear tailored resumes", use_container_width=True):
            run_update("DELETE FROM resumes WHERE is_base=0"); st.success("Tailored resumes cleared"); st.rerun()

    st.divider()
    st.subheader("Reset Profile")
    st.warning("This will clear your saved profile (name, email, LinkedIn credentials). Your `.env` API keys are not affected.")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔄 Reset profile", use_container_width=True):
            profile_path = os.path.join(PROJECT_ROOT, "user_profile.json")
            if os.path.exists(profile_path):
                os.remove(profile_path)
            st.session_state.profile = {}
            st.session_state.linkedin_password = ""
            st.success("Profile reset. Please re-enter your details in the Profile page.")
            st.rerun()
    with col4:
        if st.button("💣 Full reset (all data + profile)", use_container_width=True, type="primary"):
            for tbl in ["jobs", "applications", "feedback", "resumes"]:
                try:
                    run_update(f"DELETE FROM {tbl}")
                except Exception:
                    pass
            profile_path = os.path.join(PROJECT_ROOT, "user_profile.json")
            if os.path.exists(profile_path):
                os.remove(profile_path)
            st.session_state.profile = {}
            st.session_state.linkedin_password = ""
            st.success("✅ Full reset complete. Everything has been cleared.")
            st.rerun()

    st.divider()
    st.subheader("Database summary")
    try:
        for tbl in ["jobs","resumes","applications","feedback"]:
            count = run_query(f"SELECT COUNT(*) as n FROM {tbl}").iloc[0]["n"]
            st.text(f"  {tbl:<20}  {count} rows")
    except Exception as e:
        st.error(f"DB error: {e}")