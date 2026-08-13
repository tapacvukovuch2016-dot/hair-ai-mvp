
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from vision_analyzer import analyze_image, to_ranking_profile
from photo_validator import validate_photo
from ranking import rank

load_dotenv()

st.set_page_config(page_title="Hair AI", page_icon="✂️", layout="centered")

st.markdown("""
<style>
.block-container {
  max-width: 720px;
  padding-top: 1rem;
  padding-bottom: 4rem;
}
h1 {font-size: 2rem !important;}
.rec-card {
  padding: 16px;
  border: 1px solid rgba(128,128,128,.22);
  border-radius: 18px;
  margin: 12px 0;
}
.small {opacity:.72; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

for key, default in {
    "liked": [],
    "disliked": [],
    "last_result": None,
    "last_profile": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("✂️ Hair AI")
st.caption("MVP v0.4 — мобильный прототип подбора причёски")

st.subheader("1. Твои предпочтения")

desired = st.segmented_control(
    "Желаемая длина",
    ["short", "medium", "long"],
    default="short",
    format_func=lambda x: {
        "short":"Короткая",
        "medium":"Средняя",
        "long":"Длинная"
    }[x],
)

styling = st.segmented_control(
    "Сколько времени готов(а) тратить на укладку?",
    ["minimal", "low", "medium", "high"],
    default="low",
    format_func=lambda x: {
        "minimal":"Почти никакой",
        "low":"Немного",
        "medium":"Средне",
        "high":"Не проблема"
    }[x],
)

col1, col2 = st.columns(2)
with col1:
    no_bangs = st.checkbox("Не хочу чёлку")
with col2:
    no_long = st.checkbox("Не хочу длинные волосы")

free_text = st.text_area(
    "Что ещё важно?",
    placeholder="Например: хочу более современный образ, но без ежедневной укладки",
)

st.subheader("2. Фото")
photo = st.file_uploader(
    "Сделай фото или выбери из медиатеки",
    type=["jpg","jpeg","png","webp"],
    accept_multiple_files=False,
)

if photo:
    st.image(photo, caption="Фото для анализа", use_container_width=True)

run = st.button(
    "Проанализировать и подобрать",
    type="primary",
    use_container_width=True,
    disabled=photo is None,
)

if run:
    avoid = []
    if no_bangs:
        avoid.append("bangs")
    if no_long:
        avoid.append("long_hair")

    preferences = {
        "desired_length": desired,
        "max_styling": styling,
        "avoid": avoid,
        "history": {
            "liked": st.session_state.liked,
            "disliked": st.session_state.disliked,
        },
        "free_text": free_text,
    }

    suffix = Path(photo.name).suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(photo.getvalue())
        temp_path = tmp.name

    try:
        with st.spinner("Анализирую фото..."):
            analysis = analyze_image(temp_path)
            validation = validate_photo(analysis["photo_observation"])

            result = {
                "analysis": analysis,
                "validation": validation,
                "recommendations": [],
            }

            if validation["decision"] != "reject":
                profile = to_ranking_profile(analysis, preferences)
                result["profile"] = profile
                result["recommendations"] = rank(profile, top_n=5)
                st.session_state.last_profile = profile

            st.session_state.last_result = result
    except Exception as exc:
        st.error(f"Не удалось выполнить анализ: {exc}")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

result = st.session_state.last_result

if result:
    st.subheader("3. Проверка фото")

    validation = result["validation"]
    decision_map = {
        "accept": "✅ Фото подходит",
        "partial": "⚠️ Фото частично подходит",
        "reject": "❌ Нужен другой снимок",
    }
    st.write(decision_map.get(validation["decision"], validation["decision"]))

    if validation.get("next_requests"):
        for item in validation["next_requests"]:
            st.info(item)

    with st.expander("Что AI увидел на фото"):
        st.json(result["analysis"])

    recs = result.get("recommendations", [])
    if recs:
        st.subheader("4. Рекомендации")

        for rec in recs:
            st.markdown(
                f"""
                <div class="rec-card">
                  <h3>{rec['name']}</h3>
                  <div class="small">Совместимость MVP: {rec['score']}/100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for reason in rec.get("reasons", []):
                st.write("✓", reason)
            for warning in rec.get("warnings", []):
                st.write("⚠", warning)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍 Нравится", key=f"like_{rec['id']}", use_container_width=True):
                    if rec["id"] not in st.session_state.liked:
                        st.session_state.liked.append(rec["id"])
                    if rec["id"] in st.session_state.disliked:
                        st.session_state.disliked.remove(rec["id"])
                    st.success("Запомнил положительную реакцию.")
            with c2:
                if st.button("👎 Не моё", key=f"dislike_{rec['id']}", use_container_width=True):
                    if rec["id"] not in st.session_state.disliked:
                        st.session_state.disliked.append(rec["id"])
                    if rec["id"] in st.session_state.liked:
                        st.session_state.liked.remove(rec["id"])
                    st.warning("Запомнил отрицательную реакцию.")

        if st.session_state.last_profile:
            if st.button("Пересчитать с учётом feedback", use_container_width=True):
                profile = dict(st.session_state.last_profile)
                profile["history"] = {
                    "liked": st.session_state.liked,
                    "disliked": st.session_state.disliked,
                }
                result["recommendations"] = rank(profile, top_n=5)
                st.session_state.last_result = result
                st.rerun()

st.divider()
st.caption(
    "v0.4 — экспериментальный прототип. Он не оценивает красоту человека и не заменяет консультацию парикмахера."
)
