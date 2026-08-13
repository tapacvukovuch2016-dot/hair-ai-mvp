import os, tempfile, json, base64
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from vision_analyzer import analyze_image, to_ranking_profile
from photo_validator import validate_photo
from ranking import rank

load_dotenv()
st.set_page_config(page_title="Hair AI", page_icon="✂️", layout="centered")
META=json.loads(Path("hairstyle_metadata.json").read_text(encoding="utf-8"))

for k,v in {"liked":[],"disliked":[],"last_result":None,"last_profile":None,"photo_bytes":None}.items():
    if k not in st.session_state: st.session_state[k]=v

st.title("✂️ Hair AI")
st.caption("MVP v0.5 — рекомендации + визуальная примерка")

st.subheader("1. Твои предпочтения")
desired=st.segmented_control("Желаемая длина",["short","medium","long"],default="short",
 format_func=lambda x:{"short":"Короткая","medium":"Средняя","long":"Длинная"}[x])
styling=st.segmented_control("Сколько укладки?",["minimal","low","medium","high"],default="low",
 format_func=lambda x:{"minimal":"Почти никакой","low":"Немного","medium":"Средне","high":"Не проблема"}[x])
c1,c2=st.columns(2)
with c1: no_bangs=st.checkbox("Не хочу чёлку")
with c2: no_long=st.checkbox("Не хочу длинные волосы")
free_text=st.text_area("Что ещё важно?",placeholder="Например: современно, но без ежедневной укладки")

st.subheader("2. Фото")
photo=st.file_uploader("Сделай фото или выбери из медиатеки",type=["jpg","jpeg","png","webp"])
if photo:
    st.session_state.photo_bytes=photo.getvalue()
    st.image(photo,use_container_width=True)

if st.button("Проанализировать и подобрать",type="primary",use_container_width=True,disabled=photo is None):
    avoid=[]
    if no_bangs: avoid.append("bangs")
    if no_long: avoid.append("long_hair")
    prefs={"desired_length":desired,"max_styling":styling,"avoid":avoid,
           "history":{"liked":st.session_state.liked,"disliked":st.session_state.disliked},
           "free_text":free_text}
    suffix=Path(photo.name).suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        tmp.write(photo.getvalue()); p=tmp.name
    try:
        with st.spinner("Анализирую фото..."):
            a=analyze_image(p); v=validate_photo(a["photo_observation"])
            result={"analysis":a,"validation":v,"recommendations":[]}
            if v["decision"]!="reject":
                profile=to_ranking_profile(a,prefs)
                result["profile"]=profile
                result["recommendations"]=rank(profile,5)
                st.session_state.last_profile=profile
            st.session_state.last_result=result
    except Exception as e: st.error(f"Не удалось выполнить анализ: {e}")
    finally:
        try: os.remove(p)
        except: pass

def generate_tryon(style_name):
    if not st.session_state.photo_bytes: return None
    client=OpenAI()
    with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as f:
        f.write(st.session_state.photo_bytes); p=f.name
    try:
        with open(p,"rb") as image_file:
            r=client.images.edit(
                model=os.getenv("OPENAI_IMAGE_MODEL","gpt-image-2"),
                image=image_file,
                prompt=(f"Edit only the hairstyle of the person to a realistic {style_name}. "
                        "Preserve identity, face, skin, expression, pose, clothing, background and lighting. "
                        "Make the haircut physically plausible and salon-realistic. No beauty retouching."),
                size="1024x1024",
                quality="low"
            )
        return base64.b64decode(r.data[0].b64_json)
    finally:
        try: os.remove(p)
        except: pass

r=st.session_state.last_result
if r:
    st.subheader("3. Проверка фото")
    v=r["validation"]
    st.write({"accept":"✅ Фото подходит","partial":"⚠️ Фото частично подходит","reject":"❌ Нужен другой снимок"}.get(v["decision"]))
    for x in v.get("next_requests",[]): st.info(x)
    with st.expander("Что AI увидел на фото"): st.json(r["analysis"])

    if r.get("recommendations"):
        st.subheader("4. Рекомендации")
        st.caption("Оценка — ориентир MVP, а не объективная оценка внешности.")
        for rec in r["recommendations"]:
            st.markdown(f"### {rec['name']}")
            st.write(META.get(rec["id"],{}).get("summary",""))
            st.progress(int(rec["score"]))
            st.caption(f"Совместимость с заданными параметрами: {rec['score']}/100")
            for x in rec.get("reasons",[]): st.write("✓",x)
            for x in rec.get("warnings",[]): st.write("⚠️",x)

            a,b=st.columns(2)
            with a:
                if st.button("👍 Нравится",key="l_"+rec["id"],use_container_width=True):
                    if rec["id"] not in st.session_state.liked: st.session_state.liked.append(rec["id"])
                    if rec["id"] in st.session_state.disliked: st.session_state.disliked.remove(rec["id"])
            with b:
                if st.button("👎 Не моё",key="d_"+rec["id"],use_container_width=True):
                    if rec["id"] not in st.session_state.disliked: st.session_state.disliked.append(rec["id"])
                    if rec["id"] in st.session_state.liked: st.session_state.liked.remove(rec["id"])

            if st.button("✨ Показать на мне",key="try_"+rec["id"],use_container_width=True):
                try:
                    with st.spinner("Создаю визуальную примерку..."):
                        out=generate_tryon(rec["name"])
                    if out:
                        st.image(out,caption=f"Визуализация: {rec['name']}",use_container_width=True)
                        st.caption("AI-визуализация — ориентир. Реальный результат стрижки может отличаться.")
                except Exception as e:
                    st.error(f"Не удалось создать примерку: {e}")
            st.divider()

        if st.button("Пересчитать с учётом 👍/👎",use_container_width=True):
            p=dict(st.session_state.last_profile)
            p["history"]={"liked":st.session_state.liked,"disliked":st.session_state.disliked}
            r["recommendations"]=rank(p,5); st.session_state.last_result=r; st.rerun()

st.caption("v0.5 — экспериментальный прототип. Не оценивает красоту человека.")
