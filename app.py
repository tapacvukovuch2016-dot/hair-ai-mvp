import os, tempfile, json, base64
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps
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
    from io import BytesIO
    raw = photo.getvalue()
    try:
        normalized = Image.open(BytesIO(raw))
        normalized = ImageOps.exif_transpose(normalized)
        if normalized.mode not in ("RGB", "RGBA"):
            normalized = normalized.convert("RGB")
        buf = BytesIO()
        normalized.save(buf, format="PNG")
        st.session_state.photo_bytes = buf.getvalue()
        st.image(st.session_state.photo_bytes, use_container_width=True)
    except Exception:
        st.session_state.photo_bytes = raw
        st.image(raw, use_container_width=True)

if st.button("Проанализировать и подобрать",type="primary",use_container_width=True,disabled=photo is None):
    avoid=[]
    if no_bangs: avoid.append("bangs")
    if no_long: avoid.append("long_hair")
    prefs={"desired_length":desired,"max_styling":styling,"avoid":avoid,
           "history":{"liked":st.session_state.liked,"disliked":st.session_state.disliked},
           "free_text":free_text}
    suffix=".png"
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        tmp.write(st.session_state.photo_bytes); p=tmp.name
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
    if not st.session_state.photo_bytes:
        return None

    client = OpenAI()
    p = None
    try:
        # Normalize whatever the iPhone/browser uploaded into a real RGB PNG.
        # This avoids sending JPEG/HEIC/etc. bytes under the wrong file extension.
        from io import BytesIO
        img = Image.open(BytesIO(st.session_state.photo_bytes))
        img = ImageOps.exif_transpose(img)

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # Keep a reasonable input size while preserving aspect ratio.
        max_side = 2048
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            p = f.name
        img.save(p, format="PNG", optimize=True)

        with open(p, "rb") as image_file:
            r = client.images.edit(
                model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
                image=image_file,
                prompt=(
                    f"Change ONLY the hairstyle to a realistic {style_name}. "
                    "The person's identity must remain exactly the same. Do not alter facial geometry, "
                    "eyes, eyebrows, nose, mouth, teeth, smile, ears, skin texture, glasses, facial hair, "
                    "head pose, body, clothing, hands, objects, background, camera perspective or lighting. "
                    "Preserve the person's natural hair color, realistic density, hairline and visible scalp characteristics "
                    "unless the requested haircut necessarily changes hair length. Do not beautify or retouch the face. "
                    "Keep the result photorealistic and salon-plausible. The only intended edit is the haircut."
                ),
                size="1024x1024",
                quality="low"
            )

        generated_bytes = base64.b64decode(r.data[0].b64_json)

        # Restore the source photo's aspect ratio for display.
        # This avoids showing the user a permanently square/cropped result.
        from io import BytesIO
        source = Image.open(BytesIO(st.session_state.photo_bytes))
        source = ImageOps.exif_transpose(source)
        generated = Image.open(BytesIO(generated_bytes)).convert("RGB")
        target_ratio = source.width / source.height
        gen_ratio = generated.width / generated.height

        if abs(gen_ratio - target_ratio) > 0.01:
            if gen_ratio > target_ratio:
                new_w = int(generated.height * target_ratio)
                left = max(0, (generated.width - new_w) // 2)
                generated = generated.crop((left, 0, left + new_w, generated.height))
            else:
                new_h = int(generated.width / target_ratio)
                top = max(0, (generated.height - new_h) // 2)
                generated = generated.crop((0, top, generated.width, top + new_h))

        out = BytesIO()
        generated.save(out, format="JPEG", quality=95)
        return out.getvalue()
    finally:
        if p:
            try:
                os.remove(p)
            except:
                pass

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
