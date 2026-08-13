import os, tempfile, json, base64
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFilter
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
st.caption("MVP v0.9 — Local Head Edit")

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



def _local_head_crop_box(source: Image.Image):
    """
    v0.9: choose a square local crop around the upper/central portrait area.
    We edit this crop only, never the full photo.
    """
    w, h = source.size

    # Large enough for hair + upper face, but not the whole scene.
    side = int(min(w, h * 0.58))
    side = max(512, min(side, w, h))

    cx = w // 2
    left = max(0, cx - side // 2)
    right = left + side
    if right > w:
        right = w
        left = w - side

    # Start near the top for typical portrait/selfie framing.
    top = 0
    bottom = side
    if bottom > h:
        bottom = h
        top = h - side

    return (left, top, right, bottom)


def _build_local_edit_mask(size=(1024, 1024)) -> Image.Image:
    """
    Transparent = editable; opaque = protected.
    This mask is defined in the local square crop coordinate system.
    """
    w, h = size
    mask = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)

    # Editable upper-head/hair zone.
    draw.ellipse(
        (int(w*0.17), int(h*0.02), int(w*0.83), int(h*0.62)),
        fill=(255, 255, 255, 0)
    )

    # Protect the central/lower face.
    draw.ellipse(
        (int(w*0.29), int(h*0.33), int(w*0.71), int(h*0.82)),
        fill=(255, 255, 255, 255)
    )

    alpha = mask.getchannel("A").filter(ImageFilter.GaussianBlur(radius=8))
    mask.putalpha(alpha)
    return mask


def _build_local_composite_alpha(size=(1024, 1024)) -> Image.Image:
    """
    White = take AI pixels; black = keep original crop pixels.
    Narrower than the edit mask to keep the face physically original.
    """
    w, h = size
    alpha = Image.new("L", size, 0)
    draw = ImageDraw.Draw(alpha)

    draw.ellipse(
        (int(w*0.18), int(h*0.03), int(w*0.82), int(h*0.60)),
        fill=255
    )

    # Remove the face area from the generated contribution.
    draw.ellipse(
        (int(w*0.28), int(h*0.31), int(w*0.72), int(h*0.84)),
        fill=0
    )

    return alpha.filter(ImageFilter.GaussianBlur(radius=7))


def generate_tryon(style_name):
    """
    v0.9 Local Head Edit:
    1. Crop only the head/upper-face area from the original.
    2. Resize this crop to 1024x1024.
    3. Edit only the hair region inside that crop.
    4. Composite only hair-area AI pixels onto the ORIGINAL crop.
    5. Paste that crop back at the exact original coordinates.

    No full-frame AI result is ever stretched over the original photo.
    """
    if not st.session_state.photo_bytes:
        return None

    client = OpenAI()
    image_path = None
    mask_path = None

    try:
        from io import BytesIO

        source = Image.open(BytesIO(st.session_state.photo_bytes))
        source = ImageOps.exif_transpose(source).convert("RGB")

        crop_box = _local_head_crop_box(source)
        original_crop = source.crop(crop_box).convert("RGB")

        work_size = (1024, 1024)
        work_crop = original_crop.resize(work_size, Image.Resampling.LANCZOS)
        edit_mask = _build_local_edit_mask(work_size)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            image_path = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            mask_path = f.name

        work_crop.save(image_path, format="PNG")
        edit_mask.save(mask_path, format="PNG")

        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            r = client.images.edit(
                model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
                image=image_file,
                mask=mask_file,
                prompt=(
                    f"Change only the hairstyle in this local head crop to a photorealistic {style_name}. "
                    "Keep the exact same person and head pose. Preserve natural hair color, plausible density, "
                    "visible scalp and hairline. Do not change the eyes, eyebrows, nose, mouth, teeth, smile, "
                    "ears, skin, glasses, facial hair, lighting, perspective or background visible in the crop. "
                    "The result should look like the same original photo after a real salon haircut."
                ),
                size="1024x1024",
                quality="medium"
            )

        generated_bytes = base64.b64decode(r.data[0].b64_json)
        generated_crop = Image.open(BytesIO(generated_bytes)).convert("RGB")

        # Composite AI hair onto the ORIGINAL crop in the exact same local coordinates.
        local_alpha = _build_local_composite_alpha(work_size)
        protected_work = work_crop.convert("RGB")
        composited_work = Image.composite(generated_crop, protected_work, local_alpha)

        # Scale back exactly to the original crop dimensions.
        composited_crop = composited_work.resize(original_crop.size, Image.Resampling.LANCZOS)

        # Paste only this local crop back into a full copy of the untouched original.
        final = source.copy()
        final.paste(composited_crop, (crop_box[0], crop_box[1]))

        out = BytesIO()
        final.save(out, format="JPEG", quality=96)
        return out.getvalue()

    finally:
        for p in (image_path, mask_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
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
                        st.caption("AI-визуализация — ориентир. v0.9 редактирует только локальный квадрат вокруг головы и вставляет его обратно по тем же координатам.")
                except Exception as e:
                    st.error(f"Не удалось создать примерку: {e}")
            st.divider()

        if st.button("Пересчитать с учётом 👍/👎",use_container_width=True):
            p=dict(st.session_state.last_profile)
            p["history"]={"liked":st.session_state.liked,"disliked":st.session_state.disliked}
            r["recommendations"]=rank(p,5); st.session_state.last_result=r; st.rerun()

st.caption("v0.9 — экспериментальный прототип. Не оценивает красоту человека.")
