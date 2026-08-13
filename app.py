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

for k,v in {"liked":[],"disliked":[],"last_result":None,"last_profile":None,"photo_bytes":None,"hair_polygon_cache":None,"hair_polygon_photo_key":None}.items():
    if k not in st.session_state: st.session_state[k]=v

st.title("✂️ Hair AI")
st.caption("MVP v0.21 — Live Diagnostic Timing")

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
        new_bytes = buf.getvalue()
        import hashlib
        new_key = hashlib.sha256(new_bytes).hexdigest()
        if st.session_state.get("hair_polygon_photo_key") != new_key:
            st.session_state.hair_polygon_cache = None
            st.session_state.hair_polygon_photo_key = new_key
        st.session_state.photo_bytes = new_bytes
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
    v0.12 keeps generation local to the upper portrait region.
    """
    w, h = source.size
    side = int(min(w, h * 0.78))
    side = max(512, min(side, w, h))

    cx = w // 2
    left = max(0, cx - side // 2)
    right = left + side
    if right > w:
        right = w
        left = w - side

    top = 0
    bottom = side
    if bottom > h:
        bottom = h
        top = h - side

    return (left, top, right, bottom)


def _crop_to_data_url(img: Image.Image) -> str:
    from io import BytesIO
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _detect_hair_polygon(work_crop: Image.Image):
    """
    Ask the vision model for an approximate editable hairstyle polygon in the
    LOCAL 1024x1024 crop. Coordinates are normalized 0..1000.

    The polygon should contain the current hair/scalp silhouette plus a small
    margin needed to reshape the haircut, while excluding face and background
    as much as possible.
    """
    client = OpenAI()
    schema = {
        "type": "object",
        "properties": {
            "polygon": {
                "type": "array",
                "minItems": 6,
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "y": {"type": "integer", "minimum": 0, "maximum": 1000}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                }
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["polygon", "confidence"],
        "additionalProperties": False
    }

    response = client.responses.create(
        model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.6-luna"),
        instructions=(
            "Locate only the visible hairstyle/scalp region of the main foreground person. "
            "Return one polygon around the entire editable hairstyle silhouette, including top hair, "
            "temples and side hair, plus a small margin that would allow a haircut silhouette to change. "
            "Do NOT include eyes, nose, mouth, teeth, glasses, neck, clothing, or broad background areas. "
            "Use the local image coordinates normalized from 0 to 1000. "
            "Follow the outer contour clockwise. If uncertain, be conservative."
        ),
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Find the editable hairstyle polygon for this portrait crop."},
                {"type": "input_image", "image_url": _crop_to_data_url(work_crop), "detail": "high"}
            ]
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "hair_polygon",
                "schema": schema,
                "strict": True
            }
        }
    )

    data = json.loads(response.output_text)
    if data.get("confidence", 0) < 0.45:
        return None

    w, h = work_crop.size
    pts = []
    for p in data["polygon"]:
        pts.append((
            int(p["x"] / 1000 * (w - 1)),
            int(p["y"] / 1000 * (h - 1))
        ))
    return pts


def _fallback_hair_polygon(size=(1024, 1024)):
    """
    Conservative fallback if semantic polygon detection fails.
    """
    w, h = size
    return [
        (int(w*.18), int(h*.05)),
        (int(w*.35), int(h*.01)),
        (int(w*.65), int(h*.01)),
        (int(w*.82), int(h*.06)),
        (int(w*.90), int(h*.25)),
        (int(w*.83), int(h*.48)),
        (int(w*.72), int(h*.58)),
        (int(w*.62), int(h*.45)),
        (int(w*.38), int(h*.45)),
        (int(w*.28), int(h*.58)),
        (int(w*.17), int(h*.48)),
        (int(w*.10), int(h*.25)),
    ]




def _style_mask_profile(style_name: str):
    """
    v0.14: controlled expansion of the detected hair mask.

    Instead of pushing polygon vertices outward (which can swallow chunks of
    background), rasterize the real detected hair region and dilate it by a
    limited number of pixels. This keeps the editable area close to the hair.
    """
    name = style_name.lower()

    profile = {
        "dilate_px": 14,
        "extra_top_px": 10,
        "protect_face_top": 0.34,
        "protect_face_bottom": 0.91,
    }

    short_keywords = [
        "crew cut", "buzz", "caesar", "crop", "fade",
        "short back", "taper", "ivy league"
    ]
    long_keywords = [
        "bro flow", "shoulder", "long center", "wolf",
        "shag", "curtain", "mullet", "undercut"
    ]
    volume_keywords = ["quiff", "pompadour", "slick back", "textured quiff"]

    if any(k in name for k in short_keywords):
        # Short cuts need only a modest ring around current hair to remove
        # length and reshape temples/sides without consuming background.
        profile.update({
            "dilate_px": 20,
            "extra_top_px": 8,
            "protect_face_top": 0.365,
            "protect_face_bottom": 0.92,
        })
    elif any(k in name for k in long_keywords):
        profile.update({
            "dilate_px": 34,
            "extra_top_px": 28,
            "protect_face_top": 0.33,
            "protect_face_bottom": 0.91,
        })
    elif any(k in name for k in volume_keywords):
        profile.update({
            "dilate_px": 28,
            "extra_top_px": 34,
            "protect_face_top": 0.35,
            "protect_face_bottom": 0.91,
        })

    return profile


def _semantic_hair_alpha(size, polygon, style_name):
    """
    v0.16 dual-zone mask.

    Zone 1: the detected hair + modest dilation can be regenerated freely.
    Zone 2: a wider, feathered transition ring allows a natural silhouette,
            but the final composite fades rapidly back to the exact original.
    Face remains protected.
    """
    from PIL import ImageChops

    w, h = size
    profile = _style_mask_profile(style_name)

    base = Image.new("L", size, 0)
    ImageDraw.Draw(base).polygon(polygon, fill=255)

    # Inner zone: enough freedom to actually reshape the haircut.
    name = style_name.lower()
    if any(k in name for k in ["crew cut", "buzz", "caesar", "crop", "fade", "taper", "ivy league"]):
        inner_radius = 24
        outer_radius = 42
        top_room = 18
    elif any(k in name for k in ["bro flow", "shoulder", "long center", "wolf", "shag", "curtain", "mullet", "undercut"]):
        inner_radius = 38
        outer_radius = 68
        top_room = 42
    else:
        inner_radius = 30
        outer_radius = 54
        top_room = 34

    def dilate(mask, radius):
        # Pillow MaxFilter has a practical kernel limit; keep it odd.
        kernel = min(99, radius * 2 + 1)
        if kernel % 2 == 0:
            kernel -= 1
        return mask.filter(ImageFilter.MaxFilter(max(3, kernel)))

    inner = dilate(base, inner_radius)
    outer = dilate(base, outer_radius)

    # Extra vertical room only directly above the detected hair.
    bbox = base.getbbox()
    if bbox:
        l, t, r, b = bbox
        extra = Image.new("L", size, 0)
        ed = ImageDraw.Draw(extra)
        side_pad = max(8, int((r - l) * 0.04))
        ed.rectangle(
            (
                max(0, l - side_pad),
                max(0, t - top_room),
                min(w - 1, r + side_pad),
                min(h - 1, t + max(8, top_room // 3)),
            ),
            fill=255,
        )
        inner = ImageChops.lighter(inner, extra)
        outer = ImageChops.lighter(outer, extra)

    # Keep both zones near the detected hairstyle/head.
    if bbox:
        l, t, r, b = bbox
        clamp = Image.new("L", size, 0)
        cd = ImageDraw.Draw(clamp)
        pad_x = max(24, int((r - l) * 0.18))
        pad_bottom = max(16, int((b - t) * 0.10))
        cd.rectangle(
            (
                max(0, l - pad_x),
                max(0, t - top_room - 8),
                min(w - 1, r + pad_x),
                min(h - 1, b + pad_bottom),
            ),
            fill=255,
        )
        inner = ImageChops.darker(inner, clamp)
        outer = ImageChops.darker(outer, clamp)

    # Protect most of the face, but don't cut the hairline with a hard ellipse.
    face_protect = Image.new("L", size, 0)
    fd = ImageDraw.Draw(face_protect)
    fd.ellipse(
        (
            int(w * 0.255),
            int(h * 0.405),
            int(w * 0.745),
            int(h * 0.915),
        ),
        fill=255,
    )
    inner = ImageChops.subtract(inner, face_protect)
    outer = ImageChops.subtract(outer, face_protect)

    # API edit mask: transparent throughout the outer zone so the generator
    # has enough context to form a natural edge.
    api_zone = outer.filter(ImageFilter.GaussianBlur(radius=2.0))
    edit_alpha = Image.eval(api_zone, lambda p: 255 - p)
    edit_mask = Image.new("RGBA", size, (255, 255, 255, 255))
    edit_mask.putalpha(edit_alpha)

    # v0.19 Tight Contour Lock:
    # Keep AI contribution strong inside the hairstyle, but constrain the
    # transition ring to just a few pixels around the detected hair contour.
    inner_soft = inner.filter(ImageFilter.GaussianBlur(radius=1.0))

    # Build a much narrower shell around the inner mask.
    narrow_outer = dilate(inner, 10)
    narrow_outer = narrow_outer.filter(ImageFilter.GaussianBlur(radius=2.8))
    ring = ImageChops.subtract(narrow_outer, inner_soft)
    ring = ring.point(lambda p: int(p * 0.12))

    composite_alpha = ImageChops.lighter(inner_soft, ring)

    # v0.20 smooth contour lock:
    # keep accepted AI pixels near the actual detected hairstyle contour.
    # This avoids the straight rectangular seam seen in v0.19.
    contour_guard = dilate(base, 12).filter(ImageFilter.GaussianBlur(radius=3.5))
    composite_alpha = ImageChops.darker(composite_alpha, contour_guard)

    # Extra safety: never accept generated pixels in the central face.
    hard_face = Image.new("L", size, 0)
    hfd = ImageDraw.Draw(hard_face)
    hfd.ellipse(
        (
            int(w * 0.245),
            int(h * 0.385),
            int(w * 0.755),
            int(h * 0.925),
        ),
        fill=255,
    )
    composite_alpha = ImageChops.subtract(composite_alpha, hard_face)

    return edit_mask, composite_alpha



def generate_tryon(style_name, quality="medium", progress=None):
    """
    v0.16 Dual-Zone Hair Mask

    1. Crop only the head area.
    2. Vision model estimates the actual hairstyle polygon.
    3. Image Edit may regenerate only that polygon.
    4. After generation, ONLY polygon pixels are accepted from AI.
    5. Face and every pixel outside the hairstyle polygon come from the
       original crop, then the crop is pasted back at the exact coordinates.

    This is designed specifically to prevent background regeneration.
    """
    if not st.session_state.photo_bytes:
        return None

    client = OpenAI()
    image_path = None
    mask_path = None

    try:
        from io import BytesIO

        t_load = time.perf_counter() if "time" in globals() else __import__("time").perf_counter()
        source = Image.open(BytesIO(st.session_state.photo_bytes))
        source = ImageOps.exif_transpose(source).convert("RGB")

        crop_box = _local_head_crop_box(source)
        original_crop = source.crop(crop_box).convert("RGB")

        work_size = (1024, 1024)
        work_crop = original_crop.resize(work_size, Image.Resampling.LANCZOS)

        import time
        timings = {}
        t0 = time.perf_counter()
        def step(msg):
            if progress:
                progress(msg)

        step("1/7 Открываю и уменьшаю фото…")
        timings["load_resize"] = time.perf_counter() - t0
        step("2/7 Проверяю кэш анализа волос…")
        polygon = st.session_state.get("hair_polygon_cache")
        if polygon:
            timings["hair_analysis"] = 0.0
            timings["hair_analysis_cached"] = True
        else:
            step("3/7 Определяю контур волос через Vision API…")
            ta = time.perf_counter()
            try:
                polygon = _detect_hair_polygon(work_crop)
            except Exception:
                polygon = None
            if not polygon:
                polygon = _fallback_hair_polygon(work_size)
            st.session_state.hair_polygon_cache = polygon
            timings["hair_analysis"] = time.perf_counter() - ta
            timings["hair_analysis_cached"] = False

        step("4/7 Создаю маску волос…")
        tm = time.perf_counter()
        edit_mask, composite_alpha = _semantic_hair_alpha(work_size, polygon, style_name)
        timings["mask"] = time.perf_counter() - tm

        step("5/7 Сохраняю временные файлы…")
        tf = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            image_path = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            mask_path = f.name

        work_crop.save(image_path, format="PNG")
        edit_mask.save(mask_path, format="PNG")
        timings["temp_files"] = time.perf_counter() - tf

        step("6/7 Отправил в OpenAI — жду генерацию изображения…")
        tg = time.perf_counter()
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            r = client.images.edit(
                model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
                image=image_file,
                mask=mask_file,
                prompt=(
                    f"Transform only the editable hairstyle area into a clearly recognizable, photorealistic {style_name}. "
                    "The result must be structurally faithful to the named haircut, not merely a subtle variation. "
                    "Use the full editable hairstyle region when needed: change top length, side length, temples, crown, fringe, "
                    "volume and outer silhouette according to the chosen haircut. "
                    "For Crew Cut/Buzz/Crop/Fade families, visibly shorten the top and especially the sides/temples. "
                    "For longer styles, extend the hairstyle naturally within the editable region without changing the face. "
                    "Preserve natural hair color, believable density, visible scalp and realistic hairline. "
                    "ABSOLUTE PRESERVATION RULE: change hair only. Keep the person's face, head position, expression, "
                    "glasses, eyebrows, eyes, skin, ears, teeth, neck and clothing exactly as in the source. "
                    "Keep the background, people, objects, lighting, camera perspective and framing pixel-identical outside the hairstyle edge. "
                    "Do not beautify, retouch, reshape or reinterpret any non-hair pixels. "
                    "The result must look like the exact same photograph with only the haircut changed."
                ),
                size="1024x1024",
                quality=quality
            )
        timings["generation"] = time.perf_counter() - tg
        step(f"6/7 OpenAI ответил за {timings['generation']:.1f} с. Декодирую…")

        td = time.perf_counter()
        generated_bytes = base64.b64decode(r.data[0].b64_json)
        generated_crop = Image.open(BytesIO(generated_bytes)).convert("RGB")
        timings["decode"] = time.perf_counter() - td

        step("7/7 Собираю итоговое фото…")
        tp = time.perf_counter()

        # AI contributes ONLY semantic hair-mask pixels.
        composited_work = Image.composite(
            generated_crop,
            work_crop.convert("RGB"),
            composite_alpha
        )

        composited_crop = composited_work.resize(
            original_crop.size,
            Image.Resampling.LANCZOS
        )

        final = source.copy()
        final.paste(composited_crop, (crop_box[0], crop_box[1]))

        out = BytesIO()
        # PNG avoids introducing fresh JPEG artifacts into otherwise unchanged pixels.
        final.save(out, format="PNG", optimize=True)
        timings["processing"] = time.perf_counter() - tp
        timings["total"] = time.perf_counter() - t0
        return out.getvalue(), timings

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

            qcol1,qcol2=st.columns([2,1])
            with qcol1:
                quality_choice=st.radio("Качество", ["Быстро", "HD"], horizontal=True, key="q_"+rec["id"], label_visibility="collapsed")
            with qcol2:
                st.caption("HD медленнее")
            if st.button("✨ Показать на мне",key="try_"+rec["id"],use_container_width=True):
                try:
                    api_quality = "medium" if quality_choice == "Быстро" else "high"
                    with st.status("Создаю визуальную примерку...", expanded=True) as status:
                        live = st.empty()
                        def progress(msg):
                            live.write(msg)
                        result=generate_tryon(rec["name"], api_quality, progress=progress)
                        out, timings = result
                        status.update(label="Готово", state="complete", expanded=False)
                    if out:
                        before_col, after_col = st.columns(2)
                        with before_col:
                            st.image(st.session_state.photo_bytes, caption="До", use_container_width=True)
                        with after_col:
                            st.image(out, caption=f"После — {rec['name']}", use_container_width=True)
                        cached = "из кэша" if timings.get("hair_analysis_cached") else f"{timings.get('hair_analysis',0):.1f} с"
                        st.caption(
                            f"⏱ Фото/resize: {timings.get('load_resize',0):.1f} с · Маска: {timings.get('mask',0):.1f} с · Temp: {timings.get('temp_files',0):.1f} с · "
                            f"Анализ волос: {cached} · "
                            f"Генерация API: {timings.get('generation',0):.1f} с · "
                            f"Декодирование: {timings.get('decode',0):.1f} с · "
                            f"Обработка: {timings.get('processing',0):.1f} с · "
                            f"Всего: {timings.get('total',0):.1f} с"
                        )
                        st.caption("AI-визуализация — ориентир. v0.17 кэширует анализ волос; быстрый режим использует medium, HD — high.")
                except Exception as e:
                    st.error(f"Не удалось создать примерку: {e}")
            st.divider()

        if st.button("Пересчитать с учётом 👍/👎",use_container_width=True):
            p=dict(st.session_state.last_profile)
            p["history"]={"liked":st.session_state.liked,"disliked":st.session_state.disliked}
            r["recommendations"]=rank(p,5); st.session_state.last_result=r; st.rerun()

st.caption("v0.21 — Live Diagnostic Timing: показывает текущий этап и отдельное время Vision API, маски, временных файлов, Image API и финальной обработки.")
