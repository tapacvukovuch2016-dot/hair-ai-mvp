import os, tempfile, json, base64

IDENTITY_LOCK = """
STRICT IDENTITY LOCK:
- Change ONLY the hairstyle/hair pixels needed for the requested haircut.
- Preserve the person's face exactly: eyes, eyelids, eyebrows, nose, mouth, teeth, cheeks, jaw, ears, skin texture and expression.
- Preserve glasses and their exact geometry.
- Preserve the natural forehead size and original hairline position unless the haircut itself strictly requires a tiny edge adjustment.
- Do not enlarge, shrink, round, narrow, or reshape the skull/head.
- Preserve pose, camera perspective, lighting, body, tattoo, clothes, background and every non-hair object.
- The result must look like the same photograph with only a real barber haircut applied.\n- Never create halos, arcs, circular patches, translucent wedges, duplicated background, or seams around the head.\n- Never redraw background above or beside the hairstyle.\n- Hairline and outer haircut silhouette must be irregular and photorealistic, with fine individual hairs; avoid perfectly geometric or helmet-like edges.
- Preserve the person's natural frontal hairline shape and subtle asymmetry; do not draw a ruler-straight horizontal edge.
- At the frontal hairline, include tiny irregular individual hairs and slight density variation typical of a real haircut.
- Blend both temples gradually into the haircut; avoid hard vertical corners or stamped edges.
- Crew Cut should contain subtle natural variation in strand length, direction and density rather than a uniform painted texture.
- Keep scalp visibility physically plausible for the person's original hair color, lighting and camera distance.
- Hair density must not be perfectly uniform; include subtle natural variation across the top and crown.
- For short styles like Crew Cut, allow slight realistic scalp visibility between strands where appropriate.
- Vary individual strand direction, spacing and length by small amounts; avoid repeated or stamped texture.
- Preserve the original hair color and lighting response, including natural highlights and shadow between strands.
- Avoid overly dense plush/fuzzy texture, painted texture, synthetic fibers, or a perfectly even carpet-like surface.
- Keep the haircut believable as real human hair photographed by the same camera, not as a generated overlay.
- Avoid helmet-like hair, painted hair, artificial straight borders, duplicated texture, warped temples, or changed facial proportions.
"""

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

for k,v in {"liked":[],"disliked":[],"last_result":None,"last_profile":None,"photo_bytes":None,"hair_polygon_cache":None,"hair_polygon_photo_key":None,"tryon_cache":{}}.items():
    if k not in st.session_state: st.session_state[k]=v

st.title("✂️ Hair AI")
st.caption("MVP v0.36 — Long → Short Hair Test")

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
            st.session_state.tryon_cache = {}
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

    response = client.with_options(timeout=18.0, max_retries=0).responses.create(
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
                {"type": "input_image", "image_url": _crop_to_data_url(work_crop), "detail": "low"}
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
    v0.27 Natural Blend:
    keep the v0.26 hair-only hard guard, but use a softer multi-stage
    transition at the real hair boundary to reduce the pasted-cap effect.
    """
    from PIL import ImageChops

    w, h = size
    base = Image.new("L", size, 0)
    ImageDraw.Draw(base).polygon(polygon, fill=255)

    name = style_name.lower()
    radical_short = any(k in name for k in ["pixie", "crew cut", "buzz", "caesar", "crop", "fade", "taper", "ivy league"])

    if radical_short:
        # v0.36: short transformations must be allowed to erase the ENTIRE
        # detected source hairstyle, including long lengths hanging below the
        # future haircut.  The old version only accepted pixels close to the
        # future short silhouette, which could leave a "second hairstyle"
        # behind the generated pixie.
        api_radius = 42
        core_radius = 28
        feather_radius = 10
    elif any(k in name for k in ["bro flow", "shoulder", "long center", "wolf", "shag", "curtain", "mullet", "undercut", "long bob", "lob", "bob", "long layers", "wavy long hair"]):
        api_radius = 25
        core_radius = 11
        feather_radius = 9
    else:
        api_radius = 21
        core_radius = 9
        feather_radius = 8

    def dilate(mask, radius):
        mw, mh = mask.size
        if mw >= 768 and mh >= 768:
            small = mask.resize((mw // 2, mh // 2), Image.Resampling.NEAREST)
            rr = max(1, int(round(radius / 2)))
            kernel = min(31, rr * 2 + 1)
            if kernel % 2 == 0:
                kernel -= 1
            small = small.filter(ImageFilter.MaxFilter(max(3, kernel)))
            return small.resize((mw, mh), Image.Resampling.BILINEAR)
        kernel = min(63, radius * 2 + 1)
        if kernel % 2 == 0:
            kernel -= 1
        return mask.filter(ImageFilter.MaxFilter(max(3, kernel)))

    face_protect = Image.new("L", size, 0)
    fd = ImageDraw.Draw(face_protect)
    fd.ellipse(
        (int(w * 0.255), int(h * 0.405), int(w * 0.745), int(h * 0.925)),
        fill=255,
    )

    # API edit area remains tightly attached to actual hair.
    api_zone = dilate(base, api_radius)
    api_zone = ImageChops.subtract(api_zone, face_protect)
    api_zone = api_zone.filter(ImageFilter.GaussianBlur(radius=1.6))

    edit_alpha = Image.eval(api_zone, lambda p: 255 - p)
    edit_mask = Image.new("RGBA", size, (255, 255, 255, 255))
    edit_mask.putalpha(edit_alpha)

    # Strong generated result in the central hair region.
    # For radical shortening, the full original detected hair region is part
    # of the core so old long strands can actually disappear in the final
    # composite instead of being restored from the source photo.
    if radical_short:
        core = ImageChops.lighter(base, dilate(base, core_radius))
    else:
        core = dilate(base, core_radius)
    core = ImageChops.subtract(core, face_protect)
    core = core.filter(ImageFilter.GaussianBlur(radius=1.35))

    # Natural transition ring. It is wider than v0.26 but much weaker,
    # so original fine hairs/background remain visible through the edge.
    outer = dilate(base, core_radius + feather_radius)
    outer = ImageChops.subtract(outer, face_protect)
    outer = outer.filter(ImageFilter.GaussianBlur(radius=3.6))

    ring = ImageChops.subtract(outer, core)
    ring = ring.point(lambda p: int(p * 0.48))
    composite_alpha = ImageChops.lighter(core, ring)

    # Preserve a little source texture at the boundary instead of a 100% AI edge.
    base_soft = base.filter(ImageFilter.GaussianBlur(radius=2.0))
    boundary_texture = ImageChops.subtract(
        dilate(base, 4).filter(ImageFilter.GaussianBlur(radius=1.4)),
        base_soft
    )
    boundary_texture = boundary_texture.point(lambda p: int(p * 0.18))
    composite_alpha = ImageChops.subtract(composite_alpha, boundary_texture)

    # Critical v0.26 protection: absolutely no AI pixels far from the hair.
    hard_guard = dilate(base, api_radius + 4).filter(ImageFilter.GaussianBlur(radius=0.9))
    composite_alpha = ImageChops.darker(composite_alpha, hard_guard)
    composite_alpha = ImageChops.subtract(composite_alpha, face_protect)

    return edit_mask, composite_alpha




def _hairstyle_identity_prompt(style_name: str) -> str:
    """v0.34: make similarly short hairstyles visibly distinct without weakening identity lock."""
    name = style_name.lower()
    if "crew cut" in name:
        return (
            "HAIRSTYLE IDENTITY — CREW CUT: unmistakable classic crew cut. "
            "Top must be very short, approximately 12-22 mm, slightly longer at the front and gradually shorter toward the crown. "
            "Sides and temples must be clearly shorter than the top, approximately 3-9 mm, with a soft natural taper. "
            "No fringe, no textured crop fringe, no long swept top, no Caesar shape. "
            "The silhouette must be compact and close to the skull while preserving the person's original natural hairline. "
        )
    if "textured crop" in name or ("crop" in name and "crew" not in name):
        return (
            "HAIRSTYLE IDENTITY — TEXTURED CROP: unmistakable textured crop, clearly different from a crew cut. "
            "Keep 30-50 mm of visibly textured, piecey hair on top with irregular forward-directed sections. "
            "Create a short, soft, broken natural fringe over the upper forehead; it must not be a ruler-straight line. "
            "Sides should be short with a visible low-to-mid fade/taper, clearly tighter than the textured top. "
            "Do not turn this into a uniform crew cut, buzz cut, or smooth side-swept taper. "
        )
    if "classic taper" in name or "taper" in name:
        return (
            "HAIRSTYLE IDENTITY — CLASSIC TAPER: unmistakable classic taper, clearly different from a textured crop and crew cut. "
            "Keep substantially more length on top, approximately 45-70 mm, with a neat natural side-swept or softly combed shape and moderate volume. "
            "Only the sideburns and lower temple/ear area should gradually taper shorter; do NOT create a high fade or expose a large shaved side area. "
            "No short crop fringe, no buzzed top, no uniform short texture. The top must visibly remain longer than in Crew Cut or Textured Crop. "
        )
    if "buzz" in name:
        return (
            "HAIRSTYLE IDENTITY — BUZZ CUT: nearly uniform very short hair, approximately 3-9 mm, with only subtle natural length variation. "
            "It must be visibly shorter and more uniform than a Crew Cut. Preserve the natural source hairline; no lineup. "
        )
    if "pixie" in name:
        return (
            "HAIRSTYLE IDENTITY — PIXIE CUT: unmistakable feminine pixie cut. "
            "Keep the sides and nape short and neat, with visibly longer soft texture on top, approximately 25-55 mm. "
            "Use a light piecey fringe or side-swept front depending on the natural hairline. "
            "Do not turn it into a crew cut, buzz cut, or masculine barber fade. Keep the silhouette soft and salon-like. "
        )
    if "long bob" in name or "lob" in name:
        return (
            "HAIRSTYLE IDENTITY — LONG BOB / LOB: unmistakable long bob ending around the collarbone or just above the shoulders. "
            "Create a clean but natural perimeter with subtle movement, slightly longer front sections if flattering, and moderate volume. "
            "Do not make it a short chin bob, pixie, or long layered haircut. "
        )
    if "bob" in name:
        return (
            "HAIRSTYLE IDENTITY — BOB: unmistakable classic bob ending around chin to jaw level. "
            "Create a rounded salon silhouette with natural movement and a clean but not ruler-straight perimeter. "
            "Keep both sides balanced around the face and do not extend to the shoulders. "
        )
    if "curtain bangs" in name:
        return (
            "HAIRSTYLE IDENTITY — CURTAIN BANGS: keep the overall hair length mostly similar, but create clear center-parted curtain bangs. "
            "The fringe must split softly in the middle and sweep outward along both cheek/temple areas, with face-framing pieces. "
            "Do not make blunt straight bangs or a short crop fringe. "
        )
    if "long layers" in name:
        return (
            "HAIRSTYLE IDENTITY — LONG LAYERS: clearly long hair with visible graduated layers and face-framing sections. "
            "Maintain substantial length while adding movement, lighter ends, and natural volume. "
            "Do not turn it into a bob, lob, or uniform one-length sheet of hair. "
        )
    if "wolf cut" in name:
        return (
            "HAIRSTYLE IDENTITY — WOLF CUT: unmistakable wolf cut with layered volume around the crown, textured face-framing pieces, "
            "and visibly lighter, longer ends toward the back. The shape should feel shaggy and modern, not like a bob or generic long hair. "
        )
    if "shag" in name:
        return (
            "HAIRSTYLE IDENTITY — SHAG: unmistakable layered shag with airy crown volume, irregular textured layers, and face-framing movement. "
            "Keep the perimeter soft and lived-in; avoid a polished bob or uniform long layers. "
        )
    if "wavy long hair" in name:
        return (
            "HAIRSTYLE IDENTITY — WAVY LONG HAIR: clearly long hair extending toward or below the shoulders with natural loose waves. "
            "Use irregular S-shaped bends, realistic strand grouping, and non-uniform volume. Avoid perfect repeated curls or a wig-like surface. "
        )
    return (
        f"HAIRSTYLE IDENTITY: reproduce the defining geometry of a real {style_name}; make its top length, side length, direction, volume, fringe and silhouette visibly characteristic of that named haircut. "
        "Do not collapse it into a generic short haircut. "
    )

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

    # Same photo + same hairstyle + same quality = no reason to pay/wait twice.
    cache_key = (
        st.session_state.get("hair_polygon_photo_key"),
        style_name,
        quality,
        "v0.36-long-to-short",
    )
    cached = st.session_state.get("tryon_cache", {}).get(cache_key)
    if cached:
        cached_bytes, cached_timings = cached
        timings = dict(cached_timings)
        timings["result_cached"] = True
        timings["total"] = 0.0
        if progress:
            progress("⚡ Готовая примерка найдена в кэше.")
        return cached_bytes, timings

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

        work_crop.save(image_path, format="PNG", compress_level=1)
        edit_mask.save(mask_path, format="PNG", compress_level=1)
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
                    + _hairstyle_identity_prompt(style_name) +
                    "The result must be structurally faithful to the named haircut, not merely a subtle variation. "
                    "Use the full editable hairstyle region when needed: change top length, side length, temples, crown, fringe, "
                    "volume and outer silhouette according to the chosen haircut. "
                    "For Crew Cut/Buzz/Crop/Fade families, visibly shorten the top and especially the sides/temples. "
                    "For longer styles, extend the hairstyle naturally within the editable region without changing the face. "
                    "Preserve natural hair color, believable density, visible scalp and realistic hairline. "
                    "HAIRLINE PRESERVE — HIGHEST PRIORITY: treat the visible source frontal hairline and both temple boundaries as fixed identity geometry, not as a design target. "
                    "Do NOT redesign, straighten, square, sharpen, lower, raise, widen, narrow, symmetrize, or complete the hairline. "
                    "Shorten the existing hair behind that boundary while preserving the source boundary's position, curvature, asymmetry, recession and temple shape as closely as the photograph allows. "
                    "If any part of the true hairline is ambiguous or hidden by longer source hair, DO NOT guess a new edge: retain the closest plausible source transition and keep it soft and irregular. "
                    "Never invent a new corner, diagonal cut, notch, step, rectangular temple, sharp 90-degree angle, lineup, edge-up, or geometric forehead border. "
                    "For Crew Cut specifically, do not create a barber lineup. The front edge must look like the same person's naturally occurring hairline after the hair was shortened. "
                    "At the frontal boundary and temples, preserve subtle source asymmetry and use fine individual hairs, broken density and tiny natural gaps only where consistent with the source. "
                    "Do not make either temple match the other artificially. Do not extend generated hair onto previously exposed forehead skin merely to make the haircut look cleaner. "
                    "When uncertain, preserve source forehead pixels rather than generating a new hairline. No fuzzy halo, no excessive flyaways, and no artificial dark outline. "
                    "LONG-TO-SHORT TRANSFORMATION — CRITICAL: if the requested hairstyle is substantially shorter than the source hair, treat this as a real haircut, not as a new hairstyle layered on top of the old one. "
                    "Remove ALL source hair that would have been physically cut off: long side lengths, hair behind the ears, hair on the neck, shoulder-length strands, rear hanging sections, and every visible remnant outside the final short-hair silhouette. "
                    "There must be exactly ONE hairstyle in the result. Never leave the original long hair behind or underneath a Pixie, Crop, Crew Cut, Buzz Cut, Fade, Taper, or other short style. "
                    "Where removed long hair previously covered the scene, plausibly reconstruct the newly exposed local background, ear edges, neck, shoulder or clothing using the surrounding source photograph. "
                    "Do not preserve old hair merely because the pixels existed in the source. In a long-to-short haircut those pixels are intentionally removed. "
                    "CLEAN SILHOUETTE: for a shorter haircut, remove every visible remnant of the source longer hairstyle inside the editable outer hair zone. "
                    "Do not leave arcs, wisps, bands, duplicate hair, ghost hair, or the old hairstyle silhouette above, beside, or behind the new haircut. "
                    "Where old hair extended beyond the new haircut, reconstruct only the immediately adjacent original background naturally; never create a halo or circular patch. "
                    "Keep the forehead shape and exposed forehead area unchanged; shortening hair must not move the hairline forward or backward. "
                    + IDENTITY_LOCK +
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
        # Lossless, but much faster than optimize=True.
        final.save(out, format="PNG", compress_level=1)
        timings["processing"] = time.perf_counter() - tp
        timings["total"] = time.perf_counter() - t0
        timings["result_cached"] = False

        final_bytes = out.getvalue()
        st.session_state.tryon_cache[cache_key] = (final_bytes, dict(timings))
        return final_bytes, timings

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
                quality_choice=st.radio("Качество", ["⚡ Черновик", "Быстро", "HD"], horizontal=True, index=1, key="q_"+rec["id"], label_visibility="collapsed")
            with qcol2:
                st.caption("⚡ Черновик — максимум скорости · Быстро — баланс · HD — максимум качества")
            if st.button("✨ Показать на мне",key="try_"+rec["id"],use_container_width=True):
                try:
                    if quality_choice == "⚡ Черновик":
                        api_quality = "low"
                    elif quality_choice == "Быстро":
                        api_quality = "medium"
                    else:
                        api_quality = "high"
                    with st.status("Создаю визуальную примерку...", expanded=True) as status:
                        live = st.empty()
                        def progress(msg):
                            live.write(msg)
                        result=generate_tryon(rec["name"], api_quality, progress=progress)
                        out, timings = result
                        status.update(label="Готово", state="complete", expanded=False)
                    if out:
                        if timings.get("result_cached"):
                            st.success("⚡ Повторная примерка загружена мгновенно из кэша.")
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
                        st.caption("v0.36: Long → Short — старая длина должна полностью удаляться, а открывшиеся участки локально восстанавливаться.")
                except Exception as e:
                    st.error(f"Не удалось создать примерку: {e}")
            st.divider()

        if st.button("Пересчитать с учётом 👍/👎",use_container_width=True):
            p=dict(st.session_state.last_profile)
            p["history"]={"liked":st.session_state.liked,"disliked":st.session_state.disliked}
            r["recommendations"]=rank(p,5); st.session_state.last_result=r; st.rerun()

        st.subheader("5. Женские причёски — тестовый каталог")
        st.caption("Можно примерять на любом человеке. Это отдельный тестовый каталог и пока не участвует в AI-рейтинге рекомендаций.")

        WOMENS_STYLES = [
            {"id":"w_pixie","name":"Pixie Cut","summary":"Короткая салонная стрижка с мягкой текстурой сверху и аккуратными боками."},
            {"id":"w_bob","name":"Classic Bob","summary":"Классический боб примерно до подбородка/линии челюсти."},
            {"id":"w_lob","name":"Long Bob (Lob)","summary":"Удлинённый боб примерно до ключиц или чуть выше плеч."},
            {"id":"w_curtain","name":"Curtain Bangs","summary":"Мягкая центральная чёлка-шторка с прядями, обрамляющими лицо."},
            {"id":"w_layers","name":"Long Layers","summary":"Длинные волосы с выраженными слоями и движением по длине."},
            {"id":"w_wolf","name":"Wolf Cut","summary":"Объёмная макушка, выраженная слоистость и более лёгкие длинные концы."},
            {"id":"w_shag","name":"Shag","summary":"Воздушная многослойная стрижка с небрежной текстурой."},
            {"id":"w_waves","name":"Wavy Long Hair","summary":"Длинные волосы с естественными свободными волнами."},
        ]

        for ws in WOMENS_STYLES:
            with st.expander(f"💇‍♀️ {ws['name']}"):
                st.write(ws["summary"])
                wq = st.radio(
                    "Качество",
                    ["⚡ Черновик", "Быстро", "HD"],
                    horizontal=True,
                    index=1,
                    key="wq_"+ws["id"],
                    label_visibility="collapsed",
                )
                if st.button("✨ Показать на мне", key="wtry_"+ws["id"], use_container_width=True):
                    try:
                        if wq == "⚡ Черновик":
                            api_quality = "low"
                        elif wq == "Быстро":
                            api_quality = "medium"
                        else:
                            api_quality = "high"

                        with st.status(f"Создаю {ws['name']}...", expanded=True) as status:
                            live = st.empty()
                            def wprogress(msg):
                                live.write(msg)
                            out, timings = generate_tryon(ws["name"], api_quality, progress=wprogress)
                            status.update(label="Готово", state="complete", expanded=False)

                        before_col, after_col = st.columns(2)
                        with before_col:
                            st.image(st.session_state.photo_bytes, caption="До", use_container_width=True)
                        with after_col:
                            st.image(out, caption=f"После — {ws['name']}", use_container_width=True)

                        cached = "из кэша" if timings.get("hair_analysis_cached") else f"{timings.get('hair_analysis',0):.1f} с"
                        st.caption(
                            f"⏱ Анализ волос: {cached} · Генерация API: {timings.get('generation',0):.1f} с · "
                            f"Обработка: {timings.get('processing',0):.1f} с · Всего: {timings.get('total',0):.1f} с"
                        )
                    except Exception as e:
                        st.error(f"Не удалось создать примерку: {e}")

st.caption("v0.36 — Long → Short Test: усилено полное удаление исходных длинных волос при короткой стрижке.")
