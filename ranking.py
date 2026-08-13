import json
from pathlib import Path

HAIRSTYLES = json.loads(Path(__file__).with_name("hairstyles.json").read_text(encoding="utf-8"))

LENGTH = {"very_short":0,"short":1,"short_medium":2,"medium":3,"medium_long":4,"long":5}
STYLING = {"minimal":0,"low":1,"medium":2,"medium_high":3,"high":4}

def rejected(style, user):
    avoid=set(user.get("avoid",[]))
    if "bangs" in avoid and style.get("fringe") is True: return True
    if "long_hair" in avoid and LENGTH.get(style["top_length"],3)>=4: return True
    if "short_hair" in avoid and LENGTH.get(style["top_length"],3)<=1: return True
    return False

def score(style,user):
    if rejected(style,user): return None
    points=0.0; reasons=[]; warnings=[]
    hair=user.get("hair",{}); hc=float(hair.get("confidence",.5))
    if hair.get("texture") in style["texture"]:
        points+=12.5; reasons.append("подходит по текстуре волос")
    else: points+=12.5*(1-hc); warnings.append("текстура может быть ограничением")
    if hair.get("density") in style["density"]:
        points+=12.5; reasons.append("подходит по густоте")
    else: points+=12.5*(1-hc); warnings.append("густота может быть ограничением")

    face=user.get("face",{}); fc=float(face.get("confidence",.5))
    preferred={
      "oval":["crew_cut","textured_crop","ivy_league","side_part","taper"],
      "oval_rectangular":["crew_cut","textured_crop","ivy_league","side_part","taper","quiff"],
      "round":["textured_crop","quiff","high_fade","mid_fade","side_part"],
      "square":["buzz_cut","crew_cut","textured_crop","quiff","taper","slick_back"]
    }.get(face.get("shape"),[])
    if style["id"] in preferred:
        points+=20*fc; reasons.append("эвристически сочетается с формой лица")
    else: points+=8*fc

    headc=float(user.get("head_shape",{}).get("confidence",0))
    if headc>=.7:
        points+=12; reasons.append("форма головы учтена")
    else:
        points+=10; warnings.append("форма головы пока недостаточно уверенно определена")

    target={"short":1,"medium":3,"long":5}.get(user.get("desired_length"),3)
    dist=abs(LENGTH.get(style["top_length"],3)-target)
    points+=15*max(0,1-dist/4)
    if dist==0: reasons.append("соответствует желаемой длине")
    elif dist>1: warnings.append("длина отличается от желаемой")

    allowed=STYLING.get(user.get("max_styling","medium"),2)
    needed=STYLING.get(style.get("styling","medium"),2)
    if needed<=allowed:
        points+=10; reasons.append("подходит по требованиям к укладке")
    else:
        points+=10*max(0,1-(needed-allowed)/4); warnings.append("потребует больше укладки")

    hist=user.get("history",{})
    if style["id"] in hist.get("liked",[]): points+=10
    elif style["id"] in hist.get("disliked",[]): points-=10
    else: points+=5
    return {"id":style["id"],"name":style["name"],"score":round(max(0,min(100,points)),1),
            "reasons":reasons,"warnings":warnings}

def rank(user,top_n=5):
    results=[score(s,user) for s in HAIRSTYLES if not rejected(s,user)]
    results=[r for r in results if r]
    return sorted(results,key=lambda x:x["score"],reverse=True)[:top_n]

if __name__=="__main__":
    profiles=json.loads(Path(__file__).with_name("test_profiles.json").read_text(encoding="utf-8"))
    for label,user in profiles.items():
        print("\n===",label,"===")
        for r in rank(user):
            print(r["score"],r["name"])
            for x in r["reasons"]: print("  +",x)
            for x in r["warnings"]: print("  !",x)
