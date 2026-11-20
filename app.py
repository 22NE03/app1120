# 출튀 감지기
# 강의실 사진 여러 장(예: 10분마다 1장) → YOLOv8 머리 탐지 → 사진별 인원 수 → 인원 변화 확인
#
# 왜 "머리"를 세나?
#  - 위에서 찍은 CCTV에서는 몸이 책상·앞사람에 가려도 머리는 거의 항상 보임
#  - 고개를 숙이거나 엎드려도 정수리는 보임 (얼굴 탐지는 이런 경우를 놓침)
#  - 한 사람당 머리는 하나라서 같은 사람을 두 번 세는 일이 적음
#  - 누구인지 알아보는 정보(얼굴)는 쓰지 않음
#
# 사용자가 AI 결과를 고칠 수 있음
#  - "머리 아님": 가방 같은 오탐 박스를 사진에서 누르면 그 자리를 모든 회차에서 제외 (CCTV는 고정이니까)
#  - 제외한 자리에 학생이 앉으면 그 자리만 다시 셀 수 있게 해제
#  - "이건 머리야": 놓친 머리를 사진에서 눌러 추가 (그 회차 사진에만 적용)
#  - 수정 모드를 하나 골라 고치고 "완료" → 다시 다른 모드 선택
#  - 다 고친 회차는 "확정" → 그 회차 인원이 고정되고 더 이상 바뀌지 않음
#
# 사용 모델: SCUT-HEAD(강의실 CCTV 포함) 데이터로 학습된 YOLOv8n 머리 탐지 모델 (ONNX 형식)

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFile, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates
from torchvision.ops import box_iou, nms
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 끝부분이 살짝 깨진 사진도 열 수 있게

st.set_page_config(page_title="출튀를 잡아라", page_icon="🎓", layout="centered")

MODEL_PATH = "models/head_yolov8n_scut.onnx"
CONF = 0.25          # 이 점수 이상인 박스만 머리로 인정 (카메라 4대 사진 40장으로 비교해서 고른 값)
NMS_IOU = 0.3        # 이만큼 겹치는 박스는 같은 머리로 보고 하나만 남김
GRID = (3, 2)        # 사진을 가로 3 × 세로 2 조각으로 나눠서도 탐지
EXCLUDE_IOU = 0.3    # 제외한 자리와 이만큼 겹치는 박스는 세지 않음

GREEN, BLUE, RED, ORANGE = (0, 255, 120), (0, 160, 255), (255, 60, 60), (255, 150, 0)


# ---------------------------------------------------------------- 모델·탐지 함수
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH, task="detect")


def make_regions(width, height):
    """탐지할 영역 목록: 사진 전체 + 서로 조금씩 겹치는 조각들"""
    regions = [(0, 0, width, height)]
    cols, rows = GRID
    tw, th = int(width / cols * 1.25), int(height / rows * 1.25)  # 25% 크게 → 경계에서 겹침
    for i in range(cols):
        for j in range(rows):
            x = int(i * (width - tw) / (cols - 1))
            y = int(j * (height - th) / (rows - 1))
            regions.append((x, y, x + tw, y + th))
    return regions


def detect_heads(model, image):
    """머리 박스 [x1, y1, x2, y2] 배열을 돌려줌"""
    all_boxes, all_scores = [], []
    for x0, y0, x1, y1 in make_regions(*image.size):
        piece = np.array(image.crop((x0, y0, x1, y1)))
        result = model.predict(piece, conf=CONF, imgsz=640, verbose=False)[0]
        boxes = result.boxes.xyxy.numpy().copy()
        boxes[:, [0, 2]] += x0  # 조각 좌표 → 원본 사진 좌표로 옮기기
        boxes[:, [1, 3]] += y0
        all_boxes.append(boxes)
        all_scores.append(result.boxes.conf.numpy())

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    # 여러 조각에서 같은 머리가 중복으로 잡힌 박스 정리
    keep = nms(torch.tensor(boxes), torch.tensor(scores), NMS_IOU).numpy()
    return boxes[keep]


def shot_time(name):
    """파일 이름 "날짜_시간.확장자"에서 촬영 시각 읽기 (예: 20261120_145600.jpg, 2026-11-20_14-56-00.jpg)
    형식이 다르면 None"""
    digits = re.sub(r"\D", "", name.rsplit(".", 1)[0])  # 숫자만 남기기
    for fmt, size in (("%Y%m%d%H%M%S", 14), ("%Y%m%d%H%M", 12)):
        if len(digits) == size:
            try:
                return datetime.strptime(digits, fmt)
            except ValueError:
                return None
    return None


def natural_key(name):
    """img2.jpg가 img10.jpg보다 앞에 오도록 숫자 기준 정렬"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


# ---------------------------------------------------------------- 저장 공간 (세션)
# 사진과 정정 내용은 이 브라우저 세션에만 잠깐 보관되고, 서버에 파일로 저장하지 않음
defaults = {
    "photos": [],         # [{"id", "name", "image", "boxes", "manual", "frozen"}]
    "seen_ids": set(),    # 한 번 처리한 사진 (개별 삭제 후 다시 추가되지 않게)
    "excluded": [],       # 제외한 자리 [x1, y1, x2, y2] 목록 (모든 회차에 적용)
    "last_click": None,   # 이미 처리한 사진 터치 (같은 터치를 두 번 추가하지 않게)
    "edit_mode": None,    # 지금 수정 모드: None(선택 전) / "add" / "exclude" / "release"
    "picked": set(),      # 수정 모드에서 눌러서 고른 박스 번호 (완료를 누르면 적용)
    "uploader_key": 0,    # 값을 바꾸면 업로드 칸이 비워짐
}
for name, value in defaults.items():
    if name not in st.session_state:
        st.session_state[name] = value


def counted_boxes(photo):
    """AI가 찾은 박스 중 제외한 자리와 겹치지 않는 것만 (확정한 회차는 확정할 때 결과 그대로)"""
    if photo["frozen"] is not None:
        return photo["frozen"]
    boxes = photo["boxes"]
    if len(boxes) == 0 or not st.session_state.excluded:
        return boxes
    overlap = box_iou(torch.tensor(boxes), torch.tensor(np.array(st.session_state.excluded, dtype=np.float32)))
    return boxes[(overlap < EXCLUDE_IOU).all(dim=1).numpy()]


def head_count(photo):
    return len(counted_boxes(photo)) + len(photo["manual"])


def draw_result(photo, picked_heads=(), picked_excluded=()):
    """AI 머리(초록·번호), 직접 추가한 머리(파랑·+번호), 제외한 자리(빨강·X번호) 그리기
    picked_*: 수정 모드에서 눌러서 고른 박스 → 주황색 굵은 박스로 표시"""
    image = photo["image"]
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line = max(2, image.width // 500)
    font = ImageFont.load_default(size=max(16, image.width // 70))

    def box(xyxy, text, color, chosen, text_color=None):
        x1, y1, x2, y2 = xyxy
        draw.rectangle((x1, y1, x2, y2), outline=ORANGE if chosen else color, width=line * 3 if chosen else line)
        draw.text((x1, y1 - font.size - 2), text, fill=ORANGE if chosen else (text_color or color), font=font,
                  stroke_width=3, stroke_fill=(0, 0, 0))

    if photo["frozen"] is None:
        for k, xyxy in enumerate(st.session_state.excluded):
            box(xyxy, f"X{k + 1}", RED, k in picked_excluded)
    for n, xyxy in enumerate(counted_boxes(photo)):
        box(xyxy, str(n + 1), GREEN, n in picked_heads, text_color=(255, 255, 0))
    for m, xyxy in enumerate(photo["manual"]):
        box(xyxy, f"+{m + 1}", BLUE, False)
    return canvas


def find_box(boxes, x, y):
    """누른 위치의 박스 번호(0부터). 박스 안이 아니면 가까운 박스(박스 크기 1.5배 거리 안)를 고름"""
    if len(boxes) == 0:
        return None
    b = np.asarray(boxes, dtype=float)
    inside = np.where((b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3]))[0]
    centers = (b[:, :2] + b[:, 2:]) / 2
    dist = np.hypot(centers[:, 0] - x, centers[:, 1] - y)
    if len(inside):
        return int(inside[np.argmin(dist[inside])])  # 겹친 박스 안이면 중심이 가장 가까운 것
    i = int(np.argmin(dist))
    size = max(b[i, 2] - b[i, 0], b[i, 3] - b[i, 1])
    return i if dist[i] <= size * 1.5 else None


def tap_on_image(image, key):
    """누를 수 있는 사진을 보여 주고, 새로 누른 위치를 원본 사진 좌표로 돌려줌 (없으면 None)"""
    click = streamlit_image_coordinates(image, width="stretch", image_format="JPEG", key=key)
    if not click or click.get("unix_time") == st.session_state.last_click:
        return None
    st.session_state.last_click = click.get("unix_time")
    scale = image.width / click["width"]  # 화면에 보이는 크기 → 원본 사진 좌표
    return click["x"] * scale, click["y"] * scale


def toggle_pick(index):
    """고른 박스를 다시 누르면 선택 취소"""
    if index is not None:
        st.session_state.picked ^= {index}


def head_box_at(photo, x, y):
    """터치한 위치에 머리 박스 만들기 (크기는 근처에서 찾은 머리들의 평균)"""
    boxes = photo["boxes"]
    if len(boxes):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        nearest = np.argsort(np.hypot(centers[:, 0] - x, centers[:, 1] - y))[:5]
        w = float(np.median(boxes[nearest, 2] - boxes[nearest, 0]))
        h = float(np.median(boxes[nearest, 3] - boxes[nearest, 1]))
    else:
        w = h = photo["image"].width * 0.04
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


# ---------------------------------------------------------------- 버튼 동작
def add_photo(file, name=None):
    image = Image.open(file).convert("RGB")
    image.thumbnail((1920, 1920))  # 너무 큰 사진은 줄여서 속도 확보
    st.session_state.seen_ids.add(file.file_id)
    st.session_state.photos.append({
        "id": file.file_id,
        "name": name or file.name,
        "image": image,
        "boxes": detect_heads(model, image),
        "manual": [],
        "frozen": None,  # 확정하면 그때 센 박스를 여기에 고정
    })


def delete_all():
    st.session_state.photos = []
    st.session_state.seen_ids = set()
    st.session_state.excluded = []
    st.session_state.uploader_key += 1  # 업로드 칸과 카메라 칸도 비우기


def delete_photo(photo_id):
    st.session_state.photos = [p for p in st.session_state.photos if p["id"] != photo_id]


def exclude_picked(photo):
    """고른 박스 자리를 제외 목록에 추가 (모든 회차에 적용)"""
    boxes = counted_boxes(photo)
    for n in sorted(st.session_state.picked):
        st.session_state.excluded.append([float(v) for v in boxes[n]])


def release_picked():
    """고른 제외 자리를 해제 (그 자리에 앉은 학생을 다시 셈)"""
    st.session_state.excluded = [b for k, b in enumerate(st.session_state.excluded)
                                 if k not in st.session_state.picked]


def release_all_excluded():
    st.session_state.excluded = []
    st.session_state.picked = set()


def start_mode(mode):
    st.session_state.edit_mode = mode
    st.session_state.picked = set()


def finish_mode(photo):
    """완료: 고른 번호를 적용하고 모드 선택 화면으로 돌아감"""
    mode = st.session_state.edit_mode
    if mode == "exclude":
        exclude_picked(photo)
    elif mode == "release":
        release_picked()
    st.session_state.edit_mode = None
    st.session_state.picked = set()


def clear_picked():
    st.session_state.picked = set()


def confirm_photo(photo):
    """확정: 지금 센 결과를 고정 (나중에 다른 회차에서 제외해도 이 회차는 안 바뀜)"""
    photo["frozen"] = counted_boxes(photo).copy()
    st.session_state.edit_mode = None


def unconfirm_photo(photo):
    photo["frozen"] = None


# ---------------------------------------------------------------- 화면: 입력
st.title("🎓 출튀를 잡아라")
st.caption("강의실 사진을 시간 순서대로 올리면 AI가 사진마다 사람 수를 세고, 인원 변화를 보여 줘요.")

model = load_model()

tab_upload, tab_camera = st.tabs(["📁 사진 업로드 (여러 장)", "📷 카메라 촬영"])

with tab_upload:
    files = st.file_uploader(
        "강의실 사진을 올려 주세요 (여러 장 선택 가능, 나중에 더 올려도 누적돼요)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

with tab_camera:
    st.caption("찍을 때마다 다음 회차로 추가돼요.")
    use_camera = st.toggle("카메라 켜기")
    shot = None
    if use_camera:
        shot = st.camera_input("강의실을 찍어 주세요", key=f"camera_{st.session_state.uploader_key}")

# 아직 처리하지 않은 새 사진만 골라서 탐지 (파일 이름의 숫자 순서대로)
pending = [f for f in sorted(files or [], key=lambda f: natural_key(f.name))
           if f.file_id not in st.session_state.seen_ids]
if shot is not None and shot.file_id not in st.session_state.seen_ids:
    pending.append(shot)

if pending:
    progress = st.progress(0.0, text="AI가 사람을 세는 중...")
    for i, f in enumerate(pending, start=1):
        # 카메라로 찍은 사진은 "날짜_시간.jpg"(한국 시간)로 이름 붙이기 → 나중에 자동 촬영 파일과 같은 형식
        name = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S.jpg") if f is shot else None
        add_photo(f, name)
        progress.progress(i / len(pending), text=f"AI가 사람을 세는 중... ({i}/{len(pending)})")
    progress.empty()

photos = st.session_state.photos
if not photos:
    st.info("사진을 올리거나 카메라로 찍으면 분석을 시작해요. 사진을 추가하면 인원 변화를 볼 수 있어요.")
    st.info("빛 번짐, 흔들림, 너무 어두움 등으로 머리가 잘 안 보이면 AI가 못 잡을 수 있어요.")
    st.info("자리에서 일어나 움직이는 수업이라면(ex실험) 사용이 어려울 수 있어요. CCTV처럼 고정된 자리에서 앉아 있는 수업에 적합해요.")
    st.stop()

# ---------------------------------------------------------------- 화면: 인원 변화 요약
counts = [head_count(p) for p in photos]
first, last = counts[0], counts[-1]
col1, col2, col3 = st.columns(3)
col1.metric("1회차 인원", f"{first}명")
col2.metric("최근 인원", f"{last}명", delta=f"{last - first}명")
col3.metric("사진 수", f"{len(photos)}장")

if last < first:
    st.warning(f"⚠️ 1회차보다 {first - last}명 줄었어요. 출튀가 의심되면 출석을 다시 불러 확인해 보세요.")
elif len(photos) > 1:
    st.success("1회차보다 줄어든 인원이 없어요.")

times = [shot_time(p["name"]) for p in photos]
table = pd.DataFrame({
    "회차": [f"{i}회차" for i in range(1, len(photos) + 1)],
    "촬영 시각": [t.strftime("%m/%d %H:%M") if t else "-" for t in times],
    "파일": [p["name"] for p in photos],
    "인원(명)": counts,
    "1회차 대비": [c - first for c in counts],
    "직접 추가": [len(p["manual"]) for p in photos],
    "확정": ["✅" if p["frozen"] is not None else "" for p in photos],
})
st.button("🗑️ 사진 모두 삭제", on_click=delete_all, type="primary")

# ---------------------------------------------------------------- 화면: 사진 확인·정정
st.subheader("🔍 사진 확인 · 정정")
st.caption("초록 = AI가 찾은 머리 · 파랑(+) = 직접 추가한 머리 · 빨강(X) = 제외한 자리")

ids = [p["id"] for p in photos]
if st.session_state.get("photo_sel") not in ids:  # 처음이거나 선택했던 사진을 지웠으면 1회차로
    st.session_state.photo_sel = ids[0]


def photo_label(pid):
    i = ids.index(pid)
    done = " · ✅확정" if photos[i]["frozen"] is not None else ""
    return f"{i + 1}회차 · {photos[i]['name']} · {counts[i]}명{done}"


selected = st.selectbox("회차 선택", ids, key="photo_sel", format_func=photo_label)
idx = ids.index(selected)
photo = photos[idx]
if st.session_state.get("mode_photo") != photo["id"]:  # 다른 회차로 바꾸면 모드 선택부터 다시
    st.session_state.mode_photo = photo["id"]
    st.session_state.edit_mode = None
    st.session_state.picked = set()

mode = st.session_state.edit_mode
confirmed = photo["frozen"] is not None
picked = st.session_state.picked
result_image = draw_result(photo,
                           picked_heads=picked if mode == "exclude" else (),
                           picked_excluded=picked if mode == "release" else ())

if confirmed:
    # ----- 확정된 회차: 수정 잠금
    st.image(result_image, width="stretch")
    st.success(f"✅ {idx + 1}회차는 {counts[idx]}명으로 확정됐어요.")
    st.button("✏️ 다시 수정하기 (확정 취소)", on_click=unconfirm_photo, args=(photo,))

elif mode is None:
    # ----- 모드 선택
    st.image(result_image, width="stretch")
    st.markdown("**수정할 모드를 골라 주세요**")
    m1, m2, m3 = st.columns(3)
    m1.button("✋ 놓친 머리 추가", on_click=start_mode, args=("add",), width="stretch")
    m2.button("🚫 머리 아님 제외", on_click=start_mode, args=("exclude",), width="stretch")
    m3.button("↩️ 제외 취소", on_click=start_mode, args=("release",), width="stretch",
              disabled=not st.session_state.excluded)
    st.button(f"✅ {idx + 1}회차 확정 (수정 끝)", on_click=confirm_photo, args=(photo,), type="primary")

elif mode == "add":
    # ----- 놓친 머리 추가: 빈 곳을 누르면 파란 +박스 추가, 파란 박스를 다시 누르면 취소
    st.info("✋ 놓친 머리 추가 모드: 사진에서 머리를 누르세요. 잘못 추가한 파란 박스는 다시 누르면 지워져요. "
            "이 회차에만 적용돼요.")
    point = tap_on_image(result_image, key=f"click_add_{photo['id']}")
    if point:
        hit = find_box(photo["manual"], *point) if photo["manual"] else None
        if hit is not None and (photo["manual"][hit][0] <= point[0] <= photo["manual"][hit][2]
                                and photo["manual"][hit][1] <= point[1] <= photo["manual"][hit][3]):
            photo["manual"].pop(hit)          # 파란 박스 안을 눌렀으면 취소
        else:
            photo["manual"].append(head_box_at(photo, *point))
        st.rerun()
    st.button("완료", on_click=finish_mode, args=(photo,), type="primary")

elif mode == "exclude":
    # ----- 머리 아님 제외: 박스를 눌러 고르고 완료 → 그 자리를 모든 회차에서 제외
    st.info("🚫 머리 아님 제외 모드: 가방 등 머리가 아닌 초록 박스를 누르세요 (주황색으로 바뀌어요). "
            "다시 누르면 선택 취소, 완료를 누르면 모든 회차에 적용돼요.")
    point = tap_on_image(result_image, key=f"click_exclude_{photo['id']}")
    if point:
        toggle_pick(find_box(counted_boxes(photo), *point))
        st.rerun()
    st.caption("고른 박스: " + (", ".join(str(n + 1) for n in sorted(picked)) if picked else "없음"))
    c1, c2 = st.columns(2)
    c1.button("선택 비우기", on_click=clear_picked, disabled=not picked, width="stretch")
    c2.button("완료", on_click=finish_mode, args=(photo,), type="primary", width="stretch")

elif mode == "release":
    # ----- 제외 취소: 제외한 자리에 학생이 앉았을 때 빨간 X박스를 눌러 다시 셈
    st.info("↩️ 제외 취소 모드: 다시 셀 빨간 X박스를 누르세요 (주황색으로 바뀌어요). "
            "그 자리에 학생이 앉았을 때 써요. 완료를 누르면 적용돼요.")
    point = tap_on_image(result_image, key=f"click_release_{photo['id']}")
    if point:
        toggle_pick(find_box(st.session_state.excluded, *point))
        st.rerun()
    st.caption("고른 자리: " + (", ".join(f"X{k + 1}" for k in sorted(picked)) if picked else "없음"))
    c1, c2 = st.columns(2)
    c1.button("제외한 자리 모두 해제", on_click=release_all_excluded, width="stretch")
    c2.button("완료", on_click=finish_mode, args=(photo,), type="primary", width="stretch")

st.button("이 회차 사진 삭제", key=f"del_{photo['id']}", on_click=delete_photo, args=(photo["id"],))

# ---------------------------------------------------------------- 화면: 그래프·표 (맨 아래)
st.subheader("📈 회차별 인원")
if len(photos) > 1:
    # 가로축 이름: 촬영 시각을 알면 "1회차 14:56", 모르면 "1회차"
    table["가로축"] = [f"{r} {t.strftime('%H:%M')}" if t else r for r, t in zip(table["회차"], times)]
    chart = (
        alt.Chart(table)
        .mark_line(point=True)
        .encode(
            x=alt.X("가로축", title="회차", sort=list(table["가로축"]), axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("인원(명)", scale=alt.Scale(domainMin=0)),  # 인원은 0명 아래로 내려갈 수 없으니 y축은 0부터
            tooltip=["회차", "촬영 시각", "파일", "인원(명)", "1회차 대비"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, width="stretch")
st.dataframe(table.drop(columns="가로축", errors="ignore"), hide_index=True, width="stretch")