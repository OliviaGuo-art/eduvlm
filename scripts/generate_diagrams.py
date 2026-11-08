import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path("data/raw/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1200, 800
BG = "#F7F9FC"
DARK = "#172033"
BLUE = "#2878D0"
RED = "#E24A4A"
GREEN = "#26936F"
ORANGE = "#E99132"
GRAY = "#718096"
LIGHT_BLUE = "#D9EDFF"

FONT_PATH = None
for candidate in [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]:
    if candidate.exists():
        FONT_PATH = candidate
        break

if FONT_PATH is None:
    raise FileNotFoundError("未找到微软雅黑或黑体字体文件")


def get_font(size):
    return ImageFont.truetype(str(FONT_PATH), size)


TITLE_FONT = get_font(44)
TEXT_FONT = get_font(30)
SMALL_FONT = get_font(25)
FORMULA_FONT = get_font(36)


def new_canvas(title):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text(
        (WIDTH // 2, 55),
        title,
        font=TITLE_FONT,
        fill=DARK,
        anchor="mm",
    )
    draw.line((80, 105, 1120, 105), fill="#CBD5E1", width=3)
    return image, draw


def text(draw, position, content, font=TEXT_FONT, fill=DARK, anchor="mm"):
    draw.text(position, content, font=font, fill=fill, anchor=anchor)


def arrow(draw, start, end, fill=BLUE, width=8, head_size=24):
    draw.line((*start, *end), fill=fill, width=width)

    angle = math.atan2(
        end[1] - start[1],
        end[0] - start[0],
    )
    left = (
        end[0] - head_size * math.cos(angle - 0.55),
        end[1] - head_size * math.sin(angle - 0.55),
    )
    right = (
        end[0] - head_size * math.cos(angle + 0.55),
        end[1] - head_size * math.sin(angle + 0.55),
    )
    draw.polygon([end, left, right], fill=fill)


def save(image, filename):
    path = OUTPUT_DIR / filename
    image.save(path)
    print("Saved:", path)


# 1. 自由落体
image, draw = new_canvas("理想自由落体：速度均匀增大")
for y, label in [(190, "t = 0"), (360, "t = 1"), (600, "t = 2")]:
    draw.ellipse((200, y - 28, 256, y + 28), fill=BLUE)
    text(draw, (330, y), label)
arrow(draw, (150, 170), (150, 650), fill=RED)
text(draw, (105, 410), "g", font=FORMULA_FONT, fill=RED)
text(draw, (760, 290), "v = g t", font=FORMULA_FONT)
text(draw, (760, 390), "h = 1/2 g t^2", font=FORMULA_FONT)
text(draw, (760, 520), "忽略空气阻力", fill=GRAY)
save(image, "free_fall_001.png")


# 2. 牛顿第二定律
image, draw = new_canvas("牛顿第二定律：合外力决定加速度")
draw.line((100, 590, 1100, 590), fill=GRAY, width=5)
draw.rounded_rectangle((350, 350, 610, 580), radius=12, fill=LIGHT_BLUE, outline=BLUE, width=4)
text(draw, (480, 465), "质量 m", font=FORMULA_FONT)
arrow(draw, (610, 460), (960, 460), fill=RED)
text(draw, (790, 415), "合外力 F", fill=RED)
arrow(draw, (610, 300), (900, 300), fill=GREEN)
text(draw, (760, 255), "加速度 a", fill=GREEN)
text(draw, (600, 690), "F_net = m a", font=FORMULA_FONT)
save(image, "newtons_second_law_001.png")


# 3. 摩擦力
image, draw = new_canvas("摩擦力：先判断相对运动状态")
draw.line((100, 590, 1100, 590), fill=GRAY, width=5)
draw.rounded_rectangle((420, 340, 680, 580), radius=12, fill=LIGHT_BLUE, outline=BLUE, width=4)
arrow(draw, (680, 460), (980, 460), fill=RED)
text(draw, (830, 415), "外加作用", fill=RED)
arrow(draw, (420, 510), (170, 510), fill=GREEN)
text(draw, (280, 555), "摩擦力 f", fill=GREEN)
arrow(draw, (550, 340), (550, 170), fill=BLUE)
text(draw, (610, 220), "N", fill=BLUE)
arrow(draw, (550, 580), (550, 730), fill=ORANGE)
text(draw, (620, 670), "mg", fill=ORANGE)
text(draw, (940, 650), "静摩擦会调整\n滑动后再用滑动摩擦模型", font=SMALL_FONT)
save(image, "friction_001.png")


# 4. 匀速圆周运动
image, draw = new_canvas("匀速圆周运动：方向改变产生向心加速度")
center = (480, 420)
radius = 230
draw.ellipse(
    (center[0] - radius, center[1] - radius,
     center[0] + radius, center[1] + radius),
    outline=GRAY,
    width=4,
)
draw.ellipse((465, 405, 495, 435), fill=DARK)
point = (710, 420)
draw.ellipse((685, 395, 735, 445), fill=BLUE)
arrow(draw, point, center, fill=RED)
text(draw, (590, 380), "向心加速度 a", font=SMALL_FONT, fill=RED)
arrow(draw, point, (710, 160), fill=GREEN)
text(draw, (770, 230), "速度 v", fill=GREEN)
text(draw, (940, 460), "|v| 不变\n方向改变", font=FORMULA_FONT)
save(image, "uniform_circular_motion_001.png")


# 5. 机械能守恒
image, draw = new_canvas("只有重力做功：势能与动能相互转化")
draw.polygon([(180, 620), (700, 620), (700, 190)], fill="#E2E8F0", outline=GRAY)
draw.ellipse((610, 215, 675, 280), fill=BLUE)
draw.ellipse((250, 535, 315, 600), fill=GREEN)
text(draw, (780, 245), "高处：势能较大", fill=BLUE)
text(draw, (450, 545), "低处：动能较大", fill=GREEN)
arrow(draw, (930, 230), (930, 570), fill=ORANGE)
text(draw, (930, 650), "E_p  →  E_k", font=FORMULA_FONT, fill=ORANGE)
text(draw, (600, 710), "忽略摩擦等非保守力做功", fill=GRAY)
save(image, "mechanical_energy_001.png")


# 6. 动量守恒
image, draw = new_canvas("碰撞：各自动量可变，系统总动量可守恒")
text(draw, (250, 150), "碰撞前", font=FORMULA_FONT)
draw.ellipse((140, 280, 230, 370), fill=BLUE)
draw.ellipse((550, 280, 640, 370), fill=ORANGE)
text(draw, (185, 405), "m1")
text(draw, (595, 405), "m2")
arrow(draw, (240, 325), (400, 325), fill=BLUE)
arrow(draw, (540, 325), (400, 325), fill=ORANGE)

text(draw, (250, 500), "碰撞后", font=FORMULA_FONT)
draw.ellipse((140, 560, 230, 650), fill=BLUE)
draw.ellipse((550, 560, 640, 650), fill=ORANGE)
arrow(draw, (240, 605), (390, 605), fill=BLUE)
arrow(draw, (540, 605), (400, 605), fill=ORANGE)

text(draw, (900, 340), "合外力冲量\n可忽略", font=FORMULA_FONT)
text(draw, (900, 520), "系统总动量守恒", font=TEXT_FONT, fill=GREEN)
save(image, "momentum_conservation_001.png")


# 7. 力矩
image, draw = new_canvas("力矩：力、力臂和方向共同决定转动效果")
pivot = (250, 560)
force_point = (900, 300)
draw.ellipse((205, 515, 295, 605), fill=LIGHT_BLUE, outline=BLUE, width=4)
text(draw, (250, 650), "转轴 O", fill=BLUE)
draw.line((*pivot, *force_point), fill=DARK, width=24)
arrow(draw, force_point, (900, 120), fill=RED)
text(draw, (970, 190), "力 F", fill=RED)
text(draw, (570, 390), "力臂相关距离 r", font=TEXT_FONT)
text(draw, (600, 700), "τ = r F sinθ", font=FORMULA_FONT)
save(image, "torque_001.png")


# 8. 浮力
image, draw = new_canvas("阿基米德原理：关注排开液体的体积")
draw.rectangle((240, 170, 680, 660), outline=GRAY, width=5)
draw.rectangle((250, 340, 670, 650), fill="#CDEBFF")
draw.rectangle((380, 400, 550, 570), fill=ORANGE)
arrow(draw, (465, 400), (465, 220), fill=BLUE)
text(draw, (540, 280), "浮力 F_b", fill=BLUE)
arrow(draw, (465, 570), (465, 730), fill=RED)
text(draw, (550, 675), "重力 mg", fill=RED)
text(draw, (920, 350), "F_b = ρ_液 g V_排", font=FORMULA_FONT)
text(draw, (920, 480), "完全浸没后\n深度增加≠浮力必增", font=SMALL_FONT)
save(image, "buoyancy_001.png")


# 9. 欧姆定律
image, draw = new_canvas("欧姆定律：稳定条件下电压与电流成正比")
draw.line((250, 250, 480, 250), fill=DARK, width=6)
draw.line((700, 250, 900, 250), fill=DARK, width=6)
draw.line((900, 250, 900, 600), fill=DARK, width=6)
draw.line((900, 600, 250, 600), fill=DARK, width=6)
draw.line((250, 600, 250, 250), fill=DARK, width=6)
draw.rounded_rectangle((480, 195, 700, 305), radius=12, fill=LIGHT_BLUE, outline=BLUE, width=4)
text(draw, (590, 250), "电阻 R", font=FORMULA_FONT)
draw.line((200, 385, 300, 385), fill=RED, width=10)
draw.line((215, 455, 285, 455), fill=RED, width=5)
arrow(draw, (330, 145), (820, 145), fill=GREEN)
text(draw, (575, 140), "电流 I", fill=GREEN)
text(draw, (580, 690), "U = I R", font=FORMULA_FONT)
save(image, "ohms_law_001.png")