import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from fontTools.ttLib import TTFont


def font_preview():
    font_path = r"C:\Users\WONJANGHO\Desktop\micrenc.ttf"
    font_prop = FontProperties(fname=font_path)

    ttf = TTFont(font_path)
    cmap = ttf.getBestCmap()
    available_glyphs = set(cmap.keys())  # 지원하는 유니코드 목록

    characters = [chr(code) for code in cmap.keys()][:100]

    plt.figure(figsize=(6, len(characters) * 0.5))
    plt.axis("off")

    for i, char in enumerate(characters):
        unicode_value = f"U+{ord(char):04X}"

        # 만약 폰트에 없는 글자라면 기본 폰트로 출력
        if ord(char) in available_glyphs:
            char_font = font_prop
        else:
            char_font = None  # 기본 폰트 사용

        # 글자 출력 (해당 폰트 적용)
        plt.text(
            0, -i, char, fontproperties=char_font, fontsize=30, ha="left", va="center"
        )

        # 유니코드 출력 (기본 폰트)
        plt.text(0.5, -i, unicode_value, fontsize=15, ha="left", va="center")

    plt.xlim(-0.1, 1)
    plt.ylim(-len(characters), 1)
    plt.show()


if __name__ == "__main__":
    font_preview()
