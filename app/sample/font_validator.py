import os
from fontTools.ttLib import TTFont

def has_won_symbol(font_path):
    try:
        font = TTFont(font_path)
        cmap = font['cmap'].getBestCmap()
        return 0x20A9 in cmap
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return False

def check_fonts_in_directory(font_dir):
    fonts = [
        os.path.join(font_dir, p)
        for p in os.listdir(font_dir)
        if os.path.splitext(p)[1].lower() == ".ttf"
    ]

    results = {font: has_won_symbol(font) for font in fonts}
    return results

def main():
    font_directory = r""
    results = check_fonts_in_directory(font_directory)

    for font, contains_won in results.items():
        print(f"{font}: {'₩ 포함' if contains_won else '₩ 없음'}")

if __name__ == "__main__":
    main()