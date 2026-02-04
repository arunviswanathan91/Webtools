from typing import List

# Publication-ready color palettes

# Nature Publishing Group (NPG)
NPG: List[str] = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", 
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"
]

# New England Journal of Medicine (NEJM)
NEJM: List[str] = [
    "#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", 
    "#6F99AD", "#FFDC91", "#EE4C97"
]

# Journal of the American Medical Association (JAMA)
JAMA: List[str] = [
    "#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", 
    "#6A6599", "#80796B"
]

# The Lancet
LANCET: List[str] = [
    "#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", 
    "#FDAF91", "#AD002A", "#ADB6B6", "#1B1919"
]

# Colorblind Safe (Okabe-Ito, modified)
COLORBLIND_SAFE: List[str] = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", 
    "#D55E00", "#CC79A7", "#999999"
]

def get_palette(name: str) -> List[str]:
    palettes = {
        "npg": NPG,
        "nejm": NEJM,
        "jama": JAMA,
        "lancet": LANCET,
        "colorblind": COLORBLIND_SAFE
    }
    return palettes.get(name.lower(), NPG)
