from typing import List


class HebrewAlphabet:
    """All letters in the hebrew alphabet and a Habbakuk font mapping."""

    letter_str: str = (
        "Alef,Ayin,Bet,Dalet,Gimel,He,Het,Kaf,Kaf-final,Lamed,Mem,"
        "Mem-medial,Nun-final,Nun-medial,Pe,Pe-final,Qof,Resh,Samekh,Shin,"
        "Taw,Tet,Tsadi-final,Tsadi-medial,Waw,Yod,Zayin"
    )

    letter_li: List[str] = letter_str.split(",")

    font_str: str = "),(,b,d,g,h,x,k,\\,l,m,{,n,},p,v,q,r,s,$,t,+,j,c,w,y,z"

    font_li: List[str] = font_str.split(",")

    unicode_dict = {
        "Alef": u"\u05D0",
        "Ayin": u"\u05E2",
        "Bet": u"\u05D1",
        "Dalet": u"\u05D3",
        "Gimel": u"\u05D2",
        "He": u"\u05D4",
        "Het": u"\u05D7",
        "Kaf": u"\u05DB",
        "Kaf-final": u"\u05DA",
        "Lamed": u"\u05DC",
        "Mem": u"\u05DD",  # This if Mem-Final, check
        "Mem-medial": u"\u05DE",
        "Nun-final": u"\u05DF",
        "Nun-medial": u"\u05E0",  # This is Nun
        "Pe": u"\u05E4",
        "Pe-final": u"\u05E3",
        "Qof": u"\u05E7",
        "Resh": u"\u05E8",
        "Samekh": u"\u05E1",
        "Shin": u"\u05E9",
        "Taw": u"\u05EA",
        "Tet": u"\u05D8",
        "Tsadi-final": u"\u05E5",
        "Tsadi-medial": u"\u05E6",  # This is Tsadi
        "Waw": u"\u05D5",
        "Yod": u"\u05D9",
        "Zayin": u"\u05D6",
    }


class HebrewStyles:
    """Three different styles of hebrew writing."""

    style_str: str = "Archaic,Hasmonean,Herodian"

    style_li: List[str] = style_str.split(",")
