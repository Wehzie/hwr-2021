class HebrewAlphabet:
    """
    All letters in the hebrew alphabet and,
    Habbakuk font's mapping of characters to hebrew letters.
    """

    letter_str: str = (
        "Alef,Ayin,Bet,Dalet,Gimel,He,Het,Kaf,Kaf-final,Lamed,Mem,"
        + "Mem-medial,Nun-final,Nun-medial,Pe,Pe-final,Qof,Resh,Samekh,Shin,"
        + "Taw,Tet,Tsadi-final,Tsadi-medial,Waw,Yod,Zayin"
    )

    letter_li: list = letter_str.split(",")

    font_str: str = "),(,b,d,g,h,x,k,\\,l,m,{,n,},p,v,q,r,s,$,t,+,j,c,w,y,z"

    font_li: list = font_str.split(",")
