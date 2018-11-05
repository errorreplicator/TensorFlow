import turtle as tf

def ile(miesiac,dzien):
    suma_dni = 0
    if miesiac == 1:
        return dzien-1
    else:
        for x in range(1,miesiac+1):
            if x<miesiac:
                if x%2==0:
                    suma_dni+=15
                else:
                    suma_dni+=12
            else:
                suma_dni+=(dzien-1)
        return (suma_dni)


print(ile(2,2))
