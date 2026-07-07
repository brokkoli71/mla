baseline16: 
21 bis erste vmac
dann alle 1.5 zeilen vmac (32/48)
dann 9 zeilen bis ende

anzahl vmac: (x/8)^2 viele 8x8x64 blöcke also (x/8)^2 * 8 vmacs = x^2 / 8 für 8x8x8
baseline zeilen: 21 + 9 + 1.5 * vmacs = 30 + 3 / 16 * x^2 
unfused conv zeilen: 
unfused matmul zeilen: 13 + 11 + vmacs = 24 + 1 / 8 * x^2

size16:
76 zeilen conv
13 zeilen bis vmac (warmup)
11 zeilen bis ende (cooldown)

size32:
143 zeilen conv
13 zeilen bis vmac
jede zeile vmac (theorie, aber abweichungen)
11 zeilen bis ende

size64:
279 zeilen conv
13 zeilen bis vmac
jede zeile vmac (theorie, aber abweichungen)
11 zeilen bis ende
