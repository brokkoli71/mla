## Overview
- [x] 01: 10/10
- [x] 02: 9/10 kleinigkeiten -> Hannes
- [x] 03: 7.5/10 Block swizzling -> Hannes
- [x] 04: 10/10
- [x] 05: 8.50/10 config fixen -> Hannes
- [ ] 06: 8.50/10 -> Falko
- [ ] 07: 8.50/10 -> Falko
- [ ] 08: 6.50/10 -> Falko
- [x] 09: 0/10 -> Hannes
- [ ] 10: 7/10 -> Hannes
- [ ] formatierung in sphinx gegenchecken?
- [ ] skelette aufräumen
- [ ] Projekt schreiben -> Hannes, Falko
- [ ] git tag für finale abgabe



## Anmerkungen

#### 02
- [x] 02: 3b (-0.5P.): Tensor Allocations mit gebenchmarkt
- [x] 4a (-0.5P.): Kernel unterstützt nur Zweierpotenzen (nur padding mode hinschreiben? ausprobieren)
- [x] zu 4b: Gerne noch die ganze Range ...
#### 03
- [x] Anmerkung: Bilder im Report werden nicht angezeigt
- [x] -0,5P. (1b): Speedup fehlt (und auch Runtimes ;))
- [x] -2P. (Aufgabe 4): Implementierung ...
#### 05
- Kein Git Tag im Repo vorhanden
- [x] Task 3b (-0,5P.): Es müsste hier ebenfalls stride[dim_id_b] == stride[dim_id_a] * self.config.dim_sizes[dim_id_a] überprüft werden, da die Dimensionen in der Config nicht unbedingt sortiert sein müssen (Annahme, dass dim_id_a < dim_id_b reicht hier nicht aus)
- [x] 4b (-0,5P.): Optimierte Config fehlt im Report
- [x] 4c (-0,5P.): Korrekt gesplittet, jedoch ist die Logik hinter der Größenfindung nicht ganz korrekt (können wir gerne morgen besprechen)
#### 06
- -0.5P. (3a): Optimierter Kernel hat ein sehr großes k_prim. Dieses wird zwar im Kernel noch gesplittet, das wäre jedoch auch in der Config abbildbar.
- -0,5 (4a) Kernel berechnen teilweise ein falsches Ergebnis.
- -0,5 (4b): Toleranzen sind zu weit gewählt. Da die Elemente im Ergebnistensor zwischen 0 und 1 liegen, wird atol=2e-0 grundsätzlich nicht anschlagen.
- Anmerkung: Die initiale Config sollte nicht dazu führen, dass die Eingabetensoren permutiert werden müssen. Die initiale Config bildet die Strides der Tensoren so ab, wie es der Einsum-String vorgibt (ab, bc → ac bedeutet, dass Dimension b Stride 1 im linken Tensor hat, Stride |c| im rechten Tensor).
- Anmerkung 2: Die L2-Dimensionen sind etwas zu klein gewählt.
#### 07
- Dateiname 'git_link.txt' verwenden. Der Vektor-Move 'vmov' wird von der Move-Unit ausgeführt. 'mova' wird von der Load-Unit A ausgeführt. Registerklassen siehe Folien 08.
#### 08
- Datei bitte "git_link.txt" nennen. Operationen zu den exakten Slots zuordnen. Tabelle mit Operationen ist kaput. Einige Operationen fehlen in der Tabelle. Aufgabe 3 wurde indirekt in Aufgabe 4 beantwortet. Bitte Aufgaben getrennt halten. Bei Aufgabe 4 war eine Skizzierung des Data-Layouts und der Pointer-Updates gewünscht und keine Skizze, auch wenn die gut gelungen ist.
#### 09
- [x] keine abgabe
#### 10
- [ ] Kein Bericht abgeben. 
- [x] Beim MLIR code hat jeweils das zweite wait gefehlt (letzten 8 Zeilen von runtime_sequence duplizieren). Restlicher Code stimmt.