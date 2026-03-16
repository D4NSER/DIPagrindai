# II užduotis. Vieno neurono mokymas sprendžiant klasifikavimo uždavinį

## Darbo tikslas

Darbo tikslas buvo apmokyti vieną sigmoidinį neuroną spręsti dviejų klasių klasifikavimo uždavinį, panaudojant dvi mokymo strategijas:

- paketinį gradientinį nusileidimą (`batch gradient descent`)
- stochastinį gradientinį nusileidimą (`stochastic gradient descent`, `SGD`)

Taip pat buvo atliktas tyrimas, kaip rezultatus veikia epochų skaičius, mokymosi greitis ir pasirinktas gradientinio nusileidimo metodas.

Pastaba: ši ataskaitos dalis buvo parengta padedant DI įrankiui, remiantis mano parašytu kodu ir gautais rezultatais iš `Lab2`.

## Naudoti duomenys

Buvo naudojama `Breast Cancer Wisconsin Original` duomenų aibė iš UCI Machine Learning Repository. Pirminiame rinkinyje yra 699 įrašai ir 11 stulpelių:

- 1 identifikatoriaus stulpelis (`sample_code_number`)
- 9 požymiai
- 1 klasės žymė

Pradinės klasės duomenų rinkinyje buvo:

- `2` – nepiktybinis navikas
- `4` – piktybinis navikas

Sprendžiant dvejetainės klasifikacijos uždavinį, klasės buvo peržymėtos taip:

- `2 -> 0`
- `4 -> 1`

Kadangi identifikatoriaus stulpelis klasifikavimui nėra naudingas, jis buvo pašalintas. Taip pat duomenyse buvo trūkstamų reikšmių, pažymėtų simboliu `?`, todėl tokios eilutės buvo ištrintos.

Po duomenų valymo gauti tokie rezultatai:

- pradinis įrašų skaičius: `699`
- pašalintos eilutės su trūkstamomis reikšmėmis: `16`
- likęs įrašų skaičius: `683`
- naudotų požymių skaičius: `9`

Po valymo klasių pasiskirstymas buvo:

- klasė `0`: `444`
- klasė `1`: `239`

Duomenų paruošimo kodas yra faile `lab2/src/task1_prepare_data.py`, o bendra duomenų apdorojimo logika yra faile `lab2/src/breast_cancer_data.py`.

## Duomenų paruošimas

Duomenų paruošimo metu buvo atlikti šie žingsniai:

1. Nuskaitytas pradinis `.data` failas.
2. Pašalintas `sample_code_number` stulpelis.
3. Ištrintos eilutės, kuriose buvo `?`.
4. Klasių žymės pakeistos į `0` ir `1`.
5. Duomenų eilutės atsitiktinai permaišytos.

Permaišymas buvo svarbus tam, kad mokymo pradžioje nebūtų vien tik vienos klasės pavyzdžių. Tai ypač svarbu SGD metodui, nes jis svorius keičia po kiekvieno įrašo.

Pastaba: ši pastraipa buvo parengta padedant DI įrankiui, tačiau aprašo mano atliktus duomenų apdorojimo veiksmus.

## Duomenų padalinimas

Paruošti duomenys buvo padalinti į tris aibes santykiu `80:10:10`:

- mokymo aibė (`train`) – `546` įrašai
- validavimo aibė (`validation`) – `68` įrašai
- testavimo aibė (`test`) – `69` įrašai

Šių aibių paskirtis:

- mokymo aibė naudojama svorių ir poslinkio (`bias`) koregavimui
- validavimo aibė naudojama stebėti, kaip modelis generalizuoja duomenis, kurių mokymo metu tiesiogiai nematė
- testavimo aibė naudojama galutiniam modelio įvertinimui

Papildomai prieš mokymą požymiai buvo standartizuoti pagal mokymo aibės vidurkius ir standartinius nuokrypius. Tai padėjo stabilizuoti sigmoidinio neurono mokymą ir pagreitinti konvergavimą.

## Sigmoidinis neuronas

Šiame darbe buvo realizuotas vienas dirbtinis neuronas su sigmoidine aktyvacijos funkcija:

`y = sigmoid(w * x + b)`

kur:

- `x` – įvesties požymių vektorius
- `w` – svorių vektorius
- `b` – poslinkis

Sigmoidinė funkcija:

`sigmoid(z) = 1 / (1 + e^(-z))`

Kadangi sigmoidinės funkcijos išėjimas yra intervale `(0; 1)`, klasė buvo nustatoma apvalinant gautą reikšmę iki artimiausio sveiko skaičiaus:

- jei išėjimas arčiau `0`, klasė laikoma `0`
- jei išėjimas arčiau `1`, klasė laikoma `1`

Paklaidai skaičiuoti buvo naudota dvejetainė kryžminė entropija (`binary cross-entropy`), o klasifikavimo kokybei vertinti naudotas tikslumas (`accuracy`).

Sigmoidinio neurono realizacija pateikta faile `lab2/src/single_neuron.py`.

## Pradiniai svoriai

Pradiniai svoriai buvo sugeneruoti atsitiktinai, naudojant normalųjį skirstinį su maža dispersija. Tai leidžia pradėti mokymą nuo mažų reikšmių ir išvengti per didelių aktyvacijos reikšmių mokymo pradžioje. Poslinkis `bias` pradžioje buvo lygus `0`.

Naudotas fiksuotas atsitiktinių skaičių generatoriaus pradinis `seed = 2026`, kad eksperimentų rezultatai būtų stabilūs tarp paleidimų.

## Gradientinis nusileidimas

### Paketinis gradientinis nusileidimas

Paketinio gradientinio nusileidimo atveju kiekvienos epochos metu gradientas apskaičiuojamas panaudojant visą mokymo aibę. Tik po to atnaujinami svoriai ir poslinkis.

Šiuo atveju viena epocha reiškia vieną pilną perėjimą per visus mokymo duomenis ir vieną bendrą svorių atnaujinimą.

### Stochastinis gradientinis nusileidimas

Stochastinio gradientinio nusileidimo atveju svoriai atnaujinami po kiekvieno mokymo įrašo. Kiekvienos epochos pradžioje mokymo duomenys permaišomi atsitiktine tvarka.

Šiuo atveju viena epocha reiškia vieną pilną perėjimą per visus mokymo duomenis, tačiau svoriai per tą epochą atnaujinami daug kartų, po vieną kartą kiekvienam įrašui.

## Programos rezultatai

Buvo realizuoti abu reikalauti variantai:

- mokymas naudojant `batch gradient descent`
- mokymas naudojant `SGD`

Programa kiekvienam variantui išsaugo:

- gautus svorius
- `bias`
- paklaidas po kiekvienos epochos mokymo ir validavimo aibėms
- tikslumus po kiekvienos epochos mokymo ir validavimo aibėms
- testo paklaidą
- testo tikslumą
- mokymo laiką
- kiekvieno testo įrašo prognozuotą ir tikrą klasę

Šie rezultatai saugomi aplanke `lab2/results/task2`.

Galutinis geriausias modelis, parinktas pagal validavimo rezultatus, papildomai saugomas aplanke `lab2/results/final_best_model`.

## Bazinis rezultatas

Baziniu variantu buvo laikomas `batch` metodas su:

- `learning rate = 0.05`
- `epochs = 200`

Gauti šio varianto svoriai:

- `w1 = 0.563612`
- `w2 = 0.603908`
- `w3 = 0.439508`
- `w4 = 0.582850`
- `w5 = 0.501688`
- `w6 = 0.759795`
- `w7 = 0.505242`
- `w8 = 0.521035`
- `w9 = 0.321580`
- `bias = -0.492629`

Kadangi šio varianto geriausia validavimo epocha sutapo su paskutine epocha, buvo gauta:

- mokymo paklaida: `0.095547`
- validavimo paklaida: `0.100565`
- mokymo tikslumas: `0.972527`
- validavimo tikslumas: `0.970588`

Testavimo rezultatai, naudojant pagal validaciją pasirinktą šio varianto modelį:

- testo paklaida: `0.131468`
- testo tikslumas: `0.942029`
- mokymo laikas: `0.099276 s`

Detalūs šio varianto rezultatai pateikti faile `lab2/results/task2/batch/summary.json`.

## Tyrimas

Buvo ištirti trys mokymosi greičio variantai:

- `0.01`
- `0.05`
- `0.1`

Kiekvienas iš jų buvo testuojamas dviem metodais:

- `batch`
- `sgd`

Visais atvejais buvo naudota `200` epochų.

### Rezultatai pagal learning rate

| Metodas | Learning rate | Best epoch | Best val loss | Best val accuracy | Test loss | Test accuracy | Laikas, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| batch | 0.01 | 200 | 0.198756 | 0.970588 | 0.196712 | 0.942029 | 0.029907 |
| batch | 0.05 | 200 | 0.100565 | 0.970588 | 0.131468 | 0.942029 | 0.036085 |
| batch | 0.10 | 200 | 0.089245 | 0.970588 | 0.130884 | 0.942029 | 0.032690 |
| sgd | 0.01 | 13 | 0.084197 | 0.970588 | 0.145063 | 0.942029 | 1.070529 |
| sgd | 0.05 | 2 | 0.084991 | 0.970588 | 0.143846 | 0.942029 | 1.106512 |
| sgd | 0.10 | 5 | 0.085887 | 0.970588 | 0.168795 | 0.942029 | 1.100183 |

Pilna eksperimentų lentelė pateikta faile `lab2/results/task3/experiment_summary.csv`.

### Paklaidos priklausomybė nuo epochų

Paklaidų kitimas pagal epochas buvo pavaizduotas grafikuose. Iš `batch` grafiko matyti, kad tiek mokymo, tiek validavimo paklaida nuosekliai mažėja. `SGD` atveju paklaida mažėja greitai, tačiau kreivės yra labiau banguotos, nes svoriai atnaujinami po kiekvieno įrašo.

Čia reikia įterpti paveikslus:

- `lab2/results/task3/batch_lr_0_05_curves.png`
- `lab2/results/task3/sgd_lr_0_05_curves.png`

### Tikslumo priklausomybė nuo epochų

Tikslumo kreivės rodo, kad abu metodai gana greitai pasiekia aukštą validavimo tikslumą. `Batch` metodo tikslumas auga tolygiau, o `SGD` dažnai labai greitai pasiekia aukštą reikšmę jau pačiose pirmose epochose.

### Learning rate įtaka

Iš rezultatų matyti, kad `batch` metodui didesnis mokymosi greitis šiame darbe buvo naudingas. Mažiausia `test loss` reikšmė gauta su `batch` ir `learning rate = 0.1`.

`SGD` atveju visi trys variantai davė tą patį geriausią validavimo tikslumą, tačiau mažiausia validavimo paklaida buvo gauta su `learning rate = 0.01`. Taip pat matyti, kad `SGD` geriausia epocha buvo randama anksti, todėl vien paskutinės epochos stebėjimas nebūtinai parodo geriausią variantą.

### Batch ir SGD palyginimas

Pagal validavimo tikslumą abu metodai šiame darbe pasirodė labai panašiai, nes visais pagrindiniais atvejais buvo gautas `0.970588` validavimo tikslumas.

Tačiau pagal mokymo laiką skirtumas buvo labai aiškus:

- `batch` metodas veikė maždaug `0.030 - 0.036 s`
- `sgd` metodas veikė maždaug `1.071 - 1.107 s`

Taigi šiame darbe `batch` buvo ženkliai greitesnis už `SGD`. Vis dėlto mažiausia validavimo paklaida buvo gauta su `SGD`, todėl galutinis modelio pasirinkimas buvo daromas ne pagal laiką, o pagal validavimo rezultatus.

### Geriausias variantas

Pagal užduoties logiką geriausias variantas turi būti parenkamas pagal validavimo rezultatus. Todėl galutiniu modeliu buvo laikomas tas variantas, kuris pasiekė didžiausią validavimo tikslumą ir mažiausią validavimo paklaidą.

Galutiniu pasirinktu variantu tapo:

- metodas: `sgd`
- `learning rate = 0.01`
- bendras mokymo epochų skaičius: `200`
- pasirinkta geriausia epocha pagal validaciją: `13`

Šis variantas davė:

- mokymo paklaidą pasirinktoje epochoje: `0.069249`
- validavimo paklaidą: `0.084197`
- validavimo tikslumą: `0.970588`
- mokymo tikslumą pasirinktoje epochoje: `0.974359`
- testo paklaidą: `0.145063`
- testo tikslumą: `0.942029`

Kadangi keli variantai pasiekė tą patį validavimo tikslumą, galutinis pasirinkimas buvo daromas pagal mažesnę validavimo paklaidą. Galutinio varianto suvestinė pateikta faile `lab2/results/final_best_model/summary.json`.

## Testavimo duomenų klasifikavimo rezultatai

Kiekvienam testavimo įrašui buvo išsaugota:

- prognozuota tikimybė
- prognozuota klasė
- tikroji klasė

Šie duomenys pateikti failuose:

- `lab2/results/task2/batch/test_predictions.csv`
- `lab2/results/task2/sgd/test_predictions.csv`
- `lab2/results/final_best_model/test_predictions.csv`

Ataskaitoje galima įdėti visą lentelę arba jos dalį, priklausomai nuo reikalaujamo detalumo.

## Kodo komentarai ir struktūra

Kodas sąmoningai buvo rašomas kuo paprasčiau, kad būtų aiškus ir tinkamas ataskaitai. Logika suskirstyta į kelis trumpus failus:

- `task1_prepare_data.py` – duomenų paruošimas
- `single_neuron.py` – neurono matematika ir mokymas
- `task2_train_and_evaluate.py` – mokymas, validavimas ir testavimas
- `task3_experiments.py` – tyrimai ir grafikų generavimas
- `task4_select_best_model.py` – geriausio modelio parinkimas pagal validaciją
- `run_all_lab2.py` – viso darbo paleidimas papunkčiui

Toks suskaidymas leidžia lengviau paaiškinti, kur atliekamas kiekvienas veiksmas.

## Išvados

1. Išvalius duomenų aibę ir pašalinus eilutes su trūkstamomis reikšmėmis, liko 683 tinkami įrašai su 9 požymiais.
2. Tiek paketinis, tiek stochastinis gradientinis nusileidimas leido pasiekti aukštą klasifikavimo tikslumą.
3. Visais pagrindiniais eksperimentais validavimo tikslumas siekė apie 0.9706, todėl skirtumai geriau matėsi iš paklaidos ir mokymo laiko.
4. Paketinis gradientinis nusileidimas šiame darbe veikė gerokai greičiau už stochastinį gradientinį nusileidimą.
5. Mažiausia validavimo paklaida buvo gauta taikant `SGD` su `learning rate = 0.01`, o geriausia epocha buvo pasiekta jau 13-oje epochoje.
6. `SGD` metodui buvo svarbu modelį rinkti pagal validavimo epochą, nes paskutinė epocha nebūtinai davė geriausią variantą.
7. Pagal testo rezultatus vieno sigmoidinio neurono modelis gebėjo gana tiksliai spręsti dvejetainės klasifikacijos uždavinį.

## Priedai

Prie ataskaitos galima pridėti:

- svarbiausių kodo failų ištraukas
- `batch` ir `sgd` grafikų paveikslus
- eksperimentų lenteles
- testavimo rezultatų lentelę
