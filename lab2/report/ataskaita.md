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

1. Jei žaliavinio duomenų failo nėra, jis automatiškai parsiunčiamas iš UCI saugyklos.
2. Nuskaitytas pradinis `.data` failas.
3. Pašalintas `sample_code_number` stulpelis.
4. Ištrintos eilutės, kuriose buvo `?`.
5. Klasių žymės pakeistos į `0` ir `1`.
6. Duomenų eilutės atsitiktinai permaišytos.

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

Papildomai prieš mokymą požymiai buvo standartizuoti pagal mokymo aibės vidurkius ir standartinius nuokrypius. Tas pats transformavimas po to buvo pritaikytas validavimo ir testavimo aibėms. Šis žingsnis nebuvo tiesiogiai reikalaujamas užduoties tekste, tačiau buvo taikytas tam, kad mokymas būtų stabilesnis ir kad skirtingo mastelio požymiai neįtakotų svorių atnaujinimo nevienodai.

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

Paklaidai skaičiuoti buvo naudota vidutinė kvadratinė paklaida pagal formulę:

`MSE = 0.5 * mean((t - y)^2)`

kur:

- `t` – tikroji klasė
- `y` – neurono išėjimas

Tokia forma buvo pasirinkta todėl, kad ji atitinka užduoties skaidrėse pateikiamą kvadratinės paklaidos logiką ir leidžia tiesiogiai lyginti mokymo, validavimo ir testavimo paklaidas net tada, kai aibių dydžiai skiriasi. Klasifikavimo kokybei vertinti buvo naudotas tikslumas (`accuracy`).

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

- `w1 = 0.240772`
- `w2 = 0.336747`
- `w3 = 0.142702`
- `w4 = 0.374266`
- `w5 = 0.308551`
- `w6 = 0.347500`
- `w7 = 0.255242`
- `w8 = 0.300155`
- `w9 = 0.134334`
- `bias = -0.216236`

Kadangi šio varianto geriausia validavimo epocha sutapo su paskutine epocha, buvo gauta:

- mokymo paklaida: `0.021965`
- validavimo paklaida: `0.022595`
- mokymo tikslumas: `0.970696`
- validavimo tikslumas: `0.970588`

Testavimo rezultatai, naudojant pagal validaciją pasirinktą šio varianto modelį:

- testo paklaida: `0.024702`
- testo tikslumas: `0.942029`
- mokymo laikas: `0.084069 s`

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
| batch | 0.01 | 200 | 0.061419 | 0.970588 | 0.054897 | 0.942029 | 0.028771 |
| batch | 0.05 | 200 | 0.022595 | 0.970588 | 0.024702 | 0.942029 | 0.037985 |
| batch | 0.10 | 200 | 0.016431 | 0.970588 | 0.020730 | 0.942029 | 0.036963 |
| sgd | 0.01 | 149 | 0.010629 | 0.970588 | 0.020719 | 0.942029 | 1.138135 |
| sgd | 0.05 | 200 | 0.010621 | 0.970588 | 0.021246 | 0.956522 | 1.164935 |
| sgd | 0.10 | 200 | 0.010433 | 0.970588 | 0.021635 | 0.956522 | 1.137450 |

Pilna eksperimentų lentelė pateikta faile `lab2/results/task3/experiment_summary.csv`.

### Paklaidos priklausomybė nuo epochų

Paklaidų kitimas pagal epochas buvo pavaizduotas grafikuose. Iš `batch` grafiko matyti, kad tiek mokymo, tiek validavimo paklaida nuosekliai mažėja. `SGD` atveju paklaida mažėja greitai, tačiau kreivės yra labiau banguotos, nes svoriai atnaujinami po kiekvieno įrašo.

Čia reikia įterpti paveikslus:

- `lab2/results/task3/batch_lr_0_05_curves.png`
- `lab2/results/task3/sgd_lr_0_05_curves.png`

### Tikslumo priklausomybė nuo epochų

Tikslumo kreivės rodo, kad abu metodai gana greitai pasiekia aukštą validavimo tikslumą. `Batch` metodo tikslumas auga tolygiau, o `SGD` dažnai labai greitai pasiekia aukštą reikšmę jau pačiose pirmose epochose.

### Learning rate įtaka

Iš rezultatų matyti, kad `batch` metodui didesnis mokymosi greitis šiame darbe buvo naudingas. Mažiausia `test loss` reikšmė `batch` grupėje gauta su `learning rate = 0.1`.

`SGD` atveju visi trys variantai davė tą patį geriausią validavimo tikslumą, tačiau mažiausia validavimo paklaida buvo gauta su `learning rate = 0.1`. Taip pat matyti, kad mažesnio `learning rate = 0.01` atveju geriausia epocha buvo rasta anksčiau, dar nepasiekus paskutinės epochos.

### Batch ir SGD palyginimas

Pagal validavimo tikslumą abu metodai šiame darbe pasirodė labai panašiai, nes visais pagrindiniais atvejais buvo gautas `0.970588` validavimo tikslumas.

Tačiau pagal mokymo laiką skirtumas buvo labai aiškus:

- `batch` metodas veikė maždaug `0.029 - 0.038 s`
- `sgd` metodas veikė maždaug `1.138 - 1.165 s`

Taigi šiame darbe `batch` buvo ženkliai greitesnis už `SGD`. Vis dėlto mažiausia validavimo paklaida buvo gauta su `SGD`, todėl galutinis modelio pasirinkimas buvo daromas ne pagal laiką, o pagal validavimo rezultatus.

### Geriausias variantas

Pagal užduoties logiką geriausias variantas turi būti parenkamas pagal validavimo rezultatus. Todėl galutiniu modeliu buvo laikomas tas variantas, kuris pasiekė didžiausią validavimo tikslumą ir mažiausią validavimo paklaidą.

Galutiniu pasirinktu variantu tapo:

- metodas: `sgd`
- `learning rate = 0.1`
- bendras mokymo epochų skaičius: `200`
- pasirinkta geriausia epocha pagal validaciją: `200`

Šis variantas davė:

- mokymo paklaidą pasirinktoje epochoje: `0.008201`
- validavimo paklaidą: `0.010433`
- validavimo tikslumą: `0.970588`
- mokymo tikslumą pasirinktoje epochoje: `0.979853`
- testo paklaidą: `0.021635`
- testo tikslumą: `0.956522`

Kadangi keli variantai pasiekė tą patį validavimo tikslumą, galutinis pasirinkimas buvo daromas pagal mažesnę validavimo paklaidą. Galutinio varianto suvestinė pateikta faile `lab2/results/final_best_model/summary.json`.

### Pastaba apie best ir last epoch

Atliekant tyrimą buvo saugomos dvi reikšmių grupės:

- `selected` arba `best` reikšmės – tos, kurios atitinka geriausią epochą pagal validavimo rezultatus
- `final` arba `last` reikšmės – tos, kurios gautos paskutinėje mokymo epochoje

Galutiniam pasirinktam variantui šios reikšmės sutapo, nes geriausia validavimo epocha buvo paskutinė, t. y. 200-oji epocha. Tačiau kitų konfigūracijų atveju jos nebūtinai sutapo, todėl ataskaitoje jas reikia interpretuoti atskirai.

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
5. Mažiausia validavimo paklaida buvo gauta taikant `SGD` su `learning rate = 0.1`.
6. Kai kurioms `SGD` konfigūracijoms geriausia epocha nesutapo su paskutine epocha, todėl modelio pasirinkimas pagal validaciją buvo svarbus.
7. Pagal testo rezultatus vieno sigmoidinio neurono modelis gebėjo gana tiksliai spręsti dvejetainės klasifikacijos uždavinį.

## Priedai

Prie ataskaitos galima pridėti:

- svarbiausių kodo failų ištraukas
- `batch` ir `sgd` grafikų paveikslus
- eksperimentų lenteles
- testavimo rezultatų lentelę
