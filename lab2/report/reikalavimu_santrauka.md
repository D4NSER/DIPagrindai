# Lab2 reikalavimų santrauka

## Kas yra privaloma iš pačios užduoties

### 1. Duomenų paruošimas

Privaloma padaryti:

- parsisiųsti `breast-cancer-wisconsin.data`
- klasių žymes pakeisti į `0` ir `1`
- pašalinti nereikalingus stulpelius, pvz. `ID`
- ištrinti eilutes, kuriose yra `?`
- atsitiktinai permaišyti įrašus

### 2. Vieno sigmoidinio neurono realizacija

Privaloma:

- naudoti vieną neuroną su sigmoidine aktyvacijos funkcija
- neurono išėjimą interpretuoti kaip reikšmę intervale `(0; 1)`
- spėjamą klasę gauti apvalinant į `0` arba `1`

### 3. Turi būti realizuoti abu mokymo metodai

Privaloma turėti:

- paketinį gradientinį nusileidimą (`batch gradient descent`)
- stochastinį gradientinį nusileidimą (`SGD`)

### 4. Turi būti trys duomenų aibės

Privaloma:

- `train`
- `validation`
- `test`

Leidžiamas ir rekomenduojamas santykis:

- `80:10:10`

### 5. Programos rezultatuose privalo būti

Kiekvienam metodui privaloma pateikti:

- 9 svoriai
- 1 `bias`
- paklaida po kiekvienos epochos mokymo duomenims
- paklaida po kiekvienos epochos validavimo duomenims
- testavimo paklaida
- klasifikavimo tikslumas po kiekvienos epochos mokymo duomenims
- klasifikavimo tikslumas po kiekvienos epochos validavimo duomenims
- testavimo tikslumas
- mokymo laikas

### 6. Tyrimo dalyje privaloma ištirti

- kaip paklaida priklauso nuo epochų mokymo ir validavimo duomenims
- kaip klasifikavimo tikslumas priklauso nuo epochų mokymo ir validavimo duomenims
- kaip rezultatai priklauso nuo skirtingų `learning rate`
- kaip rezultatus veikia `batch` ir `SGD`
- kaip `batch` ir `SGD` veikia mokymo laiką

Papildoma sąlyga:

- `learning rate` turi būti intervale `(0, 1)`
- reikia bent 3 skirtingų `learning rate` reikšmių

### 7. Ataskaitoje privaloma aprašyti

- kokie duomenys naudoti
- kiek yra eilučių ir požymių
- kaip duomenys padalinti į `train`, `validation`, `test`
- kokia kiekvienos aibės paskirtis
- kaip buvo parinkti pradiniai svoriai
- kas yra `batch gradient descent`
- kas yra `SGD`
- kas kiekvienu atveju yra epocha
- tyrimų rezultatus
- geriausio varianto svorius
- geriausio varianto epochų skaičių
- paskutinės epochos mokymo ir validavimo paklaidas
- paskutinės epochos mokymo ir validavimo tikslumus
- testavimo paklaidą
- testavimo tikslumą
- kiekvieno testavimo įrašo prognozuotą ir tikrą klasę
- išvadas

Papildoma svarbi sąlyga:

- jei buvo naudotas DI įrankis, tai turi būti pažymėta ne pabaigoje, o ten, kur tas rezultatas įdėtas

## Kas konkrečiai reikalinga iš priedo

Priede yra daug bendros teorijos. Šitam darbui realiai tiesiogiai reikalingos tik kelios dalys.

### 1. Priedo 23 psl.

Svarbi mintis:

- neuronas skaičiuoja `a = Σ(w_k * x_k) + b`
- yra svoriai ir `bias`
- svorius galima inicializuoti mažomis atsitiktinėmis reikšmėmis

Ką pasiimti į darbą:

- mums užtenka paprasto atsitiktinio svorių inicializavimo
- nereikia Xavier ar He inicializacijos, nes čia tik vienas neuronas

### 2. Priedo 26-28 psl.

Tai svarbiausia dalis apie `batch` ir `SGD`.

Ką būtinai pasiimti:

- `batch`: vienas svorių atnaujinimas po visos mokymo aibės
- `SGD`: vienas svorių atnaujinimas po vieno įrašo
- `batch` paklaidos kreivė būna lygesnė
- `SGD` paklaidos kreivė būna triukšmingesnė
- `SGD` dažniausiai turi daugiau svorių atnaujinimų per epochą

Ką tai reiškia ataskaitai:

- būtinai reikia paaiškinti, kuo tie metodai skiriasi
- verta paminėti, kad `batch` kreivės paprastai lygesnės, o `SGD` labiau banguotos

### 3. Priedo 42 psl.

Svarbi mintis:

- modelio kokybė vertinama skaičiuojant paklaidą
- paklaida gali būti vertinama mokymo, validavimo ir testavimo duomenims

Ką pasiimti:

- ataskaitoje reikia aiškiai parodyti `train loss`, `validation loss`, `test loss`

### 4. Priedo 43 psl.

Tai viena svarbiausių skaidrių.

Ką būtinai pasiimti:

- `train set` naudojama modelio mokymui
- `validation set` naudojama modelio ir hiperparametrų pasirinkimui
- `test set` naudojama galutiniam nešališkam įvertinimui

Ką tai reiškia mūsų darbui:

- `learning rate` turi būti parenkamas žiūrint į validavimo rezultatus
- testavimo duomenys neturi būti naudojami hiperparametrų rinkimui

### 5. Priedo 47 psl.

Svarbi mintis:

- pagal epochų kreives galima matyti `overfitting`
- mokymo tikslumas gali kilti, o validavimo ar testavimo kokybė po tam tikro taško blogėti

Ką pasiimti:

- grafikai pagal epochas yra ne šiaip formalumas, jie reikalingi tam, kad būtų matyti modelio elgsena
- ataskaitoje verta paminėti, ar matėsi per didelis prisitaikymas (`overfitting`)

### 6. Priedo 48 psl.

Labai svarbu tyrimų daliai.

Ką būtinai pasiimti:

- `learning rate` yra hiperparametras
- per mažas `learning rate` lemia lėtą mokymąsi
- per didelis `learning rate` gali pabloginti konvergavimą
- reikia eksperimentais parodyti, kuris `learning rate` veikia geriausiai

## Kas priede nėra būtina šitam darbui

Šitų dalykų daryti nereikia, nebent pats nori papildomai:

- `mini-batch gradient descent`
- `momentum`
- `Adam`
- `RMSProp`
- gilūs tinklai
- Xavier inicializacija
- He inicializacija
- sudėtinga reguliacija
- sudėtingos architektūros

Tai yra bendra teorija, bet ne šitos užduoties reikalavimas.

## Esminė praktinė išvada

Jei viską suspausti iki minimumo, tai darbui realiai reikia tik šitų dalykų:

1. Išvalyti ir paruošti duomenis.
2. Parašyti vieną sigmoidinį neuroną.
3. Apmokyti jį dviem būdais: `batch` ir `SGD`.
4. Turėti `train`, `validation`, `test`.
5. Skaičiuoti `loss`, `accuracy`, `time`.
6. Padaryti epochų grafikus.
7. Palyginti bent 3 `learning rate`.
8. Geriausią variantą rinkti pagal validavimo rezultatus.
9. Testą naudoti tik galutiniam įvertinimui.

