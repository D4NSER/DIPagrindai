Perėjau pirmą dalį su paveiksliukais gana griežtai. Trumpa išvada: ji dabar atrodo normaliai studentiška, suprantama ir iš esmės pakankama ataskaitai.

Kas čia yra gerai:
- kodas paprastas, be perteklinių abstrakcijų;
- architektūros normalios studentiškos: `Conv -> activation -> pool`, tada `Flatten` ir `Linear`;
- yra tai, ko prašo užduotis:
  - duomenų paruošimas į `train/validation/test`
  - bent 3 architektūros
  - `dropout` tyrimas
  - `batch normalization` tyrimas
  - aktyvacijų tyrimas
  - optimizatorių tyrimas
  - `learning rate` tyrimas
  - `loss`, `accuracy`
  - `confusion matrix`
  - apie 30 testinių pavyzdžių su tikra ir prognozuota klase
- rezultatai geri:
  - architektūrų tyrime `deeper` ir `dropout_bn` pasiekia apie `0.9909` validavimo tikslumą
  - galutinis pasirinktas vaizdų modelis turi `test_accuracy = 0.9772`
- tai yra visai stiprus rezultatas tokiam paprastam modeliui ir tokiam datasetui.

Kas svarbu ataskaitai ir ką jau turi:
- duomenų splitai:
  - [image_train.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/data/processed/image_train.csv)
  - [image_validation.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/data/processed/image_validation.csv)
  - [image_test.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/data/processed/image_test.csv)
- architektūrų suvestinė:
  - [architectures.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/architectures.csv)
- hiperparametrų tyrimai:
  - [dropout_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/dropout_summary.csv)
  - [batch_norm_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/batch_norm_summary.csv)
  - [activation_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/activation_summary.csv)
  - [optimizer_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/optimizer_summary.csv)
  - [learning_rate_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/learning_rate_summary.csv)
- galutinis modelis:
  - [summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/summary.csv)
  - [epoch_metrics.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/epoch_metrics.csv)
  - [training_curves.png](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/training_curves.png)
  - [confusion_matrix.png](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/confusion_matrix.png)
  - [test_predictions.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/test_predictions.csv)
  - [selected_30_examples.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/selected_30_examples.csv)

Ar kodas remiasi priedo logika?
- Taip.
- Tiesiogiai matosi klasikinė CNN schema iš priedo: konvoliucijos, aktyvacija, `pooling`, pasirinktinai `dropout`, pasirinktinai `batch norm`.
- Nieko egzotiško ar sunkiai paaiškinamo nėra.
- Naudojami paprasti, lengvai apginami sprendimai.

Vienintelė svarbi pastaba:
- užduoties tekstas sako geriausią atvejį rinkti pagal mokymo tikslumą ir mokymo paklaidą, kas metodologiškai nėra idealu, bet tu dabar kaip tik taip ir darai. Tai gynime gali atrodyti keistai, bet formaliai su užduoties tekstu nesipyksta.

Mano vertinimas pirmos dalies:
- studentiška: taip
- suprantama: taip
- pakankama ataskaitai: taip
- rezultatai geri: taip
- per daug profesionali: ne

Jei nori, kitas žingsnis gali būti labai praktiškas: aš tau surašau būtent pirmos dalies ataskaitos karkasą punktas po punkto iš to, ką jau turi failluose, kad galėtum beveik tiesiog pildyti tekstą.



Perėjau pirmą dalį su paveiksliukais gana griežtai. Trumpa išvada: ji dabar atrodo normaliai studentiška, suprantama ir iš esmės pakankama ataskaitai.

Kas čia yra gerai:
- kodas paprastas, be perteklinių abstrakcijų;
- architektūros normalios studentiškos: `Conv -> activation -> pool`, tada `Flatten` ir `Linear`;
- yra tai, ko prašo užduotis:
  - duomenų paruošimas į `train/validation/test`
  - bent 3 architektūros
  - `dropout` tyrimas
  - `batch normalization` tyrimas
  - aktyvacijų tyrimas
  - optimizatorių tyrimas
  - `learning rate` tyrimas
  - `loss`, `accuracy`
  - `confusion matrix`
  - apie 30 testinių pavyzdžių su tikra ir prognozuota klase
- rezultatai geri:
  - architektūrų tyrime `deeper` ir `dropout_bn` pasiekia apie `0.9909` validavimo tikslumą
  - galutinis pasirinktas vaizdų modelis turi `test_accuracy = 0.9772`
- tai yra visai stiprus rezultatas tokiam paprastam modeliui ir tokiam datasetui.

Kas svarbu ataskaitai ir ką jau turi:
- duomenų splitai:
  - [image_train.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/data/processed/image_train.csv)
  - [image_validation.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/data/processed/image_validation.csv)
  - [image_test.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/data/processed/image_test.csv)
- architektūrų suvestinė:
  - [architectures.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/architectures.csv)
- hiperparametrų tyrimai:
  - [dropout_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/dropout_summary.csv)
  - [batch_norm_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/batch_norm_summary.csv)
  - [activation_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/activation_summary.csv)
  - [optimizer_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/optimizer_summary.csv)
  - [learning_rate_summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/learning_rate_summary.csv)
- galutinis modelis:
  - [summary.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/summary.csv)
  - [epoch_metrics.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/epoch_metrics.csv)
  - [training_curves.png](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/training_curves.png)
  - [confusion_matrix.png](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/confusion_matrix.png)
  - [test_predictions.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/test_predictions.csv)
  - [selected_30_examples.csv](/c:/Users/pedse/Desktop/DIPagrindai/lab3/results/task3_images/final_best_image_model/selected_30_examples.csv)

Ar kodas remiasi priedo logika?
- Taip.
- Tiesiogiai matosi klasikinė CNN schema iš priedo: konvoliucijos, aktyvacija, `pooling`, pasirinktinai `dropout`, pasirinktinai `batch norm`.
- Nieko egzotiško ar sunkiai paaiškinamo nėra.
- Naudojami paprasti, lengvai apginami sprendimai.

Vienintelė svarbi pastaba:
- užduoties tekstas sako geriausią atvejį rinkti pagal mokymo tikslumą ir mokymo paklaidą, kas metodologiškai nėra idealu, bet tu dabar kaip tik taip ir darai. Tai gynime gali atrodyti keistai, bet formaliai su užduoties tekstu nesipyksta.

Mano vertinimas pirmos dalies:
- studentiška: taip
- suprantama: taip
- pakankama ataskaitai: taip
- rezultatai geri: taip
- per daug profesionali: ne

Jei nori, kitas žingsnis gali būti labai praktiškas: aš tau surašau būtent pirmos dalies ataskaitos karkasą punktas po punkto iš to, ką jau turi failluose, kad galėtum beveik tiesiog pildyti tekstą.