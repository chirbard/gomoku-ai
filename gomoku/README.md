https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/

https://medium.com/@samgill1256/reinforcement-learning-in-chess-73d97fad96b3

https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542/

https://medium.com/@sulbha.jindal/llm-finetuning-with-rlhf-part-2-d2cbc5453762


[[0,-1,-1],[1,1,0],[0,0,0]]
[[x,x,x],[x,x,x],[x,x,x]]

/set system "you are a ticktacktoe player. the board is given in as. [[x,x,x],[x,x,x],[x,x,x]]. You are player 1 your oponent is -1. Give me an index of the position where to place the new piece"


lahendus githubis: https://github.com/chirbard/gomoku-ai

# Märkus 
Minu lahendus kasutab Pytorchi, mitte siin kursusel kasutatud Tensorflowd, sest ma pole oma arvutis graafikakaardiga Tensorflowd tööle saanud. Ja ma eelistan tundide kaupa treenida mudelit oma arvutis mitte google colabis.

# Erinevad mudelid
Esmalt kasutasin allikat 1, et luua deep Q-learning algoritmiga tripstrapstrulli agent ja siis suurendasin seda meie 15x15 gomoku jaoks. Kuna gomokus on nii palju seise, siis Q-learning ei tööta.
Siis tegin lihtsa minimax algoritmi, aga see töötas väga halvasti.

# Lahendus - AlphaZero sarnane Monte Carlo Tree Search
Githubis on treenimise kood ja kogu ajalugu, mis muudatusi ma teinud olen. Palju algset koodi on tehtud koos ChatGPT'ga ja siis on arhitektuuri täiendatud allikaga 2. Lisaks on juurde lisatud funktsioonid:
- kõrgem hinnang mängudele, mis lõppevad kiiremini(mängulaud jääb tühjemaks)
- agent saab nuppe asetada ainult olemasolevate nuppude kõrvale
- Weights and Biases integratsioon treenimisel (tekib hea ülvaade ajaloost)
![alt text](image.png)
Weights and Biases ülevaade treenimisest

## treenimise seletus
Treenimise faili esitatud pole, aga see on githubis.
- treenimisel mudel mängib iseenda vastu
- algoritm võtab mängu hetkeseisu ja loob sellest mitmeid simulatsioone
- simulatsiooni sees kasutab see Monte Carlo Tree Searchi. See tähendab, et see genereerib rekursiivselt järgmiseid olekuid ja kasutab närvivõrku, et hinnata kui hea on hetkeseis ja annab igale järgmisele olekule hinnangu. See algoritm otsib keskteed uute lahenduste otsimise ja teatud lahenduste ära kasutamise vahel. Simuleerides ei vali see alati siiamani kõige parema väärtusega liigutust vaid uurib ka teisi liigutusi.
- Nende simulatsioonidega tekib sõnastik iga võimaliku käiguga ja hinnanguga kui hea see liigutus on. Hetkel algoritm valib alati kõige parema hinnanguga liigutuse (temperature 1.0).
- Seda liigutust ta kasutab et jätkata enda vastu mängimist.
- Mudel mängib mängu lõpuni. Saab teada kumb mängija võitis ja lisab sellele mängule auhinna selle järgi kui pikkalt see mäng kestis.
- Iga 10 mängu tagant võtab mudel need mängud kokku ja nende auhindade põhjal treenib mudel ennast 200 epohhi.
- Seda 10 mängu mängimist ja treenimst teeb mudel mitemid iteratsioone.
- Iga kahe iteratsiooni tagant salvestatakse vaheseis ja iga iteratsioon saadetakse info Weights and Biases keskkonda, et seal tekiks ülevaade.


## Närvivõrgu arhitektuuri kirjeldus
Algne kava on genreeritud ChatGPT poolt ja seda on muudetud allikas 2 põhjal

See närvivõrk genereerib kaks väljundit:
- Kui hea on iga võimalik järgmine käik
- Kui hea on hetkeseis

Kasutusel on kihid:
- Covolution kihid selle jaoks, et leida seoseid ja mingeid kindlaid omadusi mängulauast ära tunda
- Max Pooling kihid, et vähendada närvivõrgu keerukust
- Dropout kihid, et vähendada neuroneid ja sellega vältida ülesobitamist



allikad:
1) https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542/
2) https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/