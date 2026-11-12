import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler,RobustScaler,MinMaxScaler
import joblib

data=pd.read_csv("transactions_TP2.csv")
print("Partie 1 — Explorer et vérifier les données")
print("Question 1 :")

print("les cinq premières lignes")
print(data.head())
print("le nombre de lignes et de colonnes =")
print(data.shape)
print("Les types de variables:")
print(data.dtypes)

print("Question 2 :")

pourcentage= (data.isnull().sum()/len(data))*100
T=pd.DataFrame({
    "Nombre de valeur manquante" : data.isnull().sum() ,
    "Pourcentage " : pourcentage 
})
table=T.sort_values(
    by="Pourcentage ",
    ascending = False
)

print (table)

print("Question 3 :")

nb_doublons=data["id_operation"].duplicated().sum()
print("les opérations répétées avec id_operation " ,nb_doublons)
supprime= data.drop_duplicates( subset="id_operation")
print( len(supprime))

print("Question 4 :")

print("histogramme")
plt.figure(figsize=(8,5))
plt.hist(supprime["montant_dh"].dropna(),bins=30,edgecolor="black")
plt.title("histogramme des montant")
plt.xlabel("Montant (DH)")
plt.ylabel("Frequence")
plt.show()
print("Boxplot")
plt.figure(figsize=(8,5))
sns.boxplot(x=supprime["montant_dh"])
plt.title("boxplot des montant")
plt.xlabel("Montant (DH)")
plt.show()

Q1=supprime["montant_dh"].quantile(0.25)
Q3=supprime["montant_dh"].quantile(0.75)
IQR= Q3-Q1
L= Q1-1.5*IQR
U= Q3+1.5*IQR
print("Q1 =",Q1)
print("Q3 =",Q3)
print("IQR=",IQR)
print("L = ",L)
print("U = ",U)
val_aberrant= supprime[
(supprime["montant_dh"]< L) |(supprime["montant_dh"]>U)
]
print("Nombre de montant extreme :", len(val_aberrant))
print("Quelques montants extremes :")
print(val_aberrant["montant_dh"].head(10))

print("Question 5 :")

print("Nombre de traction par class:")
print(supprime["fraude"].value_counts())
print("Calculer le pourcentage de chaque class")
pour=(supprime["fraude"].value_counts()/len(supprime))*100
print(pour)
pour=(supprime["fraude"].value_counts()[1]/len(supprime))*100
print("le pourcentage de fraude=1 :" ,pour)
print("Tracer le diagramme ")
classe= supprime["fraude"].value_counts().sort_index()
plt.figure(figsize=(7 ,5))
plt.bar(
    ["Non fraude (0)","fraude (1)"],
    classe.values,
    color=["blue","pink"],
    edgecolor="black"
)
plt.title("Repartition de transaction")
plt.xlabel("class")
plt.ylabel("Nombre de transaction")
plt.show()

print("Question 6 :")

print(supprime.columns)
print("les identifiants:")
print(supprime[["id_operation","id_client"]])
print("Resulta de l enquete :")
print(supprime["resultat_enquete"])

print("Partie 2 — Préparer les données sans fuite")
print("Question 7 :")

x=supprime.drop(columns=["id_operation","id_client","resultat_enquete","fraude"])
y=supprime["fraude"]
print(x.head(10))
print(y.head(10))

print("Question 8 :")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
print("les ligne dentrainement : " ,len(x_train))
print("les ligne de test : ",len(x_test))
pou_train=y_train.value_counts(normalize=True)*100
pou_test=y_test.value_counts(normalize=True)*100

comparaison=pd.DataFrame({
    "Groupe d'entrainement(80%)":pou_train,
    "Groupe de test (20%)":pou_test
})
print(comparaison)

print("Question 9 :")

print("Liste de variable numerique:")
print(x.select_dtypes(include=['int64','float64']).columns)
print("Liste de variable categorille:")
print(x.select_dtypes(include=['object','category']).columns)

print("Question 10 :")

print("remplacer les valeurs manquantes par la médiane")
val_num=['montant_dh', 'anciennete_mois', 'nb_ops_24h', 'heure']
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
x_train_num=num_pipe.fit_transform(x_train[val_num])
x_test_num=num_pipe.transform(x_test[val_num])
print(x_train_num)
print("Nouvelles dimensions :", x_train_num.shape)
print("Nouvelles dimensions de test :", x_test_num.shape)

print("Question 11 :")

print(" applique One-hot encoding pour les variable categorielle")
var_cat =['canal', 'appareil', 'region', 'international']
cat_pipe= Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot" , OneHotEncoder(handle_unknown="ignore"))
])
x_train_categ=cat_pipe.fit_transform(x_train[var_cat])
x_test_categ=cat_pipe.transform(x_test[var_cat])

print(x_train_categ.toarray()[:5])
print("Nouvelles dimensions dentrainement:", x_train_categ.shape)
print("Nouvelles dimensions de test :", x_test_categ.shape)

print("Question 12 :")

prep= ColumnTransformer([
    ("numeric", num_pipe,val_num),
    ("categorical",cat_pipe,var_cat)
])
x_train_prep=prep.fit_transform(x_train)
x_test_prep=prep.transform(x_test)
print(" dimensions dentrainement:", x_train_prep.shape)
print(" dimensions de test :", x_test_prep.shape)

print("Question 13 :")

val_num=['montant_dh', 'anciennete_mois', 'nb_ops_24h', 'heure']
num_robust_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])
x_train_robust_num=num_robust_pipe.fit_transform(x_train[val_num])
x_test_robust_num=num_robust_pipe.transform(x_test[val_num])

print(x_train_robust_num)
mantant_index=list(val_num).index('montant_dh')
print(x_train['montant_dh'].describe())
print(x_train_robust_num[:10 , mantant_index])

print("Nouvelles dimensions :", x_train_robust_num.shape)
print("Nouvelles dimensions de test :", x_test_robust_num.shape)

montant=x_train['montant_dh'].dropna()
montant_normal=montant.median()
montant_eleve=montant.max()
val=pd.DataFrame({
    'montant_dh' : [montant_normal,montant_eleve],
    'anciennete_mois':[0,0],
    'nb_ops_24h':[0,0],
    'heure':[0,0]
})
result=num_robust_pipe.transform(val[val_num])
print("une opération ordinaire :",montant_normal)
print("valeur transformée :" ,result[0,mantant_index])
print("une opération au montant très élevé:",montant_eleve)
print("valeur transformée :" ,result[1,mantant_index])

print("Question 15 :")

Q1=x_train["montant_dh"].quantile(0.25)
Q3=x_train["montant_dh"].quantile(0.75)
IQR= Q3-Q1
print("Q1 =",Q1)
print("Q3 =",Q3)
print("IQR=",IQR)

print("Question 16 :")

L= Q1-1.5*IQR
U= Q1+1.5*IQR
print("L = ",L)
print("U = ",U)
val_aberrant= x_train[
(x_train["montant_dh"]< L) |(x_train["montant_dh"]>U)
]
print("Nombre de montant extreme :", len(val_aberrant))
print("Quelques montants extremes :")
print(val_aberrant["montant_dh"].head(10))

print("Question 17 :")

origine_montant=x_train['montant_dh'].dropna()
limit_montant = origine_montant.clip(lower=L,upper=U)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(y=origine_montant)
plt.title("boxplot des montants origine")
plt.subplot(1,2,2)
sns.boxplot(y=limit_montant)
plt.title("boxplot des montants limité")
plt.show()

print("Question 18 :")

x_train=x_train[['montant_dh']]
x_test=x_test[['montant_dh']]

minmax=MinMaxScaler()
train_minmax=minmax.fit_transform(x_train)
test_minmax=minmax.transform(x_test)
print("minmax scaler d'entrainement :" , train_minmax[:5])
print("minmax scaler de test :", test_minmax[:5])

standar=StandardScaler()
train_standar=standar.fit_transform(x_train)
test_standar=standar.transform(x_test)
print("standar scaler de test :", test_standar[:5])
print("standar scaler d'entrainement :" , train_standar[:5])

train_robust=x_train_robust_num[:,mantant_index]
test_robust=x_test_robust_num[:, mantant_index]
print("robust scaler d'entrainement :" , train_robust[:5])
print("robust scaler de test :", test_robust[:5])

print("Question 19 :")
montants_choisis = pd.DataFrame({
    'montant_dh': [ Q1, x_train['montant_dh'].median(), Q3, x_train['montant_dh'].max()],
    'anciennete_mois':[0,0,0,0],
    'nb_ops_24h':[0,0,0,0],
    'heure':[0,0,0,0]
})
montants_choisis['MinMaxScaler'] = minmax.transform(
montants_choisis[['montant_dh']]
).flatten()
montants_choisis['StandardScaler'] = standar.transform(montants_choisis[['montant_dh']]).flatten()
result_robust = num_robust_pipe.transform(montants_choisis[val_num])
montants_choisis['RobustScaler'] = result_robust[:, mantant_index]
print(montants_choisis[
['montant_dh', 'MinMaxScaler', 'StandardScaler', 'RobustScaler']
].round(3))

print("Question 20 :")

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
sns.histplot(train_minmax.flatten() ,bins=30 , kde=True)
plt.title("MinMaxScaler")
plt.xlabel("Montant transformé")
plt.ylabel("Fréquence")
plt.subplot(1, 3 ,2)
sns.histplot(train_standar.flatten(), bins=30, kde=True)
plt.title("StandardScaler (Z-score)")
plt.xlabel("Montant transformé")
plt.ylabel("Fréquence")
plt.subplot(1, 3 ,3)
sns.histplot(train_robust.flatten(), bins=30, kde=True)
plt.title("RobustScaler")
plt.xlabel("Montant transformé")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

print("Question 23 :")

print("Avant le suréchantillonnage :")
print(y_train.value_counts().sort_index())
over = RandomOverSampler(random_state=42)
x_train_over, y_train_over = over.fit_resample(x_train, y_train)
print("Après le suréchantillonnage :")
print(y_train_over.value_counts().sort_index())
print("Nombre de lignes avant :", len(y_train))

print("Question 24 :")

print("Avant le sous-échantillonnage :")
print(y_train.value_counts().sort_index())
under= RandomUnderSampler(random_state=42)
x_train_under, y_train_under = under.fit_resample(x_train, y_train)
print("Après le sous-échantillonnage :")
print(y_train_under.value_counts().sort_index())
print("Nombre de lignes avant :", len(y_train))
print("Nombre de lignes conservées :", len(y_train_under))
print("Nombre de lignes supprimées :", len(y_train) - len(y_train_under))
print("Nombre de lignes après :", len(y_train_over))

print("Question 25 :")

X = np.array([
    [2, 4],
    [6, 8],
    [10, 2],
    [12, 3],
    [14, 4],
    [16, 5]
])
y = np.array([1, 1, 0, 0, 0, 0])
print("Avant SMOTE :")
print("Classe 0 :", sum(y == 0))
print("Classe 1 :", sum(y == 1))
smote = SMOTE(random_state=42, k_neighbors=1)
X_smote, y_smote = smote.fit_resample(X, y)
print("Après SMOTE :")
print("Classe 0 :", sum(y_smote == 0))
print("Classe 1 :", sum(y_smote == 1))
print("Observations synthétiques créées :")
print(X_smote[len(X):])

print("Question 27 : ")

tableau = pd.DataFrame({"Avant rééquilibrage": y_train.value_counts().sort_index(),
    "Après RandomOverSampler": y_train_over.value_counts().sort_index(),
    "Après RandomUnderSampler": y_train_under.value_counts().sort_index()
})
tableau.index.name = "Classe de fraude"
print(tableau)
print("Groupe de vérification :")
print(y_test.value_counts().sort_index())
print("Le groupe de vérification reste inchangé.")

print("Question 28 : ")

val_num=['montant_dh', 'anciennete_mois', 'nb_ops_24h', 'heure']
var_cat =['canal', 'appareil', 'region', 'international']
num_pipe_final = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipe_final= Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot" , OneHotEncoder(handle_unknown="ignore"))
])
prep_fina=ColumnTransformer([
    ('numeric',num_pipe_final,val_num),
    ('categorical',cat_pipe_final,var_cat)
])

print("Question 29 :")

prep_fina.fit(x_train)
x_train_final= prep_fina.transform(x_train)
x_test_final= prep_fina.transform(x_test)
print("nombre de colonne train:" ,x_train_final.shape)
print("nombre de colonne test:" ,x_test_final.shape)
print(np.isnan(x_train_final).sum())
print(np.isnan(x_test_final).sum())
print("verifie de meme nombre de colonne :" ,x_train_final.shape[1]==x_test_final.shape[1])

print("Question 30 :")

new_ligne=x_train.iloc[[0]].copy()
new_ligne['canal']='inconnu'
print(new_ligne)
new_ligne_transformee= prep_fina.transform(new_ligne)
print(new_ligne_transformee)
print(new_ligne_transformee.shape)

print("Question 31 :")

joblib.dump(prep_fina,'preprocessing.pkl')
prep_charge= joblib.load('preprocessing.pkl')
r1=prep_fina.transform(x_test.iloc[:5])
r2=prep_charge.transform(x_test.iloc[:5])
print("Resultat 1 ;", r1)
print("Resultat 2 ;", r2)
print("les resultats sont identique : ", (r1==r2).all())
print(prep_charge)
