# Application Gradient Descent Graphique

Application Django complète pour l'implémentation de l'algorithme Gradient Descent pour la régression linéaire et logistique avec une interface utilisateur en français.

## Fonctionnalités

- **Import de données** : Import et analyse de fichiers CSV
- **Statistiques descriptives** : Affichage de statistiques complètes des données
- **Traitement des données** : Préparation des données (valeurs manquantes, normalisation, encodage)
- **Algorithme Gradient Descent** : Implémentation de la régression linéaire et logistique
- **Résultats visuels** : Affichage de la courbe d'apprentissage et des résultats de performance
- **Interface française** : Support complet de la langue française avec mise en page LTR

## Exigences système

- Python 3.8+
- pip (gestionnaire de paquets)

## Installation et configuration

### 1. Créer l'environnement virtuel

```bash
python -m venv venv
```

### 2. Activer l'environnement virtuel

**Windows :**
```bash
venv\Scripts\activate
```

**macOS/Linux :**
```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la base de données

```bash
cd gradient_descent_app
python manage.py migrate
```

### 5. Démarrer le serveur

```bash
python manage.py runserver
```

### 6. Ouvrir l'application

Ouvrez le navigateur et allez à : `http://localhost:8000/`

## Utilisation de l'application

### 1. Import des données
- Allez à la page "Importer des données"
- Choisissez un fichier CSV (moins de 5 mégaoctets)
- Assurez-vous que la dernière colonne est la variable cible

### 2. Examiner les statistiques
- Affichage des statistiques descriptives des données
- Vérification des valeurs manquantes et des types de données

### 3. Préparation des données
- Choisissez la méthode de traitement des valeurs manquantes
- Sélectionnez le type de normalisation requis
- Activez l'encodage des variables catégorielles si nécessaire

### 4. Exécution de l'algorithme
- Choisissez le type de modèle (linéaire ou logistique)
- Définissez le taux d'apprentissage et le nombre d'itérations
- Exécutez l'algorithme

### 5. Examiner les résultats
- Affichage des paramètres finaux
- Examen des performances du modèle
- Analyse de la courbe d'apprentissage

## Structure du projet

```
gradient_descent_app/
├── gradient_descent_app/
│   ├── settings.py          # Paramètres Django
│   ├── urls.py             # Routage URLs principal
│   └── wsgi.py             # Configuration WSGI
├── core/                   # Application principale
│   ├── views.py            # Logique des vues
│   ├── forms.py            # Formulaires Django
│   ├── utils.py            # Fonctions utilitaires
│   ├── urls.py             # Routage URLs
│   └── templatetags/       # Filtres de templates
├── templates/              # Templates HTML
│   ├── base.html           # Template de base
│   └── core/               # Templates de l'application
├── static/                 # Fichiers statiques
│   └── css/
└── media/                  # Fichiers utilisateur
    ├── uploads/            # Fichiers importés
    └── processed/          # Fichiers traités
```

## Algorithmes pris en charge

### Régression linéaire
- **Utilisation** : Prédiction de valeurs continues
- **Fonction de coût** : Mean Squared Error (MSE)
- **Équation** : `y = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ`

### Régression logistique
- **Utilisation** : Classification binaire
- **Fonction de coût** : Log-Likelihood
- **Équation** : `P(y=1) = 1 / (1 + e^(-(θ₀ + θ₁x₁ + ... + θₙxₙ)))`

## Traitement des données

### Valeurs manquantes
- **Suppression des lignes** : Supprime les lignes contenant des valeurs manquantes
- **Remplissage par la moyenne** : Remplace les valeurs manquantes par la moyenne de la colonne
- **Remplissage par une valeur constante** : Remplace les valeurs manquantes par une valeur spécifiée

### Normalisation
- **Normalisation standard** : `(x - moyenne) / écart-type`
- **Normalisation Min-Max** : `(x - min) / (max - min)`

### Encodage
- **One-Hot Encoding** : Conversion des variables catégorielles en variables numériques

## Conseils d'utilisation

### Choix du taux d'apprentissage
- **0.001** : Pour les données volumineuses ou complexes
- **0.01** : Pour les données normales (par défaut)
- **0.1** : Pour les petites données ou simples

### Nombre d'itérations
- **100-500** : Pour les tests rapides
- **1000** : Pour l'utilisation normale (par défaut)
- **5000+** : Pour les données complexes

## Dépannage

### Problèmes courants et solutions

1. **L'algorithme ne converge pas**
   - Réduisez le taux d'apprentissage
   - Assurez-vous d'appliquer la normalisation
   - Augmentez le nombre d'itérations

2. **Performances faibles**
   - Essayez un taux d'apprentissage différent
   - Vérifiez la qualité des données
   - Essayez une méthode de normalisation différente

3. **Erreur lors de l'import du fichier**
   - Assurez-vous que le fichier est au format CSV
   - Vérifiez la taille du fichier (moins de 5 mégaoctets)
   - Assurez-vous qu'il y a des données numériques

## Contribution

Les contributions sont les bienvenues ! Veuillez :
1. Faire un Fork du projet
2. Créer une nouvelle branche pour la fonctionnalité
3. Apporter les modifications
4. Envoyer une Pull Request

## Licence

Ce projet est sous licence MIT.

## Support

Pour obtenir de l'aide ou signaler des problèmes, veuillez créer une Issue dans le dépôt.