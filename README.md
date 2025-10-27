#  Email Spam Classifier - Naive Bayes Implementation

Un système intelligent de détection de spam utilisant l'algorithme Naive Bayes Multinomial et le traitement automatique du langage naturel (NLP).

##  Description

**Email Spam Classifier** est un système de machine learning qui utilise l'algorithme Naive Bayes Multinomial pour classifier automatiquement les emails en **SPAM** ou **HAM** (non-spam). Le projet démontre une implémentation pratique du NLP et des techniques de classification textuelle.

##  Fonctionnalités

###  Gestion des Données
- **Extraction automatique** du corps des emails
- **Support multiple encodages** (latin1, utf-8)
- **Prétraitement intelligent** des fichiers email
- **Structure de données organisée** avec Pandas

###  Traitement du Texte
- **Vectorisation** avec CountVectorizer
- **Gestion des stop words** personnalisable
- **Création de features** basées sur la fréquence des mots
- **Transformation texte→vecteur** optimisée

###  Modèle de Machine Learning
- **Algorithme Naive Bayes Multinomial** 
- **Entraînement supervisé** sur données labellisées
- **Prédictions en temps réel**
- **Interface de test simple**

###  Pipeline Complet
- **Chargement** des données d'entraînement
- **Prétraitement** et vectorisation
- **Entraînement** du modèle
- **Évaluation** sur nouveaux emails

##  Architecture du Modèle

### Stack Technologique
```python
# Core ML Libraries
scikit-learn >= 1.0.0        
pandas >= 1.5.0              
numpy >= 1.21.0              

# NLP Components
CountVectorizer              # Text to Numerical Features
MultinomialNB                # Naive Bayes Classifier
