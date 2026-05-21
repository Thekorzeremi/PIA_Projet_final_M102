Créer un environnement virtuel pour python avec :
```sh
python -m venv <path>
```

Puis exécuter l'environnement :

```sh
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".\Projet_Final_M102-PIA\.venv\Scripts\Activate.ps1")   
```

Enfin installer les paquets :
```
python -m pip install -r ./requirements.txt
```