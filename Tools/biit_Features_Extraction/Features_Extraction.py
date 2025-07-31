import os
import sys
import subprocess

class Tool():
    # Nom affiché dans BioImageIT
    name = "Features Extraction"

    # Description visible pour l'utilisateur
    description = "This tool extracts features from a 4D image sequence (T,Z,Y,X) using various methods."

    # Catégorie dans laquelle l'outil apparaîtra
    categories = ['Astroca', 'Features']

    # Environnement conda spécifique
    environment = 'astroca-env'

    # Dépendances (tu peux adapter si besoin)
    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas', 'openpyxl'],
        pip=[]
    )

    # Définition des entrées attendues
    inputs = [
        dict(name='events_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path',
             autoColumn=True),
        dict(name='image_amplitude', help='Chemin vers le fichier .tif 4D (T,Z,Y,X) représentant l\'amplitude de l\'image.', required=True, type='Path'),
        dict(name='ids_events', help="Nombre d'évènements détectés", required=True, type='Path', autoColumn=True),
        dict(name='voxel_size_x', help='Taille du voxel en X en µm', required=True, type='Float', default=0.1025),
        dict(name='voxel_size_y', help='Taille du voxel en Y en µm', required=True, type='Float', default=0.1025),
        dict(name='voxel_size_z', help='Taille du voxel en Z en µm', required=True, type='Float', default=0.1344),
        dict(name='threshold_median_localized', help='Seuil de la médiane localisée pour la détection des caractéristiques.', required=True, type='Float', default=4.0),
        dict(name='volume_localized', help='Volume localisé pour la détection des caractéristiques.', required=True, type='Float', default=0.0434),
    ]

    outputs = [
        dict(name='features', help='Caractéristiques extraites de l\'image.', default='features_extracted.csv',
             type='Path')
    ]

    def setup_environment(self):
        try:
            import astroca
            print("Package astroca déjà disponible")
            return
        except ImportError:
            print("Installation du package astroca depuis GitHub...")

        repo_url = "git+ssh://git@github.com/audigiem/AstrocytesSegmentation.git@bioimageIT_src"
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", repo_url
            ])
            print("Package astroca installé avec succès")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Échec de l'installation pip du package astroca : {e}")

    def processData(self, args):
        """
        Traite les données en extrayant des caractéristiques à partir d'une séquence d'images 4D.
        """
        # Configuration de l'environnement
        self.setup_environment()

        # Import des modules après installation
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.features.featuresComputation import save_features_from_events
        except ImportError as e:
            raise ImportError("Impossible d'importer les modules nécessaires. "
                              "Vérifiez que le module 'astroca' est présent.") from e

        events_path = str(args.events_image)
        # Vérification du fichier d'entrée
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {events_path}")
        events = load_data(events_path)

        amplitude_path = str(args.image_amplitude)
        if not os.path.exists(amplitude_path):
            raise FileNotFoundError(f"Le fichier d'image dynamique est introuvable : {amplitude_path}")
        amplitude = load_data(amplitude_path)

        nb_events_path = str(args.ids_events)
        if not os.path.exists(nb_events_path):
            raise FileNotFoundError(f"Le fichier d'identifiants d'événements est introuvable : {nb_events_path}")
        with open(nb_events_path, 'r') as f:
            lines = f.readlines()
            ids_events = int(lines[-1].strip())

        voxel_size_x = float(args.voxel_size_x)
        voxel_size_y = float(args.voxel_size_y)
        voxel_size_z = float(args.voxel_size_z)
        threshold_median_localized = float(args.threshold_median_localized)
        volume_localized = float(args.volume_localized)
        output_feature = args.features

        param_features_extraction = {
            'features_extraction': {
                'ids_events': ids_events,
                'voxel_size_x': voxel_size_x,
                'voxel_size_y': voxel_size_y,
                'voxel_size_z': voxel_size_z,
                'threshold_median_localized': threshold_median_localized,
                'volume_localized': volume_localized
            },
            'save': {'save_features': 1},
            'paths': {'output_dir': os.path.dirname(output_feature) + "/"}
        }
        save_features_from_events(events, ids_events, amplitude, param_features_extraction)

    def processAllData(self, argsList):
        for args in argsList:
            try:
                self.processData(args)
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {args.events_image}: {e}")
                continue
