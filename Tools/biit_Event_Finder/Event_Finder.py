import os
import sys
import subprocess

class Tool():
    # Nom affiché dans BioImageIT
    name = "Calcium Active Voxel Finder"

    # Description visible pour l'utilisateur
    description = "Apply a connex component analysis to a 4D image sequence (T,Z,Y,X) to detect active voxels based on dynamic changes."

    # Catégorie dans laquelle l'outil apparaîtra
    categories = ['Astroca', 'Event Detection']

    # Environnement conda spécifique
    environment = 'astroca-env'

    # Dépendances (tu peux adapter si besoin)
    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas', 'numba', 'matplotlib'],
        pip=[]
    )

    # Définition des entrées attendues
    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path',
             autoColumn=True),
        dict(name='threshold_size_3d', help='Taille minimale des composants connexes en 3D pour être considérées actives.',
             default=400, type='Integer'),
        dict(name='threshold_correlation', help='Seuil de corrélation pour détecter les changements dynamiques.',
             default=0.6, type='Float'),
        dict(name='threshold_size_3d_remove',
             help='Taille minimale des composants connexes en 3D pour être retirées de la détection.',
             default=20, type='Integer'),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='calciumEvents.tif',
             type='Path'),
        dict(name='ids_events', help='Identifiants des événements détectés (de 1 à ids_events)', default='data.txt', type='Path')
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
        Traite les données en appliquant la fermeture d'espace.

        Paramètres :
            args : objet avec les attributs nécessaires pour l'image d'entrée, le rayon et le mode de bordure

        Retour :
            None
        """
        # Configuration de l'environnement
        self.setup_environment()

        # Import des modules après installation
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.events.eventDetector import detect_calcium_events_opti
        except ImportError as e:
            raise ImportError("Impossible d'importer les modules nécessaires. "
                              "Vérifiez que le module 'astroca' est présent.") from e

        av_path = str(args.input_image)
        # Vérification du fichier d'entrée
        if not os.path.exists(av_path):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {av_path}")
        data = load_data(av_path)

        threshold_size_3d = int(args.threshold_size_3d)
        threshold_correlation = float(args.threshold_correlation)
        threshold_size_3d_remove = int(args.threshold_size_3d_remove)

        params_event_detection = {
            'save' : {'save_events' : 0},
            'paths' : {'output_dir': None},
            'events_extraction' : {
                'threshold_size_3d': threshold_size_3d,
                'threshold_corr': threshold_correlation,
                'threshold_size_3d_removed': threshold_size_3d_remove
            }
        }

        output_image = args.output_image

        id_connections, ids_events = detect_calcium_events_opti(data, params_event_detection)

        # open data.txt file to write ids_events
        ids_events_path = str(args.ids_events)
        if not os.path.exists(os.path.dirname(ids_events_path)):
            os.makedirs(os.path.dirname(ids_events_path))
        with open(ids_events_path, 'w') as f:
            f.write(f"{ids_events}")

        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-4]
        export_data(id_connections, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name)

    def processAllData(self, argsList):
        if len(argsList) > 1:
            for args in argsList:
                try:
                    self.processData(args)
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {args.input_image}: {e}")
                    continue

